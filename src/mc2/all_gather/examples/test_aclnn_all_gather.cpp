/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_all_reducer_direct.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <mpi.h>
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "../op_host/op_api/aclnn_all_gather.h"
#include "tilexr_api.h"
#include "tilexr_comm.h"
#include "tilexr_internal.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize_ = 1;
    for (auto i : shape) {
        shapeSize_ *= i;
    }
    return shapeSize_;
}

const uint32_t MACHINE_NUM = 1;
const uint32_t NPU_NUM_PER_MACHINE = 2;

const uint32_t ITER_NUM = 10;
constexpr int RANK_DIM = MACHINE_NUM * NPU_NUM_PER_MACHINE;
const std::vector<int64_t> aShape = {64, 1};
const std::vector<int64_t> outShape = {64*RANK_DIM, 1};
const long long shapeSize = GetShapeSize(aShape);
const long long outShapeSize = GetShapeSize(outShape);

struct TestData {
    std::vector<std::vector<op::fp16_t>> input;
    std::vector<op::fp16_t> out;
    TestData() {
        input.resize(RANK_DIM);
        for (auto &v : input) v.resize(shapeSize);
        out.resize(outShapeSize);
    }
};



template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CompareVector(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2)
{
    const float tolerance = 0.001f; // 千分之一
    for (size_t i = 0; i < vec1.size(); ++i) {
        float a = static_cast<float>(vec1[i]);
        float b = static_cast<float>(vec2[i]);

        float diff = std::fabs(a - b);
        float maxAbs = std::max(std::fabs(a), std::fabs(b));
        if (maxAbs > 1e-6f) {
            float relativeError = diff / maxAbs;
            if (relativeError > tolerance) {
                return 1;
            }
        } else {
            if (diff > tolerance) {
                return 1;
            }
        }
    }
    return 0;
}

template <class T>
void print_vector(const std::vector<T>& vec,
                  std::ostream& os = std::cout)
{
    std::cout << '{';
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << (float)vec[i];
        if (i + 1 != vec.size())
            std::cout << ", ";
    }
    std::cout << '}';
    std::cout << '\n';
}

struct Args {
    int localRankId;
    int globalRankId;
    TileXR::TileXRComm* comm;
    HcclComm hcclCommP2P;
    aclrtStream stream;
    aclrtContext context;
  };

int LaunchOneThreadAllGather(Args &args, TestData &testData)
{
    int ret;
//    int ret = aclrtSetCurrentContext(args.context);
//    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] rank = %d, LaunchOneThreadAllGather\n", args.globalRankId);

    char hcomNameP2P[128] = {0};
    ret = HcclGetCommName(args.hcclCommP2P, hcomNameP2P);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] P2P HcclGetCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomNameP2P = %s, stream = %p\n", args.globalRankId, hcomNameP2P, args.stream);

    void *deviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    aclTensor *out = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    std::vector<op::fp16_t> hostData(shapeSize, 0);
    std::vector<op::fp16_t> outHostData(outShapeSize, 0);

    // 根据随机生成的测试数据填充host侧输入
    std::copy(testData.input[args.globalRankId].begin(), testData.input[args.globalRankId].end(), hostData.begin());

    // 启动初始化 SDMA 任务的 aicpu kernel
	void *tileXrCtx;
	ret = TileXrInit(args.hcclCommP2P, args.stream, &tileXrCtx);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] TileXrInit failed. ret: %d\n", ret); return ret);
     
    for (int iter=0; iter < ITER_NUM; iter++) {
        std::vector<op::fp16_t> outputData(outShapeSize, 0);
        ret = AllGather(hostData.data(), outputData.data(), shapeSize, aclDataType::ACL_FLOAT16, args.comm, args.hcclCommP2P, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Allgather failed. ret = %d \n", ret););
        // 比较AllReduceDirect结果
        if(args.globalRankId == 0){
               std::cout<<"outputData data" << std::endl;
               print_vector(outputData);
               std::cout<<"testData.out" << std::endl;
               print_vector(testData.out);
        }
        ret = CompareVector(outputData, testData.out);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] output compare failed. dev %d \n", args.globalRankId));
        if (ret == ACL_SUCCESS) {
            LOG_PRINT("[INFO] device_%d aclnnAllReduceDirect golden compare successfully.\n", args.globalRankId);
        }
        if (input != nullptr) {
            aclDestroyTensor(input);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
    }

    auto hcclRet = HcclCommDestroy(args.hcclCommP2P);
    CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] P2P HcclCommDestroy failed. ret = %d \n", hcclRet));

    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.localRankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return 0;
}

// 随机生成[-5, 5]之间的数据填充vector
int RandomVectorGenerator(std::vector<op::fp16_t> &vec, long long size)
{
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
    for (auto& elem : vec) {
        elem = static_cast<op::fp16_t>(distribution(generator));
    }
    return 0;
}

// 拼接两个大小相同的vector
int GatherVectors(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2, std::vector<op::fp16_t> &vec3)
{
    vec3.clear();
    vec3.reserve(vec1.size() + vec2.size());
    vec3.insert(vec3.end(), vec1.begin(), vec1.end());
    vec3.insert(vec3.end(), vec2.begin(), vec2.end());
    return 0;
}

// 对两个大小相同的vector执行加法
int AddVectors(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2, std::vector<op::fp16_t> &vec3)
{
    vec3.clear();
    vec3.resize(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        vec3[i] = vec1[i] + vec2[i];
    }
    return 0;
}

int AddVectors(std::vector<op::fp16_t> &vec1, std::vector<op::fp16_t> &vec2)
{
    for (size_t i = 0; i < vec1.size(); ++i) {
        vec2[i] = vec1[i] + vec2[i];
    }
    return 0;
}

int GenerateTestData(TestData &testData)
{
    std::vector<op::fp16_t> vec;
    for(int i=0; i<RANK_DIM;i++){
        int ret;
        //ret = RandomVectorGenerator(testData.input[i], testData.input[i].size());
        //CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] RandomVectorGenerate rank0 failed. ret = %d \n", ret);  return ret);
        std::fill(testData.input[i].begin(), testData.input[i].end(), i);

//        std::cout<<"input data" << i << std::endl;
//        print_vector(testData.input[i]);
        ret = GatherVectors(vec, testData.input[i], testData.out);
        vec = testData.out;
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] Generate output failed. ret = %d \n", ret);  return ret);
    }
    return 0;
}


int run_example_on_A2(int mpi_rank, HcclRootInfo rootInfo, TileXR::TileXRComm* comm, int rankId, TestData &testData)
{
    Args args;
    aclrtStream stream;
    aclrtContext context;
    int ret;
    // ret = aclrtSetDevice(rankId);

    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d", ret));
    ret = aclrtCreateContext(&context, rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d", ret));
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));

    HcclComm hcclCommP2P = nullptr;
    int rank_id = rankId;

    ret = HcclCommInitRootInfo(NPU_NUM_PER_MACHINE*MACHINE_NUM, &rootInfo, rank_id, &hcclCommP2P);
    if (ret != HCCL_SUCCESS) {
        std::cout << "[ERROR] rank "<< rank_id << " HCCL P2P CommInitClusterInfo failed. ret = " << ret << std::endl;
        return ret;
    }
    std::cout << "[INFO] mpi_rank "<< mpi_rank << "P2P HcclCommInitClusterInfo success, rank_id:" << rank_id << ", rankSize:" << RANK_DIM
              << ", hcclComm:" << hcclCommP2P << std::endl;


    args.localRankId = rankId;
    args.globalRankId = rank_id;
    args.comm = comm;
    args.hcclCommP2P = hcclCommP2P;
    args.stream = stream;
    args.context = context;
    LaunchOneThreadAllGather(args, testData);
    return 0;
}

template<typename T>
inline void LcclExpectEq(const T& a, const T& b, int line = -1)
{
    if (a != b) {
        std::cout << "LcclExpectEq error. line: " << line << "a: " << a << "b: " << b << std::endl;
    }
}

TileXR::TileXRComm* InitComm(int rank, int rankSize)
{
    TileXR::TileXRComm *comm = nullptr;
    TileXRUniqueId uniqueId;
    if (rank == 0) {
        int ret = TileXRGetUniqueId(&uniqueId, 0);
        LcclExpectEq(ret, TileXR::TILEXR_SUCCESS, __LINE__);
    }
    MPI_Bcast(&uniqueId, TILEXRUNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
    int ret = TileXRCommInitRank(uniqueId, rankSize, rank, reinterpret_cast<TileXRCommPtr*>(&comm));
    LcclExpectEq(ret, TileXR::TILEXR_SUCCESS, __LINE__);
    return comm;
}

int SetDev(int rank, int &devId, int devOffset = 0)
{
    aclError ret = aclInit(NULL);
    if (ret != ACL_SUCCESS) {
        std::cout << "aclInit failed! ret: " << ret << std::endl;
        return TileXR::TILEXR_ERROR_INTERNAL;
    }

    uint32_t localRankSize = 0;
    if (aclrtGetDeviceCount(&localRankSize) != 0) {
        std::cout << "aclrtGetDeviceCount err" << std::endl;
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    devId = (rank + devOffset) % localRankSize;

    std::cout << "curRank : " << rank << ", devId to set: " << devId << std::endl;
    if (aclrtSetDevice(devId) != ACL_SUCCESS) {
        std::cout << "aclrtSetDevice err rank: " << rank << ", localRankSize: " << localRankSize << ", devId: " << devId << std::endl;
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    return TileXR::TILEXR_SUCCESS;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const char* env_var_name = "RANK_TABLE_FILE and RANK_TABLE_FILE_P2P and FIRST_RANK_ID";
    // 生成测试数据
    TestData testData;
    int ret = GenerateTestData(testData);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GenerateTestData failed. ret = %d \n", ret);  return ret);

    LOG_PRINT("[INFO] %s are identified and example on <Atlas A2> will be executed!\n", env_var_name);
    
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d", ret));

    HcclRootInfo rootInfo;
    if (mpi_rank == 0) {
        ret = HcclGetRootInfo(&rootInfo);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetRootInfo failed. ret = %d", ret));
    }
    MPI_Bcast(&rootInfo, sizeof(rootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

    int devId = 0;
    LcclExpectEq(SetDev(mpi_rank, devId, 0), TileXR::TILEXR_SUCCESS);
    TileXR::TileXRComm *comm = nullptr;
    comm = InitComm(mpi_rank, mpi_size);

    ret = comm->Init();
    LcclExpectEq(ret, TileXR::TILEXR_SUCCESS, __LINE__);
    if (ret != TileXR::TILEXR_SUCCESS) {
        std::cout << "[ERROR] rank "<< mpi_rank << " LCCL P2P CommInitClusterInfo failed. ret = " << ret << std::endl;
        aclFinalize();
        MPI_Finalize();
        return -1;
    }
    run_example_on_A2(mpi_rank, rootInfo, comm, mpi_rank, testData);


    ret = TileXRCommDestroy(reinterpret_cast<TileXRCommPtr>(comm));
    CHECK_RET(ret == TileXR::TILEXR_SUCCESS, LOG_PRINT("[ERROR] TileXRCommDestroy failed. ret = %d", ret));
    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    MPI_Finalize();

    return 0;
}