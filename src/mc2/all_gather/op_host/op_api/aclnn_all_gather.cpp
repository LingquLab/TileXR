/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_all_gather.h"
#include "securec.h"
#include "acl/acl.h"
#include <vector>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "hccl_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" { 
#endif
static constexpr size_t DIMS_NUM = 2;
enum NnopbaseHcclServerType : uint32_t {
  NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
  NNOPBASE_HCCL_SERVER_TYPE_MTE,
  NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerAllGatherGetWorkspaceSize(const aclTensor *a, char* group, int64_t group_comm,
                                                          int64_t rankSize, 
                                                          const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                                          aclOpExecutor **executor);
extern aclnnStatus aclnnInnerAllGather(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

static bool CheckNotNull(const aclTensor *a, const aclTensor *gatherout)
{
    OP_CHECK_NULL(a, return false);
    OP_CHECK_NULL(gatherout, return false);
    return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16
};

static bool CheckDtypeValid(const aclTensor* a, const aclTensor* gatherout)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(a, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gatherout, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor *a, const aclTensor *gatherOut, int64_t rankSize)
{
    OP_CHECK_WRONG_DIMENSION(a, DIMS_NUM, return false);

    auto aLen = a->GetViewShape().GetDim(0);
    auto gatherOutLen = gatherOut->GetViewShape().GetDim(0);

    OP_API_CHECK((aLen * rankSize != gatherOutLen), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
        "The operand's shape should satisfy: aDim0 * rankSize = bDim0 = cDim0 = gatherOutDim0.");
        return false;
    });

    return true;
}

static aclnnStatus CheckParams(const aclTensor *a, const aclTensor *gatherout, int64_t rankSize)
{
    CHECK_RET(CheckNotNull(a, gatherout), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(a, gatherout), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(a, gatherout, rankSize), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize_ = 1;
    for (auto i : shape) {
        shapeSize_ *= i;
    }
    return shapeSize_;
}

static size_t getDataTypeSize(aclDataType type) {
    switch (type) {
        case aclDataType::ACL_FLOAT:    return sizeof(float);
        case aclDataType::ACL_FLOAT16:  return 2;  // float16 是 2 字节
        case aclDataType::ACL_INT8:     return sizeof(int8_t);
        case aclDataType::ACL_INT32:    return sizeof(int32_t);
        case aclDataType::ACL_UINT8:    return sizeof(uint8_t);
        default:                        return 0;
    }
}

static int CreateAclTensor(const void *hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * getDataTypeSize(dataType);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    OP_API_CHECK(ret != ACL_SUCCESS, {OP_LOGW("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret;});
    ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    OP_API_CHECK(ret != ACL_SUCCESS, {OP_LOGW("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret;});
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

aclnnStatus aclnnAllGatherGetWorkspaceSize(const aclTensor *a, char* group, int64_t group_comm,
                                              int64_t rankSize, 
                                              const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                              aclOpExecutor **executor) 
{
    auto retParam = CheckParams(a, gatherOutOut, rankSize);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    OP_LOGD("A is %s.", a->ToString().GetString());
    OP_LOGD("GatherOut is %s.", gatherOutOut->ToString().GetString());

    aclnnStatus ret = aclnnInnerAllGatherGetWorkspaceSize(a, group, group_comm, rankSize,
                                                             gatherOutOut, workspaceSize, executor);
    OP_LOGD("AllGather, aclnnInnerGetWorkspaceSize ret = %d.", ret);

    return ret;
}

aclnnStatus aclnnAllGather(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                              aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    }
    auto ret = aclnnInnerAllGather(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore,ret = %d.", ret);
        return ret;
    }
    return ret;
}



int AllGather(void *sendBuf, void *recvBuf, int64_t sendCount, aclDataType dataType, TileXR::TileXRComm* comm, HcclComm hcclComm, 
                aclrtStream stream)
{
    char hcomNameP2P[128] = {0};
    int ret = HcclGetCommName(hcclComm, hcomNameP2P);
    OP_API_CHECK(ret != ACL_SUCCESS, {OP_LOGE(ACLNN_ERR_INNER, "[ERROR] P2P HcclGetCommName failed. ret: %d\n", ret); return ret;});

    void *deviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    aclTensor *out = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    int64_t rankSize = comm->GetRankSize();
    const std::vector<int64_t> inShape = {sendCount, 1};
    const std::vector<int64_t> outShape = {sendCount*rankSize, 1};
    int64_t outShapeSize = sendCount*rankSize;

    ret = CreateAclTensor(sendBuf, inShape, &deviceAddr, dataType, &input);
    OP_API_CHECK(ret != ACL_SUCCESS, {return ret;});
    ret = CreateAclTensor(recvBuf, outShape, &outDeviceAddr, dataType, &out);
    OP_API_CHECK(ret != ACL_SUCCESS, {return ret;});

    int64_t commPtr = (int64_t) (comm->GetCommArgsPtr());


    // 调用第一阶段接口
    ret = aclnnAllGatherGetWorkspaceSize(
        input, hcomNameP2P, commPtr, rankSize, out, &workspaceSize, &executor);
    OP_API_CHECK(ret != ACL_SUCCESS,
        {OP_LOGE(ACLNN_ERR_INNER, "aclnnAllGatherGetWorkspaceSize failed. ret = %d \n", ret); return ret;});
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        OP_API_CHECK(ret != ACL_SUCCESS, {OP_LOGE(ACLNN_ERR_INNER, "aclrtMalloc workspace failed. ret = %d \n", ret);  return ret;});
    }
    // 调用第二阶段接口
    ret = aclnnAllGather(workspaceAddr, workspaceSize, executor, stream);
    OP_API_CHECK(ret != ACL_SUCCESS, {OP_LOGE(ACLNN_ERR_INNER, "aclnnAllGather failed. ret = %d \n", ret); return ret;});
    // 同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(stream, 10000);
    OP_API_CHECK(ret != ACL_SUCCESS, {OP_LOGE(ACLNN_ERR_INNER, "aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
          return ret;});
    // 算子计算结果拷贝到host侧
   std::vector<op::fp16_t> outputData(outShapeSize, 0);
   ret = aclrtMemcpy(recvBuf, outShapeSize * getDataTypeSize(dataType), outDeviceAddr,
                     outShapeSize * getDataTypeSize(dataType), ACL_MEMCPY_DEVICE_TO_HOST);
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
   return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif