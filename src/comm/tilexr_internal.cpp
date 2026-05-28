/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tilexr_internal.h"
#include <map>
#include <mutex>
#include <vector>
#include <mki/utils/log/log.h>
#include <mki/utils/env/env.h>
#include <runtime/kernel.h>
#include "ccl_kernel_args.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"

constexpr int AI_CORE_NUM_24 = 24;
constexpr int AI_CORE_NUM_20 = 20;
constexpr int AI_CORE_NUM_2 = 2;

using namespace std;
using namespace Mki;

extern const int TILEXR_CCE_BIN_STR[] = {0};
// asm(R"(.section .rodata, "a", @progbits
// TILEXR_CCE_BIN_STR:.incbin "/tmp/tilexr_cce.o"
// .byte 0
// .previous)");

constexpr int LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246;
constexpr int COC_RT_DEV_BINARY_MAGIC_ELF = 0x43554245;
constexpr int TILEXR_1OP_BIN_SIZE = 3000000;

namespace TileXR {
const std::map<HcclDataType, std::string> DATATYPE2NAME = {
    { HCCL_DATA_TYPE_INT32, "int" },
    { HCCL_DATA_TYPE_INT16, "int16_t" },
    { HCCL_DATA_TYPE_INT8, "int8_t" },
    { HCCL_DATA_TYPE_INT64, "int64_t" },
    { HCCL_DATA_TYPE_FP32, "float" },
    { HCCL_DATA_TYPE_FP16, "float16_t" },
    { HCCL_DATA_TYPE_BFP16, "bfloat16_t" }
};

const std::unordered_map<std::string, ChipName> CHIP_MAP = {
    {"Ascend310P", ChipName::CHIP_310P3},
    {"Ascend910B1", ChipName::CHIP_910B1},
    {"Ascend910B2", ChipName::CHIP_910B2},
    {"Ascend910B2C", ChipName::CHIP_910B2C},
    {"Ascend910B3", ChipName::CHIP_910B3},
    {"Ascend910B4", ChipName::CHIP_910B4},
    {"Ascend910B4-1", ChipName::CHIP_910B41},
    {"Ascend910_9391", ChipName::CHIP_910_9391},
    {"Ascend910_9381", ChipName::CHIP_910_9381},
    {"Ascend910_9392", ChipName::CHIP_910_9392},
    {"Ascend910_9382", ChipName::CHIP_910_9382},
    {"Ascend910_9372", ChipName::CHIP_910_9372},
    {"Ascend910_9361", ChipName::CHIP_910_9361},
    {"Ascend950", ChipName::CHIP_950},
    {"Ascend950DT", ChipName::CHIP_950},
    {"Ascend950DT_9581", ChipName::CHIP_950},
    {"Ascend950PR", ChipName::CHIP_950}
};

template<class T>
int RegisterBinaryKernel(const string &funcName, int8_t *funSig, const T *binStrPtr, int magic, int len = 0)
{
    rtDevBinary_t binary;
    void *binHandle = nullptr;
    binary.data = binStrPtr;
    binary.length = (len == 0 ? TILEXR_1OP_BIN_SIZE : len);

    binary.magic = magic;
    binary.version = 0;
    rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
    if (rtRet != RT_ERROR_NONE) {
        MKI_LOG(WARN) << "rtDevBinaryRegister failed! " << to_string(rtRet) << ", funcName = " << funcName;
        return TILEXR_ERROR_INTERNAL;
    }
    rtRet = rtFunctionRegister(binHandle, funSig, funcName.c_str(), funcName.c_str(), 0);
    if (rtRet != RT_ERROR_NONE) {
        MKI_LOG(WARN) << "rtFunctionRegister failed! " << to_string(rtRet) << ", funcName = " << funcName;
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int8_t *GetFunSig(TileXRType type, HcclDataType dataType, uint64_t devType = 0)
{
    constexpr int sigOffset = 16;
    return reinterpret_cast<int8_t *>((static_cast<uint64_t>(type) << sigOffset << sigOffset) +
        (static_cast<uint64_t>(dataType)<< sigOffset) + devType);
}

const int* FindNextOpStart(const int opStartMaigc, const int* cclBinEndPtr, const int* cclBinPtr)
{
    if (cclBinPtr == nullptr) {
        MKI_LOG(ERROR) << "FindNextOpStart failed! cclBinPtr is nullptr";
        return nullptr;
    }
    while (*cclBinPtr != opStartMaigc and cclBinPtr < cclBinEndPtr) {
        cclBinPtr++;
    }
    if (*cclBinPtr == opStartMaigc) {
        cclBinPtr++;
    }
    return cclBinPtr;
}

int RegistCCLOp3Kernel(const int* cclBinPtr)
{
    const int* cclBinStr = TILEXR_CCE_BIN_STR;
    auto cclBinEndPtr = cclBinStr + TILEXR_1OP_BIN_SIZE / sizeof(int);
    const int opStartMaigc = 0x44444444;
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT8,
                                           HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16,
                                           HCCL_DATA_TYPE_INT64 };

    // 这里得按.cpp的文件名字排序，比如 allgather.cpp.o, allreduce.cpp.o, reduce_scatter.cpp.o
    std::vector<TileXRType> registerCCLTypesOp3 = {
        TileXRType::ALL2ALL_V_C, TileXRType::ALL_GATHER, TileXRType::ALL_REDUCE, TileXRType::BROADCAST, TileXRType::REDUCE_SCATTER
    };
    for (auto ccl : registerCCLTypesOp3) {
        cclBinPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, cclBinPtr);
        if (cclBinPtr == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }
        if (cclBinPtr == cclBinEndPtr) {
            return TILEXR_SUCCESS;
        }
        ++cclBinPtr;
        for (auto t : registerTypes) {
            RegisterBinaryKernel(TILEXR_TYPE2NAME.at(ccl) + "_" + DATATYPE2NAME.at(t),
                                 GetFunSig(ccl, t, 1), cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC);
        }
    }
    return TILEXR_SUCCESS;
}

int RegistCCLOp2Kernel(const int* cclBinPtr, const int* nextPtr)
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT8,
                                           HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16,
                                           HCCL_DATA_TYPE_INT64 };
    std::vector<TileXRType> registerCCLTypesOp2 = { // 完成算子实现后在这里添加算子注册
        TileXRType::ALL_GATHER, TileXRType::REDUCE_SCATTER, TileXRType::ALL2ALL_V_C,
        TileXRType::SEND, TileXRType::RECV, TileXRType::LOCAL_REDUCE, TileXRType::GATHER, TileXRType::ALL2ALL,
    };
    int res = TILEXR_SUCCESS;
    for (auto ccl : registerCCLTypesOp2) {
        for (auto t : registerTypes) {
            res = RegisterBinaryKernel(TILEXR_TYPE2NAME.at(ccl) + "_" + DATATYPE2NAME.at(t), GetFunSig(ccl, t),
                cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC, (nextPtr - cclBinPtr) * sizeof(int));
        }
    }
    if (res != TILEXR_SUCCESS) {
        return res;
    }
    res = RegisterBinaryKernel(TILEXR_TYPE2NAME.at(TileXRType::BROADCAST),
        GetFunSig(TileXRType::BROADCAST, HCCL_DATA_TYPE_RESERVED), cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC);
    if (res != TILEXR_SUCCESS) {
        return res;
    }
    res = RegisterBinaryKernel(TILEXR_TYPE2NAME.at(TileXRType::BANDWIDTH),
        GetFunSig(TileXRType::BANDWIDTH, HCCL_DATA_TYPE_RESERVED), cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC);
    return res;
}

int RegistCCLKernel(const bool enableProfiling)
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT8,
                                           HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16,
                                           HCCL_DATA_TYPE_INT64 };
    const int* cclBinStr = TILEXR_CCE_BIN_STR;
    auto cclBinEndPtr = cclBinStr + TILEXR_1OP_BIN_SIZE / sizeof(int);
    const int* cclBinPtr = cclBinStr + 1;
    const int opStartMaigc = 0x44444444;
    const int* nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, cclBinPtr);
    if (nextPtr == nullptr) {
        return TILEXR_ERROR_INTERNAL;
    }
    // 相同的算子被注册了两组(DDDD分组)，第一组是加了dump的算子，第二组是正常算子
    if (!enableProfiling) {
        cclBinPtr = nextPtr;
        nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, nextPtr);
        if (cclBinPtr == nullptr || cclBinPtr == cclBinEndPtr || nextPtr == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }
    }
    std::vector<TileXRType> registerCCLTypesOp1 = { // 完成算子实现后在这里添加算子注册
        TileXRType::ALL_REDUCE
    };
    for (auto ccl : registerCCLTypesOp1) {
        for (auto t : registerTypes) {
            RegisterBinaryKernel(TILEXR_TYPE2NAME.at(ccl) + "_" + DATATYPE2NAME.at(t), GetFunSig(ccl, t),
                                 cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC, (nextPtr - cclBinPtr) * sizeof(int));
        }
    }

    if (!enableProfiling) {
        cclBinPtr = nextPtr;
        nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, nextPtr);
        if (cclBinPtr == nullptr || cclBinPtr == cclBinEndPtr || nextPtr == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }
    }
    int ret = 0;
    ret = RegistCCLOp2Kernel(cclBinPtr, nextPtr);
    if (ret != TILEXR_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }
    ret = RegistCCLOp3Kernel(cclBinPtr);
    if (ret != TILEXR_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

void RegistCoCKernel()
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16 };
    vector<vector<TileXRType>> registerCOCTypes = {
        { TileXRType::PURE_MATMUL },
        { TileXRType::MATMUL_ALL_REDUCE },
        { TileXRType::MATMUL_REDUCE_SCATTER },
        { TileXRType::ALL_GATHER_MATMUL, TileXRType::ALL_GATHER_MATMUL_V2 },
        { TileXRType::ALL_GATHER_MATMUL_REDUCE_SCATTER},
        { TileXRType::ALLTOALLV_ALLGATHER_MATMUL, TileXRType::ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN},
        { TileXRType::MATMUL_REDUCESCATTER_ALLTOALLV, TileXRType::MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN},
    };

    auto cocCceBinStr = TILEXR_CCE_BIN_STR + TILEXR_1OP_BIN_SIZE / sizeof(int);
    for (auto tilexrTypeGroup : registerCOCTypes) {
        for (auto tilexrType : tilexrTypeGroup) {
            for (auto t : registerTypes) {
                RegisterBinaryKernel(TILEXR_TYPE2NAME.at(tilexrType) + "_" + DATATYPE2NAME.at(t), GetFunSig(tilexrType, t),
                    cocCceBinStr, COC_RT_DEV_BINARY_MAGIC_ELF);
            }
        }
        cocCceBinStr += TILEXR_1OP_BIN_SIZE / sizeof(int);
    }
}

int RegistKernel(const bool enableProfiling)
{
    static bool init = false;
    static mutex mut;
    lock_guard<mutex> guard(mut);
    if (init) {
        return 0;
    }
    RegistCoCKernel();
    RegistCCLKernel(enableProfiling);
    init = true;
    return TILEXR_SUCCESS;
}

int64_t Count2Size(int64_t count, const HcclDataType &dataType)
{
    int64_t dataSize = TILEXR_INVALID_VALUE;
    if (dataType == HCCL_DATA_TYPE_INT8 || dataType == HCCL_DATA_TYPE_UINT8) {
        dataSize = count;
    } else if (dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_FP16 ||
               dataType == HCCL_DATA_TYPE_BFP16 || dataType == HCCL_DATA_TYPE_UINT16) {
        dataSize = count * sizeof(int16_t);
    } else if (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_INT32 ||
               dataType == HCCL_DATA_TYPE_UINT32) {
        dataSize = count * sizeof(int32_t);
    } else if (dataType == HCCL_DATA_TYPE_INT64 || dataType == HCCL_DATA_TYPE_UINT64) {
        dataSize = count * sizeof(int64_t);
    } else {
        MKI_LOG(ERROR) << "unknown datatype";
    }
    return dataSize;
}

int LoadMTE(TileXRType cclType, AscendCCLKernelArgs &args, uint32_t blockDim, HcclDataType dataType, aclrtStream stream)
{
    int error = 0;
    MKI_LOG(DEBUG) << "LoadMTE " << TILEXR_TYPE2NAME.at(cclType) << " count:" << args.count << " dataType:" << dataType
                   << " op:" << args.op << " blockDim:" << blockDim << " rootRank:" << args.root
                   << ", magic: " << args.magic;
    int64_t dataSize = Count2Size(args.count, dataType);
    if (dataSize == TILEXR_INVALID_VALUE || blockDim == 0) {
        MKI_LOG(ERROR) << ("LoadMTE args are invalid");
        return TILEXR_ERROR_MKIRT;
    }

    static const char *ENV = Mki::GetEnv("LCCL_PARALLEL");
    if (ENV && (string(ENV) == "1" || string(ENV) == "true") && dataSize >= IPC_BUFF_MAX_SIZE) {
        MKI_LOG(ERROR) << ("LoadMTE args are invalid, because LCCL_PARALLEL is open, and dataSize is too big.");
        return TILEXR_ERROR_MKIRT;
    }

    rtTaskCfgInfo_t cfgInfo{};
    cfgInfo.schemMode = 1;

    rtArgsEx_t argsInfo{};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);
    // 如果想要发射910A5算子，那么需要把GetFunSig第三个参数添加进去（加1）
    if (cclType == TileXRType::BANDWIDTH) {
        args.count = dataSize;
        error = rtKernelLaunchWithFlagV2(GetFunSig(cclType, HCCL_DATA_TYPE_RESERVED),
                                         blockDim, &argsInfo, nullptr, stream, 0, &cfgInfo);
    } else {
        error = rtKernelLaunchWithFlagV2(GetFunSig(cclType, dataType),
                                         blockDim, &argsInfo, nullptr, stream, 0, &cfgInfo);
    }
    if (error != RT_ERROR_NONE) {
        MKI_LOG(ERROR) << "AsdRtFunctionLaunch -:" << TILEXR_TYPE2NAME.at(cclType) << to_string(error);
        return TILEXR_ERROR_MKIRT;
    }
    return error;
}

template<typename T, typename M>
size_t OffsetOf(M T::*member, T obj)
{
    return reinterpret_cast<size_t>(&(obj.*member)) - reinterpret_cast<size_t>(&obj);
}

/**
 * @brief 用于获取芯片名称
 */
ChipName GetChipName()
{
    // 在分配内存时用到
    static ChipName curChipName = ChipName::RESERVED;
    if (curChipName != ChipName::RESERVED) {
        return curChipName;
    }
    constexpr int socVerLength = 100; // asd没有相应的宏和常量，这里和asd测试代码中的长度保持一致
    char ver[socVerLength];
    auto ret = rtGetSocVersion(ver, socVerLength);
    if (ret != RT_ERROR_NONE) {
        MKI_LOG(ERROR) << "rtGetSocVersion failed, not sure whether the function is normal, please use it with caution";
        return ChipName::RESERVED;
    }
    string chipName(ver);
    MKI_LOG(DEBUG) << "rtGetSocVersion -- The result after converting ver to string is:" << chipName;

    auto it = CHIP_MAP.find(chipName);
    if (it != CHIP_MAP.end()) {
        curChipName = it->second;
    } else {
        MKI_LOG(WARN) << "There is no commitment to the supported chip types yet," <<
                      " and it is not certain whether the functions will work properly.";
    }
    return curChipName;
}

uint32_t GetCoreNum(ChipName chipName)
{
    switch (chipName) {
        case ChipName::CHIP_910B1:
        case ChipName::CHIP_910B2:
        case ChipName::CHIP_910_9391:
        case ChipName::CHIP_910_9381:
        case ChipName::CHIP_910_9392:
        case ChipName::CHIP_910_9382:
        case ChipName::CHIP_910B2C:
        case ChipName::CHIP_950:
            return AI_CORE_NUM_24;
        case ChipName::CHIP_910B3:
        case ChipName::CHIP_910B4:
        case ChipName::CHIP_910B41:
        case ChipName::CHIP_910_9372:
        case ChipName::CHIP_910_9361:
        case ChipName::CHIP_910A5:
            return AI_CORE_NUM_20;
        case ChipName::CHIP_310P3:
            return AI_CORE_NUM_2;
        default:
            MKI_LOG(ERROR) << "Unknown chip name";
            return 0;
    }
}
}
