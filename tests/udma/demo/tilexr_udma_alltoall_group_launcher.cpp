/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <cstddef>
#include <cstdint>
#include <mutex>

#include "runtime/kernel.h"

extern "C" {
extern const unsigned char TileXRGroupedAllToAllKernelBinaryData[];
extern const std::size_t TileXRGroupedAllToAllKernelBinarySize;
}

namespace {

constexpr uint32_t TILEXR_RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246U;
constexpr char TILEXR_GROUPED_ALLTOALL_KERNEL_NAME[] =
    "tilexr_udma_all_to_all_group_kernel";
constexpr char TILEXR_GROUPED_ALLTOALL_BATCH_KERNEL_NAME[] =
    "tilexr_udma_all_to_all_group_batch_kernel";
constexpr char TILEXR_GROUPED_ALLTOALL_CREDIT_KERNEL_NAME[] =
    "tilexr_udma_all_to_all_group_credit_kernel";
constexpr char TILEXR_GROUPED_ALLTOALL_BATCH_CREDIT_KERNEL_NAME[] =
    "tilexr_udma_all_to_all_group_batch_credit_kernel";

struct alignas(8) GroupedAllToAllKernelArgs {
    uint8_t* commArgs;
    uint8_t* input;
    uint8_t* output;
    uint8_t* registeredMemory;
    uint8_t* debug;
    uint32_t invocationId;
    int32_t elementsPerPeer;
    int32_t chunkElements;
    uint32_t passCount;
    uint32_t groupCount;
    uint64_t payloadOffset0;
    uint64_t payloadOffset1;
    uint64_t signalOffset0;
    uint64_t signalOffset1;
    uint8_t* groupTrace;
    uint32_t traceIteration;
    uint32_t routeStage;
    uint32_t multiChannel;
    uint32_t primaryRouteParts;
    uint32_t groupWidth;
    uint32_t quietBatch;
};

struct alignas(8) GroupedAllToAllCreditKernelArgs {
    uint8_t* commArgs;
    uint8_t* input;
    uint8_t* output;
    uint8_t* registeredMemory;
    uint8_t* debug;
    uint32_t invocationId;
    int32_t elementsPerPeer;
    int32_t chunkElements;
    uint32_t passCount;
    uint32_t groupCount;
    uint64_t payloadOffset0;
    uint64_t payloadOffset1;
    uint64_t signalOffset0;
    uint64_t signalOffset1;
    uint64_t creditOffset0;
    uint64_t creditOffset1;
    uint8_t* groupTrace;
    uint32_t traceIteration;
    uint32_t routeStage;
    uint32_t multiChannel;
    uint32_t primaryRouteParts;
    uint32_t groupWidth;
    uint32_t quietBatch;
    uint32_t ingressWindow;
};

static_assert(sizeof(GroupedAllToAllKernelArgs) == 128U,
    "grouped alltoall kernel argument ABI changed");
static_assert(offsetof(GroupedAllToAllKernelArgs, payloadOffset0) == 64U,
    "grouped alltoall payload offset ABI changed");
static_assert(offsetof(GroupedAllToAllKernelArgs, groupTrace) == 96U,
    "grouped alltoall trace argument ABI changed");
static_assert(sizeof(GroupedAllToAllCreditKernelArgs) == 152U,
    "grouped alltoall credit kernel argument ABI changed");
static_assert(offsetof(GroupedAllToAllCreditKernelArgs, creditOffset0) == 96U,
    "grouped alltoall credit offset ABI changed");
static_assert(offsetof(GroupedAllToAllCreditKernelArgs, groupTrace) == 112U,
    "grouped alltoall credit trace argument ABI changed");

std::mutex g_groupedKernelRegistrationMutex;
bool g_groupedKernelRegistered = false;
rtError_t g_groupedKernelRegistrationStatus = RT_ERROR_NONE;
void* g_groupedKernelBinaryHandle = nullptr;
uint8_t g_groupedKernelFunctionToken = 0U;
uint8_t g_groupedBatchKernelFunctionToken = 0U;
uint8_t g_groupedCreditKernelFunctionToken = 0U;
uint8_t g_groupedBatchCreditKernelFunctionToken = 0U;

int8_t* GroupedKernelFunctionSignature()
{
    return reinterpret_cast<int8_t*>(&g_groupedKernelFunctionToken);
}

int8_t* GroupedBatchKernelFunctionSignature()
{
    return reinterpret_cast<int8_t*>(&g_groupedBatchKernelFunctionToken);
}

int8_t* GroupedCreditKernelFunctionSignature()
{
    return reinterpret_cast<int8_t*>(&g_groupedCreditKernelFunctionToken);
}

int8_t* GroupedBatchCreditKernelFunctionSignature()
{
    return reinterpret_cast<int8_t*>(&g_groupedBatchCreditKernelFunctionToken);
}

rtError_t RegisterGroupedKernel(int8_t* signature, const char* name)
{
    return rtFunctionRegister(
        g_groupedKernelBinaryHandle, signature, name, name, 0U);
}

rtError_t EnsureGroupedKernelRegistered()
{
    std::lock_guard<std::mutex> guard(g_groupedKernelRegistrationMutex);
    if (g_groupedKernelRegistered) {
        return g_groupedKernelRegistrationStatus;
    }

    rtDevBinary_t binary {};
    binary.data = TileXRGroupedAllToAllKernelBinaryData;
    binary.length = static_cast<uint32_t>(TileXRGroupedAllToAllKernelBinarySize);
    binary.magic = TILEXR_RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0U;

    g_groupedKernelRegistrationStatus =
        rtDevBinaryRegister(&binary, &g_groupedKernelBinaryHandle);
    if (g_groupedKernelRegistrationStatus != RT_ERROR_NONE) {
        return g_groupedKernelRegistrationStatus;
    }
    g_groupedKernelRegistrationStatus = RegisterGroupedKernel(
        GroupedKernelFunctionSignature(), TILEXR_GROUPED_ALLTOALL_KERNEL_NAME);
    if (g_groupedKernelRegistrationStatus != RT_ERROR_NONE) {
        return g_groupedKernelRegistrationStatus;
    }
    g_groupedKernelRegistrationStatus = RegisterGroupedKernel(
        GroupedBatchKernelFunctionSignature(),
        TILEXR_GROUPED_ALLTOALL_BATCH_KERNEL_NAME);
    if (g_groupedKernelRegistrationStatus != RT_ERROR_NONE) {
        return g_groupedKernelRegistrationStatus;
    }
    g_groupedKernelRegistrationStatus = RegisterGroupedKernel(
        GroupedCreditKernelFunctionSignature(),
        TILEXR_GROUPED_ALLTOALL_CREDIT_KERNEL_NAME);
    if (g_groupedKernelRegistrationStatus != RT_ERROR_NONE) {
        return g_groupedKernelRegistrationStatus;
    }
    g_groupedKernelRegistrationStatus = RegisterGroupedKernel(
        GroupedBatchCreditKernelFunctionSignature(),
        TILEXR_GROUPED_ALLTOALL_BATCH_CREDIT_KERNEL_NAME);
    if (g_groupedKernelRegistrationStatus == RT_ERROR_NONE) {
        g_groupedKernelRegistered = true;
    }
    return g_groupedKernelRegistrationStatus;
}

} // namespace

int launch_tilexr_udma_all_to_all_group(
    uint32_t blockDim, void* stream, uint8_t* commArgs, uint8_t* input, uint8_t* output,
    uint8_t* registeredMemory, uint8_t* debug, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    uint64_t creditOffset0, uint64_t creditOffset1,
    uint8_t* groupTrace, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel, uint32_t primaryRouteParts,
    uint32_t groupWidth, uint32_t quietBatch, uint32_t ingressWindow)
{
    const rtError_t registrationStatus = EnsureGroupedKernelRegistered();
    if (registrationStatus != RT_ERROR_NONE) {
        return static_cast<int>(registrationStatus);
    }

    const bool useCredit = ingressWindow != 0U;
    const bool useBatch = quietBatch != 1U;
    GroupedAllToAllKernelArgs args {};
    GroupedAllToAllCreditKernelArgs creditArgs {};
    int8_t* functionSignature = nullptr;
    if (useCredit) {
        creditArgs = {
            commArgs, input, output, registeredMemory, debug, invocationId,
            elementsPerPeer, chunkElements, passCount, groupCount,
            payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
            creditOffset0, creditOffset1, groupTrace, traceIteration,
            routeStage, multiChannel, primaryRouteParts, groupWidth,
            quietBatch, ingressWindow,
        };
        functionSignature = useBatch ? GroupedBatchCreditKernelFunctionSignature() :
            GroupedCreditKernelFunctionSignature();
    } else {
        args = {
            commArgs, input, output, registeredMemory, debug, invocationId,
            elementsPerPeer, chunkElements, passCount, groupCount,
            payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
            groupTrace, traceIteration, routeStage, multiChannel,
            primaryRouteParts, groupWidth, quietBatch,
        };
        functionSignature = useBatch ? GroupedBatchKernelFunctionSignature() :
            GroupedKernelFunctionSignature();
    }
    rtArgsEx_t argsInfo {};
    argsInfo.args = useCredit ? static_cast<void*>(&creditArgs) : static_cast<void*>(&args);
    argsInfo.argsSize = useCredit ? sizeof(creditArgs) : sizeof(args);

    rtTaskCfgInfo_t cfgInfo {};
    cfgInfo.schemMode = RT_SCHEM_MODE_NORMAL;
    return static_cast<int>(rtKernelLaunchWithFlagV2(
        functionSignature, blockDim, &argsInfo, nullptr,
        static_cast<rtStream_t>(stream), 0U, &cfgInfo));
}
