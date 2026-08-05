/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <cstddef>
#include <cstdint>
#include <limits>

#include "acl/acl_rt.h"
#include "tilexr_udma_alltoall_group_layout.h"

namespace {

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
    uint32_t simtMode;
    uint32_t groupWidth;
    uint32_t quietBatch;
    uint32_t prewarmSq;
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
    uint32_t simtMode;
    uint32_t groupWidth;
    uint32_t quietBatch;
    uint32_t ingressWindow;
    uint32_t prewarmSq;
};

static_assert(sizeof(GroupedAllToAllKernelArgs) == 136U,
    "grouped alltoall kernel argument ABI changed");
static_assert(offsetof(GroupedAllToAllKernelArgs, payloadOffset0) == 64U,
    "grouped alltoall payload offset ABI changed");
static_assert(offsetof(GroupedAllToAllKernelArgs, groupTrace) == 96U,
    "grouped alltoall trace argument ABI changed");
static_assert(offsetof(GroupedAllToAllKernelArgs, simtMode) == 120U,
    "grouped alltoall SIMT argument ABI changed");
static_assert(sizeof(GroupedAllToAllCreditKernelArgs) == 160U,
    "grouped alltoall credit kernel argument ABI changed");
static_assert(offsetof(GroupedAllToAllCreditKernelArgs, creditOffset0) == 96U,
    "grouped alltoall credit offset ABI changed");
static_assert(offsetof(GroupedAllToAllCreditKernelArgs, groupTrace) == 112U,
    "grouped alltoall credit trace argument ABI changed");
static_assert(offsetof(GroupedAllToAllCreditKernelArgs, simtMode) == 136U,
    "grouped alltoall credit SIMT argument ABI changed");

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
    uint32_t simtMode, uint32_t groupWidth, uint32_t quietBatch,
    uint32_t ingressWindow, uint32_t prewarmSq)
{
    const bool useCredit = ingressWindow != 0U;
    const bool useBatch = quietBatch != 1U;
    GroupedAllToAllKernelArgs args {};
    GroupedAllToAllCreditKernelArgs creditArgs {};
    if (useCredit) {
        creditArgs = {
            commArgs, input, output, registeredMemory, debug, invocationId,
            elementsPerPeer, chunkElements, passCount, groupCount,
            payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
            creditOffset0, creditOffset1, groupTrace, traceIteration,
            routeStage, multiChannel, primaryRouteParts, simtMode, groupWidth,
            quietBatch, ingressWindow, prewarmSq,
        };
    } else {
        args = {
            commArgs, input, output, registeredMemory, debug, invocationId,
            elementsPerPeer, chunkElements, passCount, groupCount,
            payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
            groupTrace, traceIteration, routeStage, multiChannel,
            primaryRouteParts, simtMode, groupWidth, quietBatch, prewarmSq,
        };
    }

    uint32_t dynamicUbSize = 0U;
    if (simtMode != 0U) {
        int32_t deviceId = 0;
        int64_t totalUbSize = 0;
        aclError aclRet = aclrtGetDevice(&deviceId);
        if (aclRet != ACL_SUCCESS) {
            return static_cast<int>(aclRet);
        }
        aclRet = aclrtGetDeviceInfo(
            static_cast<uint32_t>(deviceId),
            ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &totalUbSize);
        if (aclRet != ACL_SUCCESS) {
            return static_cast<int>(aclRet);
        }
        if (totalUbSize <= static_cast<int64_t>(
                TileXR::Demo::kAllToAllGroupDcacheBytes) ||
            totalUbSize > static_cast<int64_t>(
                std::numeric_limits<uint32_t>::max())) {
            return static_cast<int>(ACL_ERROR_INVALID_PARAM);
        }
        dynamicUbSize = static_cast<uint32_t>(totalUbSize) -
            TileXR::Demo::kAllToAllGroupDcacheBytes;
    }

    // bisheng lowers every launch below to rtKernelLaunchWithFlagV2. Its host
    // wrapper also completes the fat binary's delayed runtime registration.
    if (useCredit) {
        if (useBatch) {
            tilexr_udma_all_to_all_group_batch_credit_kernel<<<
                blockDim, dynamicUbSize, stream>>>(
                creditArgs.commArgs, creditArgs.input, creditArgs.output,
                creditArgs.registeredMemory, creditArgs.debug,
                creditArgs.invocationId, creditArgs.elementsPerPeer,
                creditArgs.chunkElements, creditArgs.passCount,
                creditArgs.groupCount, creditArgs.payloadOffset0,
                creditArgs.payloadOffset1, creditArgs.signalOffset0,
                creditArgs.signalOffset1, creditArgs.creditOffset0,
                creditArgs.creditOffset1, creditArgs.groupTrace,
                creditArgs.traceIteration, creditArgs.routeStage,
                creditArgs.multiChannel, creditArgs.primaryRouteParts,
                creditArgs.simtMode, creditArgs.groupWidth,
                creditArgs.quietBatch, creditArgs.ingressWindow,
                creditArgs.prewarmSq);
        } else {
            tilexr_udma_all_to_all_group_credit_kernel<<<
                blockDim, dynamicUbSize, stream>>>(
                creditArgs.commArgs, creditArgs.input, creditArgs.output,
                creditArgs.registeredMemory, creditArgs.debug,
                creditArgs.invocationId, creditArgs.elementsPerPeer,
                creditArgs.chunkElements, creditArgs.passCount,
                creditArgs.groupCount, creditArgs.payloadOffset0,
                creditArgs.payloadOffset1, creditArgs.signalOffset0,
                creditArgs.signalOffset1, creditArgs.creditOffset0,
                creditArgs.creditOffset1, creditArgs.groupTrace,
                creditArgs.traceIteration, creditArgs.routeStage,
                creditArgs.multiChannel, creditArgs.primaryRouteParts,
                creditArgs.simtMode, creditArgs.groupWidth,
                creditArgs.quietBatch, creditArgs.ingressWindow,
                creditArgs.prewarmSq);
        }
    } else if (useBatch) {
        tilexr_udma_all_to_all_group_batch_kernel<<<
            blockDim, dynamicUbSize, stream>>>(
            args.commArgs, args.input, args.output, args.registeredMemory,
            args.debug, args.invocationId, args.elementsPerPeer,
            args.chunkElements, args.passCount, args.groupCount,
            args.payloadOffset0, args.payloadOffset1, args.signalOffset0,
            args.signalOffset1, args.groupTrace, args.traceIteration,
            args.routeStage, args.multiChannel, args.primaryRouteParts,
            args.simtMode, args.groupWidth, args.quietBatch, args.prewarmSq);
    } else {
        tilexr_udma_all_to_all_group_kernel<<<
            blockDim, dynamicUbSize, stream>>>(
            args.commArgs, args.input, args.output, args.registeredMemory,
            args.debug, args.invocationId, args.elementsPerPeer,
            args.chunkElements, args.passCount, args.groupCount,
            args.payloadOffset0, args.payloadOffset1, args.signalOffset0,
            args.signalOffset1, args.groupTrace, args.traceIteration,
            args.routeStage, args.multiChannel, args.primaryRouteParts,
            args.simtMode, args.groupWidth, args.quietBatch, args.prewarmSq);
    }
    return static_cast<int>(ACL_SUCCESS);
}
