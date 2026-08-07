#include "ep_plan_host.h"

#include <cstddef>
#include <cstdint>

#include "ep_plan_peer_mailbox.h"
#include "tilexr_types.h"

namespace TileXREp {
namespace Plan {
namespace {

bool IsAligned(const void *pointer, uint64_t alignment)
{
    return pointer != nullptr &&
        (reinterpret_cast<uintptr_t>(pointer) & static_cast<uintptr_t>(alignment - 1)) == 0;
}

bool ValidInt32Pointer(const int32_t *pointer)
{
    return IsAligned(pointer, alignof(int32_t));
}

bool ValidPlanPointers(const TileXRMoonEPPlanDesc &plan)
{
    return ValidInt32Pointer(plan.dst) && ValidInt32Pointer(plan.cuSeqlens) &&
        ValidInt32Pointer(plan.expertsToCopy) && ValidInt32Pointer(plan.remoteStats) && ValidInt32Pointer(plan.dupGroups) &&
        ValidInt32Pointer(plan.dupLoffs) && ValidInt32Pointer(plan.dupCounts) &&
        ValidInt32Pointer(plan.status);
}

bool PlanIdentityMatches(const PlanHostArguments &arguments, int rankSize)
{
    const TileXRMoonEPPlanDesc &plan = *arguments.plan;
    const TileXRMoonEPPlanConfig &config = *arguments.config;
    return plan.s == arguments.s && plan.k == arguments.topK && plan.r == rankSize &&
        plan.e == arguments.expertNum && plan.b == config.prefetchSlots &&
        plan.cap == config.rankTokenCapacity && plan.nvS == config.nvS &&
        plan.tokenPadding == config.tokenPadding && plan.epoch != 0;
}

int ValidateRuntimeMetadata(const PlanRuntimeMetadata &runtime, int64_t expertNum)
{
    const TileXR::CommArgs &commArgs = *runtime.hostCommArgs;
    if (runtime.deviceCommArgs == nullptr) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    for (int rank = 0; rank < commArgs.rankSize; ++rank) {
        if (commArgs.peerMems[rank] == nullptr) {
            return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
        }
    }
    const PlanPeerMailboxLayout mailbox =
        BuildPlanPeerMailboxLayout(commArgs.rankSize, expertNum);
    if (mailbox.inputBytes > mailbox.rowBytes ||
        mailbox.totalBytes > static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace

int BuildPlanCallHeader(int64_t rankSize, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig &config, uint64_t epoch, uint64_t topologyHash, PlanCallHeader *header)
{
    if (header == nullptr || epoch == 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    PlanWorkspaceLayout layout {};
    const int ret = BuildPlanWorkspaceLayout(rankSize, s, topK, expertNum, config, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    PlanCallHeader result {};
    result.abiVersion = kPlanAbiVersion;
    result.headerBytes = static_cast<int32_t>(sizeof(PlanCallHeader));
    result.rankSize = static_cast<int32_t>(rankSize);
    result.s = s;
    result.k = topK;
    result.expertNum = expertNum;
    result.prefetchSlots = config.prefetchSlots;
    result.rankTokenCapacity = config.rankTokenCapacity;
    result.nvS = config.nvS;
    result.tokenPadding = config.tokenPadding;
    result.tokenRouteLimitPerPair = config.tokenRouteLimitPerPair;
    result.cardsPerServer = config.cardsPerServer;
    result.cardsPerCabinet = config.cardsPerCabinet;
    result.crossCandidateCount = config.crossCandidateCount;
    result.epoch = epoch;
    result.topologyHash = topologyHash;
    *header = result;
    return TileXR::TILEXR_SUCCESS;
}

bool PlanCallHeadersMatch(const PlanCallHeader &lhs, const PlanCallHeader &rhs)
{
    return lhs.abiVersion == rhs.abiVersion && lhs.headerBytes == rhs.headerBytes &&
        lhs.rankSize == rhs.rankSize && lhs.s == rhs.s && lhs.k == rhs.k &&
        lhs.expertNum == rhs.expertNum && lhs.prefetchSlots == rhs.prefetchSlots &&
        lhs.rankTokenCapacity == rhs.rankTokenCapacity && lhs.nvS == rhs.nvS &&
        lhs.tokenPadding == rhs.tokenPadding &&
        lhs.tokenRouteLimitPerPair == rhs.tokenRouteLimitPerPair &&
        lhs.cardsPerServer == rhs.cardsPerServer && lhs.cardsPerCabinet == rhs.cardsPerCabinet &&
        lhs.crossCandidateCount == rhs.crossCandidateCount && lhs.epoch == rhs.epoch &&
        lhs.topologyHash == rhs.topologyHash;
}

bool IsCommittedPlanReusable(const PlanCallHeader &committedHeader,
    const PlanCallHeader &requestedHeader, const PlanEpochState &epochState)
{
    if (epochState.committedEpoch == 0 ||
        epochState.requestedEpoch != epochState.committedEpoch ||
        committedHeader.epoch != epochState.committedEpoch ||
        requestedHeader.epoch != epochState.committedEpoch ||
        (epochState.reserved & kPlanAffinityCacheValid) == 0 ||
        epochState.topologyHash != committedHeader.topologyHash) {
        return false;
    }
    return PlanCallHeadersMatch(committedHeader, requestedHeader);
}

int ValidatePlanHostArguments(
    const PlanHostArguments &arguments, const PlanRuntimeMetadata &runtime, PlanHostContext *context)
{
    if (context == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *context = PlanHostContext {};

    if (arguments.topkExperts == nullptr || arguments.tokensPerExpert == nullptr ||
        arguments.globalRankIds == nullptr || arguments.config == nullptr || arguments.plan == nullptr ||
        arguments.localWorkspace == nullptr || arguments.registeredMetaWorkspace == nullptr ||
        arguments.waitIterations == 0 || arguments.stream == nullptr || runtime.hostCommArgs == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!IsAligned(arguments.config, alignof(TileXRMoonEPPlanConfig)) ||
        !IsAligned(arguments.plan, alignof(TileXRMoonEPPlanDesc))) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const bool hasRemoteExperts = arguments.remoteExperts != nullptr;
    const bool hasExpertTargets = arguments.expertTargets != nullptr;
    if (hasRemoteExperts != hasExpertTargets ||
        (hasRemoteExperts && (!ValidInt32Pointer(arguments.remoteExperts) ||
            !IsAligned(arguments.expertTargets, alignof(uint64_t)))) ||
        !ValidInt32Pointer(arguments.topkExperts) || !ValidInt32Pointer(arguments.tokensPerExpert) ||
        !ValidInt32Pointer(arguments.globalRankIds) || !ValidPlanPointers(*arguments.plan) ||
        !IsAligned(arguments.localWorkspace, kPlanWorkspaceAlignment) ||
        !IsAligned(arguments.registeredMetaWorkspace, kPlanWorkspaceAlignment)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const TileXR::CommArgs &commArgs = *runtime.hostCommArgs;
    if (commArgs.rankSize <= 0 || commArgs.rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        commArgs.rank < 0 || commArgs.rank >= commArgs.rankSize) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if ((commArgs.extraFlag & TileXR::ExtraFlag::TOPO_910A5) == 0) {
        return TileXR::TILEXR_ERROR_NOT_SUPPORT;
    }
    if (!PlanIdentityMatches(arguments, commArgs.rankSize)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    PlanWorkspaceLayout layout {};
    int ret = BuildPlanWorkspaceLayout(
        commArgs.rankSize, arguments.s, arguments.topK, arguments.expertNum, *arguments.config, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    ret = ValidatePlanWorkspaceBytes(
        layout, arguments.localWorkspaceBytes, arguments.registeredMetaBytes);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    ret = ValidateRuntimeMetadata(runtime, arguments.expertNum);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    PlanCallHeader header {};
    ret = BuildPlanCallHeader(commArgs.rankSize, arguments.s, arguments.topK, arguments.expertNum,
        *arguments.config, arguments.plan->epoch, 0, &header);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    context->layout = layout;
    context->callHeader = header;
    context->rank = commArgs.rank;
    context->deviceCommArgs = runtime.deviceCommArgs;
    return TileXR::TILEXR_SUCCESS;
}

} // namespace Plan
} // namespace TileXREp
