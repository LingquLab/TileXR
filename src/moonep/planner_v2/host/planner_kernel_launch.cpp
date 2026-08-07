// This host-only launch translation unit must not emit Ascend C's per-kernel
// tiling-key storage. The device-kernel translation unit owns that symbol.
// Defining the customization hook before kernel_operator.h prevents a second
// g_tilingKey definition when Bisheng links both translation units.
#define TILING_KEY_VAR 0ULL
#include "kernel_operator.h"

#include "comm_args.h"
#include "runtime/kernel.h"

extern "C" __global__ __aicore__ void tilexr_ep_plan_kernel(GM_ADDR commArgsGM,
    GM_ADDR topkExpertsGM, GM_ADDR tokensPerExpertGM, GM_ADDR globalRankIdsGM, GM_ADDR dstGM,
    GM_ADDR cuSeqlensGM, GM_ADDR expertsToCopyGM, GM_ADDR remoteExpertsGM,
    GM_ADDR expertTargetsGM, GM_ADDR remoteStatsGM, GM_ADDR statusGM,
    GM_ADDR localWorkspaceGM, GM_ADDR metaWorkspaceGM, int64_t rank, int64_t rankSize,
    int64_t s, int64_t topK, int64_t expertNum, int64_t prefetchSlots,
    int64_t rankTokenCapacity, int64_t nvS, int64_t tokenPadding, int64_t tokenRouteLimitPerPair,
    int32_t cardsPerServer, int32_t cardsPerCabinet, int32_t crossCandidateCount,
    uint64_t epoch, uint64_t waitIterations, int64_t magic);

rtError_t launch_tilexr_ep_plan_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs,
    GM_ADDR topkExperts, GM_ADDR tokensPerExpert, GM_ADDR globalRankIds, GM_ADDR dst,
    GM_ADDR cuSeqlens, GM_ADDR expertsToCopy, GM_ADDR remoteExperts, GM_ADDR expertTargets,
    GM_ADDR remoteStats, GM_ADDR status,
    GM_ADDR localWorkspace, GM_ADDR metaWorkspace, int64_t rank, int64_t rankSize,
    int64_t s, int64_t topK, int64_t expertNum, int64_t prefetchSlots,
    int64_t rankTokenCapacity, int64_t nvS, int64_t tokenPadding, int64_t tokenRouteLimitPerPair,
    int32_t cardsPerServer, int32_t cardsPerCabinet, int32_t crossCandidateCount,
    uint64_t epoch, uint64_t waitIterations, int64_t magic)
{
#if defined(TILEXR_PLAN_DIRECT_RT_V2_LAUNCH)
    struct PlanKernelArgs {
        GM_ADDR commArgs;
        GM_ADDR topkExperts;
        GM_ADDR tokensPerExpert;
        GM_ADDR globalRankIds;
        GM_ADDR dst;
        GM_ADDR cuSeqlens;
        GM_ADDR expertsToCopy;
        GM_ADDR remoteExperts;
        GM_ADDR expertTargets;
        GM_ADDR remoteStats;
        GM_ADDR status;
        GM_ADDR localWorkspace;
        GM_ADDR metaWorkspace;
        int64_t rank;
        int64_t rankSize;
        int64_t s;
        int64_t topK;
        int64_t expertNum;
        int64_t prefetchSlots;
        int64_t rankTokenCapacity;
        int64_t nvS;
        int64_t tokenPadding;
        int64_t tokenRouteLimitPerPair;
        int32_t cardsPerServer;
        int32_t cardsPerCabinet;
        int32_t crossCandidateCount;
        uint64_t epoch;
        uint64_t waitIterations;
        int64_t magic;
    } args {
        commArgs, topkExperts, tokensPerExpert, globalRankIds, dst, cuSeqlens,
        expertsToCopy, remoteExperts, expertTargets, remoteStats, status, localWorkspace, metaWorkspace, rank,
        rankSize, s, topK, expertNum, prefetchSlots, rankTokenCapacity, nvS,
        tokenPadding, tokenRouteLimitPerPair, cardsPerServer, cardsPerCabinet,
        crossCandidateCount, epoch, waitIterations, magic
    };

    rtArgsEx_t argsInfo {};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo {};
    cfgInfo.schemMode = 1;
    return rtKernelLaunchWithFlagV2(
        reinterpret_cast<const void *>(tilexr_ep_plan_kernel), blockDim,
        &argsInfo, nullptr, static_cast<rtStream_t>(stream), 0, &cfgInfo);
#else
    // Keep the compiler-generated registration path used by CANN 9.1 while
    // owning all Host launch code outside the device-kernel translation unit.
    tilexr_ep_plan_kernel<<<blockDim, nullptr, stream>>>(commArgs, topkExperts,
        tokensPerExpert, globalRankIds, dst, cuSeqlens, expertsToCopy, remoteExperts,
        expertTargets, remoteStats, status, localWorkspace, metaWorkspace, rank, rankSize, s, topK, expertNum,
        prefetchSlots, rankTokenCapacity, nvS, tokenPadding, tokenRouteLimitPerPair,
        cardsPerServer, cardsPerCabinet, crossCandidateCount, epoch, waitIterations,
        magic);
    return RT_ERROR_NONE;
#endif
}
