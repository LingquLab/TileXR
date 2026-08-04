#include "planner_launch.h"

#include "planner_common.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

extern "C" void tilexr_moonep_planner_kernel(
    GM_ADDR commArgs, GM_ADDR topkExpertIds, GM_ADDR tokensPerExpert, GM_ADDR workspace,
    GM_ADDR dst, GM_ADDR cuSeqlens, GM_ADDR expertsToCopy, GM_ADDR remoteStats,
    GM_ADDR plannerStatus, int64_t s, int64_t k, int64_t expertCount,
    int64_t expertsPerRank, int64_t routeCount, int64_t dispatchedCapacity,
    uint64_t waitIterations, uint64_t tpePrefixOffset, uint64_t blockHistogramOffset,
    uint64_t allocPrefixOffset, uint64_t expertOffsetsOffset, uint64_t zOffset,
    uint64_t groupTotalsOffset, int64_t magic);

namespace TileXRMoonEp {

int TileXRMoonEpLaunchKernel(const PlannerParams &params, const PlannerLaunchContext &context)
{
    int64_t magic = 0;
    const int ret = TileXRCommNextMagic(params.comm, &magic);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    const PlannerLayout &layout = context.layout;
    struct PlannerKernelArgs {
        GM_ADDR commArgs;
        GM_ADDR topkExpertIds;
        GM_ADDR tokensPerExpert;
        GM_ADDR workspace;
        GM_ADDR dst;
        GM_ADDR cuSeqlens;
        GM_ADDR expertsToCopy;
        GM_ADDR remoteStats;
        GM_ADDR plannerStatus;
        int64_t s;
        int64_t k;
        int64_t expertCount;
        int64_t expertsPerRank;
        int64_t routeCount;
        int64_t dispatchedCapacity;
        uint64_t waitIterations;
        uint64_t tpePrefixOffset;
        uint64_t blockHistogramOffset;
        uint64_t allocPrefixOffset;
        uint64_t expertOffsetsOffset;
        uint64_t zOffset;
        uint64_t groupTotalsOffset;
        int64_t magic;
    } args {
        context.devArgs,
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.topkExpertIds)),
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.tokensPerExpert)),
        static_cast<GM_ADDR>(params.workspace), reinterpret_cast<GM_ADDR>(params.dst),
        reinterpret_cast<GM_ADDR>(params.cuSeqlens),
        reinterpret_cast<GM_ADDR>(params.expertsToCopy),
        reinterpret_cast<GM_ADDR>(params.remoteStats),
        reinterpret_cast<GM_ADDR>(params.plannerStatus), layout.s, layout.k,
        layout.expertCount, layout.expertsPerRank, layout.routeCount,
        layout.dispatchedCapacity, params.waitIterations, layout.tpePrefixOffset,
        layout.blockHistogramOffset, layout.allocPrefixOffset,
        layout.expertOffsetsOffset, layout.zOffset, layout.groupTotalsOffset, magic
    };

    rtArgsEx_t argsInfo {};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfgInfo {};
    cfgInfo.schemMode = 1;

    const rtError_t launchRet = rtKernelLaunchWithFlagV2(
        reinterpret_cast<const void *>(tilexr_moonep_planner_kernel),
        static_cast<uint32_t>(layout.blockDim), &argsInfo, nullptr,
        static_cast<rtStream_t>(params.stream), 0, &cfgInfo);
    return launchRet == RT_ERROR_NONE ? TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_MKIRT;
}

} // namespace TileXRMoonEp
