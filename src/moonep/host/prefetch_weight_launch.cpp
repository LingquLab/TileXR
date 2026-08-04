#include "prefetch_weight_launch.h"

#include "runtime/kernel.h"
#include "tilexr_types.h"

extern "C" void tilexr_moonep_prefetch_weight_kernel(GM_ADDR commArgs,
    GM_ADDR expertsToCopy, GM_ADDR gate, GM_ADDR up, GM_ADDR down, GM_ADDR status,
    uint64_t gateOffset, uint64_t upOffset, uint64_t downOffset,
    uint32_t gateRowBytes, uint32_t upRowBytes, uint32_t downRowBytes,
    int32_t rank, int32_t rankSize, int32_t expertsPerRank, uint32_t qpNum);

namespace TileXRMoonEp {

int LaunchPrefetchWeight(const PrefetchWeightLayout &layout, GM_ADDR commArgs,
    GM_ADDR expertsToCopy, GM_ADDR status, aclrtStream stream)
{
    struct KernelArgs {
        GM_ADDR commArgs;
        GM_ADDR expertsToCopy;
        GM_ADDR gate;
        GM_ADDR up;
        GM_ADDR down;
        GM_ADDR status;
        uint64_t gateOffset;
        uint64_t upOffset;
        uint64_t downOffset;
        uint32_t gateRowBytes;
        uint32_t upRowBytes;
        uint32_t downRowBytes;
        int32_t rank;
        int32_t rankSize;
        int32_t expertsPerRank;
        uint32_t qpNum;
    } args {
        commArgs, expertsToCopy, layout.gate.localBase, layout.up.localBase,
        layout.down.localBase, status, layout.gate.registryOffset,
        layout.up.registryOffset, layout.down.registryOffset,
        layout.gate.rowBytes, layout.up.rowBytes, layout.down.rowBytes,
        layout.rank, layout.rankSize, layout.expertsPerRank, layout.qpNum
    };

    rtArgsEx_t argsInfo {};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo {};
    cfgInfo.schemMode = 1;
    const rtError_t ret = rtKernelLaunchWithFlagV2(
        reinterpret_cast<const void *>(tilexr_moonep_prefetch_weight_kernel),
        layout.blockDim, &argsInfo, nullptr, static_cast<rtStream_t>(stream), 0, &cfgInfo);
    return ret == RT_ERROR_NONE ? TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_MKIRT;
}

} // namespace TileXRMoonEp
