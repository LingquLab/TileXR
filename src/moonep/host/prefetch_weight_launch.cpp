#include "prefetch_weight_launch.h"

#include "tilexr_types.h"

extern "C" void launch_tilexr_moonep_prefetch_weight_kernel(uint32_t blockDim,
    void *stream, GM_ADDR commArgs, GM_ADDR expertsToCopy, GM_ADDR gate, GM_ADDR up,
    GM_ADDR down, GM_ADDR status, uint64_t gateOffset, uint64_t upOffset,
    uint64_t downOffset, uint32_t gateRowBytes, uint32_t upRowBytes,
    uint32_t downRowBytes, int32_t rank, int32_t rankSize, int32_t expertsPerRank,
    uint32_t qpNum);

namespace TileXRMoonEp {

int LaunchPrefetchWeight(const PrefetchWeightLayout &layout, GM_ADDR commArgs,
    GM_ADDR expertsToCopy, GM_ADDR status, aclrtStream stream)
{
    launch_tilexr_moonep_prefetch_weight_kernel(
        layout.blockDim, stream,
        commArgs, expertsToCopy, layout.gate.localBase, layout.up.localBase,
        layout.down.localBase, status, layout.gate.registryOffset,
        layout.up.registryOffset, layout.down.registryOffset,
        layout.gate.rowBytes, layout.up.rowBytes, layout.down.rowBytes,
        layout.rank, layout.rankSize, layout.expertsPerRank, layout.qpNum
    );
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXRMoonEp
