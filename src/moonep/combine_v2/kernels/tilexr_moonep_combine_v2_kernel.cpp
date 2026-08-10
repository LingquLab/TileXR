#include "tilexr_moonep_combine_v2_kernel.h"

extern "C" __global__ __aicore__ void
tilexr_moonep_combine_v2_kernel(GM_ADDR commArgs,
    GM_ADDR registeredWorkspace, GM_ADDR dstLocal, uint64_t profileOffset,
    uint64_t scratchEpoch0Offset, uint64_t scratchEpoch1Offset,
    uint64_t doneOffset, uint64_t grantOffset,
    uint64_t controlSourceOffset, uint64_t failureOffset, int64_t bs,
    int64_t h, int64_t topK, int64_t nvS, int64_t magic)
{
    __gm__ TileXR::CommArgs *args =
        reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
    const uint32_t rankSize = args == nullptr || args->rankSize <= 0 ? 0U :
        static_cast<uint32_t>(args->rankSize);
    const uint32_t activeCoreCount =
        TileXRMoonEp::MoonEpCombineV2RankSizeSupported(rankSize) ?
        TileXRMoonEp::MoonEpCombineV2ActiveCoreCount(rankSize) : 0U;
    if (static_cast<uint32_t>(GetBlockIdx()) >= activeCoreCount) {
        return;
    }
    AscendC::TPipe pipe;
    MoonEpCombineV2 op;
    op.Init(commArgs, registeredWorkspace, dstLocal, profileOffset,
        scratchEpoch0Offset, scratchEpoch1Offset, doneOffset, grantOffset,
        controlSourceOffset, failureOffset, bs, h, topK, nvS, magic, &pipe);
    op.Process();
}
