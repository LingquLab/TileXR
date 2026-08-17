#include "tilexr_moonep_combine_v2_kernel.h"
#include "tilexr_moonep_combine_v2_group_kernel.h"

extern "C" __global__ __aicore__ void
tilexr_moonep_combine_v2_kernel(GM_ADDR commArgs,
    GM_ADDR registeredWorkspace, GM_ADDR dstLocal, uint64_t profileOffset,
    uint64_t scratchEpoch0Offset, uint64_t scratchEpoch1Offset,
    uint64_t doneOffset, uint64_t grantOffset,
    uint64_t controlSourceOffset, uint64_t failureOffset,
    uint64_t fullSyncReceiveOffset, uint64_t fullSyncSourceOffset,
    uint64_t fullSyncBarrierOffset,
    uint64_t outputOffset, int64_t bs, int64_t h, int64_t topK,
    int64_t nvS, uint64_t rowBytes, uint64_t reduceHidden,
    int64_t magic)
{
    AscendC::TPipe pipe;
    __gm__ TileXR::CommArgs *args =
        reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgs);
    if (args != nullptr && args->rankSize == 128) {
        TileXRGroup128::MoonEpCombineV2Group op;
        op.Init(commArgs, registeredWorkspace, dstLocal, profileOffset,
            scratchEpoch0Offset, scratchEpoch1Offset, doneOffset, grantOffset,
            controlSourceOffset, failureOffset, fullSyncReceiveOffset,
            fullSyncSourceOffset, fullSyncBarrierOffset, outputOffset, bs, h,
            topK, nvS, rowBytes, reduceHidden != 0U, magic, &pipe);
        op.Process();
        return;
    }

    MoonEpCombineV2 op;
    op.Init(commArgs, registeredWorkspace, dstLocal, profileOffset,
        scratchEpoch0Offset, scratchEpoch1Offset, doneOffset, grantOffset,
        controlSourceOffset, failureOffset, fullSyncReceiveOffset,
        fullSyncSourceOffset, fullSyncBarrierOffset, outputOffset, bs, h,
        topK, nvS, rowBytes, reduceHidden != 0U, magic, &pipe);
    op.Process();
}
