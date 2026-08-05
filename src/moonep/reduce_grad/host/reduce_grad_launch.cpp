#include "reduce_grad_host.h"

#include "reduce_grad_common.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

extern "C" void tilexr_moonep_reduce_grad_kernel(
    GM_ADDR commArgs, GM_ADDR expertsToCopy, GM_ADDR gate, GM_ADDR up, GM_ADDR down,
    GM_ADDR workspace, GM_ADDR status, int64_t rank, int64_t rankSize,
    int64_t expertCount, int64_t expertsPerRank, int64_t controlBlockCount,
    uint64_t gateRowElements, uint64_t upRowElements, uint64_t downRowElements,
    uint64_t gateRowBytes, uint64_t upRowBytes, uint64_t downRowBytes,
    uint32_t gateTransport, uint32_t upTransport, uint32_t downTransport,
    uint32_t reserved0, uint64_t peerRecordBaseOffset, uint64_t peerHalfBytes,
    uint64_t peerSlotStrideBytes, uint64_t peerChunkPayloadBytes,
    uint64_t udmaStateOffset, uint64_t udmaOutboundOffset,
    uint64_t udmaInboundOffset, uint64_t udmaChunkBytes, uint64_t workspaceBytes,
    uint64_t waitIterations, int64_t magic);

namespace TileXRMoonEp {

int TileXRMoonEpLaunchReduceGradKernel(const ReduceGradParams &params,
    const ReduceGradLaunchContext &context)
{
    int64_t magic = 0;
    const int ret = TileXRCommNextMagic(params.comm, &magic);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    const ReduceGradLayout &layout = context.layout;
    ReduceGradKernelArgs args {
        context.devArgs,
        reinterpret_cast<GM_ADDR>(params.plan->expertsToCopy),
        reinterpret_cast<GM_ADDR>(params.gradients[kReduceGradGate]->data),
        reinterpret_cast<GM_ADDR>(params.gradients[kReduceGradUp]->data),
        reinterpret_cast<GM_ADDR>(params.gradients[kReduceGradDown]->data),
        reinterpret_cast<GM_ADDR>(params.workspace),
        reinterpret_cast<GM_ADDR>(params.status->data),
        layout.rank,
        layout.rankSize,
        layout.expertCount,
        layout.expertsPerRank,
        layout.controlBlockCount,
        layout.rowElements[kReduceGradGate],
        layout.rowElements[kReduceGradUp],
        layout.rowElements[kReduceGradDown],
        layout.rowBytes[kReduceGradGate],
        layout.rowBytes[kReduceGradUp],
        layout.rowBytes[kReduceGradDown],
        layout.transports[kReduceGradGate],
        layout.transports[kReduceGradUp],
        layout.transports[kReduceGradDown],
        0,
        layout.peerRecordBaseOffset,
        layout.peerHalfBytes,
        layout.peerSlotStrideBytes,
        layout.peerChunkPayloadBytes,
        layout.udmaStateOffset,
        layout.udmaOutboundOffset,
        layout.udmaInboundOffset,
        layout.udmaChunkBytes,
        layout.workspaceBytes,
        params.waitIterations,
        magic,
    };

    rtArgsEx_t argsInfo {};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo {};
    cfgInfo.schemMode = 1;
    const rtError_t launchRet = rtKernelLaunchWithFlagV2(
        reinterpret_cast<const void *>(tilexr_moonep_reduce_grad_kernel),
        static_cast<uint32_t>(layout.blockDim), &argsInfo, nullptr,
        static_cast<rtStream_t>(params.stream), 0, &cfgInfo);
    return launchRet == RT_ERROR_NONE ? TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_MKIRT;
}

} // namespace TileXRMoonEp
