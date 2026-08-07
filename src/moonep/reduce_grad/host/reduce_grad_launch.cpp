#include "reduce_grad_launch.h"

#include <cstddef>

#include "moonep_kernel_launch.h"

extern "C" {
extern const unsigned char TileXRMoonEpReduceGradKernelBinaryData[];
extern const std::size_t TileXRMoonEpReduceGradKernelBinarySize;
}

namespace TileXRMoonEp {
namespace {

constexpr const char *kReduceGradKernelName = "tilexr_moonep_reduce_grad_kernel";
KernelRegistrationState g_reduceGradRegistration;

} // namespace

int TileXRMoonEpLaunchReduceGradKernel(
    const ReduceGradParams &params, const ReduceGradLaunchContext &context)
{
    struct ReduceGradKernelArgs {
        GM_ADDR commArgs;
        GM_ADDR expertsToCopy;
        GM_ADDR fullGateGrad;
        GM_ADDR fullUpGrad;
        GM_ADDR fullDownGrad;
        GM_ADDR gateReduceBuffer;
        GM_ADDR upReduceBuffer;
        GM_ADDR downReduceBuffer;
        GM_ADDR status;
        int64_t e;
        int64_t b;
        int64_t expertsPerRank;
        int64_t gateRowBytes;
        int64_t gateChunkBytes;
        int64_t gateChunkStride;
        int64_t gateChunkCount;
        int64_t upRowBytes;
        int64_t upChunkBytes;
        int64_t upChunkStride;
        int64_t upChunkCount;
        int64_t downRowBytes;
        int64_t downChunkBytes;
        int64_t downChunkStride;
        int64_t downChunkCount;
        int64_t iterationCount;
        uint64_t waitIterations;
        int64_t magic;
    } args {
        context.devArgs,
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.expertsToCopy)),
        reinterpret_cast<GM_ADDR>(params.fullGateGrad),
        reinterpret_cast<GM_ADDR>(params.fullUpGrad),
        reinterpret_cast<GM_ADDR>(params.fullDownGrad),
        reinterpret_cast<GM_ADDR>(params.gateReduceBuffer),
        reinterpret_cast<GM_ADDR>(params.upReduceBuffer),
        reinterpret_cast<GM_ADDR>(params.downReduceBuffer),
        reinterpret_cast<GM_ADDR>(params.status), context.layout.e, context.layout.b,
        context.layout.expertsPerRank, context.layout.gate.rowBytes,
        context.layout.gate.chunkBytes, context.layout.gate.chunkStride,
        context.layout.gate.chunkCount, context.layout.up.rowBytes,
        context.layout.up.chunkBytes, context.layout.up.chunkStride,
        context.layout.up.chunkCount, context.layout.down.rowBytes,
        context.layout.down.chunkBytes, context.layout.down.chunkStride,
        context.layout.down.chunkCount, context.layout.iterationCount,
        context.waitIterations, context.magic
    };
    return LaunchRegisteredMoonEpKernel(g_reduceGradRegistration,
        TileXRMoonEpReduceGradKernelBinaryData, TileXRMoonEpReduceGradKernelBinarySize,
        KernelSignature(kReduceGradKernelSignature),
        kReduceGradKernelName, "reduce-grad",
        static_cast<uint32_t>(context.layout.blockDim), &args, sizeof(args),
        static_cast<rtStream_t>(params.stream));
}

} // namespace TileXRMoonEp
