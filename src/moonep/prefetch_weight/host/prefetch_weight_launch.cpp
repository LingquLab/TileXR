#include "prefetch_weight_launch.h"

#include <cstddef>

#include "moonep_kernel_launch.h"

extern "C" {
extern const unsigned char TileXRMoonEpPrefetchWeightKernelBinaryData[];
extern const std::size_t TileXRMoonEpPrefetchWeightKernelBinarySize;
}

namespace TileXRMoonEp {
namespace {

constexpr const char *kPrefetchWeightKernelName = "tilexr_moonep_prefetch_weight_kernel";
KernelRegistrationState g_prefetchWeightRegistration;

} // namespace

int TileXRMoonEpLaunchPrefetchWeightKernel(
    const PrefetchWeightParams &params, const PrefetchWeightLaunchContext &context)
{
    struct PrefetchWeightKernelArgs {
        GM_ADDR commArgs;
        GM_ADDR expertsToCopy;
        GM_ADDR fullGateWeight;
        GM_ADDR fullUpWeight;
        GM_ADDR fullDownWeight;
        GM_ADDR status;
        int64_t e;
        int64_t b;
        int64_t expertsPerRank;
        int64_t gateRowBytes;
        int64_t gateChunkBytes;
        int64_t gateChunkCount;
        int64_t upRowBytes;
        int64_t upChunkBytes;
        int64_t upChunkCount;
        int64_t downRowBytes;
        int64_t downChunkBytes;
        int64_t downChunkCount;
        int64_t iterationCount;
        uint64_t waitIterations;
        int64_t magic;
    } args {
        context.devArgs,
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.expertsToCopy)),
        reinterpret_cast<GM_ADDR>(params.fullGateWeight),
        reinterpret_cast<GM_ADDR>(params.fullUpWeight),
        reinterpret_cast<GM_ADDR>(params.fullDownWeight),
        reinterpret_cast<GM_ADDR>(params.status), context.layout.e, context.layout.b,
        context.layout.expertsPerRank, context.layout.gate.rowBytes,
        context.layout.gate.chunkBytes, context.layout.gate.chunkCount,
        context.layout.up.rowBytes, context.layout.up.chunkBytes,
        context.layout.up.chunkCount, context.layout.down.rowBytes,
        context.layout.down.chunkBytes, context.layout.down.chunkCount,
        context.layout.iterationCount, context.waitIterations, context.magic
    };

    return LaunchRegisteredMoonEpKernel(g_prefetchWeightRegistration,
        TileXRMoonEpPrefetchWeightKernelBinaryData,
        TileXRMoonEpPrefetchWeightKernelBinarySize,
        KernelSignature(kPrefetchWeightKernelSignature),
        kPrefetchWeightKernelName, "prefetch-weight",
        static_cast<uint32_t>(context.layout.blockDim), &args, sizeof(args),
        static_cast<rtStream_t>(params.stream));
}

} // namespace TileXRMoonEp
