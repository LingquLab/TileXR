#include "combine_launch.h"

#include <cstddef>

#include "moonep_kernel_launch.h"

extern "C" {
extern const unsigned char TileXRMoonEpCombineKernelBinaryData[];
extern const std::size_t TileXRMoonEpCombineKernelBinarySize;
}

namespace TileXRMoonEp {
namespace {

constexpr const char *kCombineKernelName = "tilexr_moonep_combine_kernel";
KernelRegistrationState g_combineRegistration;

} // namespace

int TileXRMoonEpLaunchCombineKernel(
    const CombineParams &params, const CombineLaunchContext &context)
{
    struct CombineKernelArgs {
        GM_ADDR commArgs;
        GM_ADDR dstLocal;
        GM_ADDR dst;
        GM_ADDR dupGroups;
        GM_ADDR dupLoffs;
        GM_ADDR dupCounts;
        GM_ADDR hiddenNvsh;
        GM_ADDR routeWeightsNvs;
        GM_ADDR hiddenSh;
        GM_ADDR routeWeightsSk;
        GM_ADDR status;
        int64_t s;
        int64_t k;
        int64_t n;
        int64_t nvS;
        int64_t hiddenRowBytes;
        int64_t hiddenChunkBytes;
        int64_t hiddenChunkStride;
        int64_t chunkCount;
        uint64_t sourceHiddenOffset;
        uint64_t receiveHiddenOffset;
        uint64_t hiddenPayloadBytes;
        uint64_t sourceWeightsOffset;
        uint64_t receiveWeightsOffset;
        uint64_t routeWeightsBytes;
        uint64_t duplicateMaskOffset;
        uint64_t doneOffset;
        uint64_t coreStatusOffset;
        uint64_t windowBytes;
        uint64_t waitIterations;
        uint64_t flags;
        int64_t magic;
    } args {
        static_cast<GM_ADDR>(context.devArgs),
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.dstLocal)),
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.dst)),
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.dupGroups)),
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.dupLoffs)),
        reinterpret_cast<GM_ADDR>(const_cast<int32_t *>(params.dupCounts)),
        reinterpret_cast<GM_ADDR>(const_cast<void *>(params.hiddenNvsh)),
        reinterpret_cast<GM_ADDR>(const_cast<float *>(params.routeWeightsNvs)),
        reinterpret_cast<GM_ADDR>(params.hiddenSh),
        reinterpret_cast<GM_ADDR>(params.routeWeightsSk),
        reinterpret_cast<GM_ADDR>(params.status), context.layout.s, context.layout.k,
        context.layout.n, context.layout.nvS,
        static_cast<int64_t>(context.layout.hiddenRowBytes),
        static_cast<int64_t>(context.layout.hiddenChunkBytes),
        static_cast<int64_t>(context.layout.hiddenChunkStride), context.layout.chunkCount,
        context.layout.sourceHiddenOffset, context.layout.receiveHiddenOffset,
        context.layout.hiddenPayloadBytes, context.layout.sourceWeightsOffset,
        context.layout.receiveWeightsOffset, context.layout.routeWeightsBytes,
        context.layout.duplicateMaskOffset, context.layout.doneOffset,
        context.layout.coreStatusOffset, context.layout.windowBytes,
        context.waitIterations, params.flags,
        context.magic
    };

    return LaunchRegisteredMoonEpKernel(g_combineRegistration,
        TileXRMoonEpCombineKernelBinaryData, TileXRMoonEpCombineKernelBinarySize,
        KernelSignature(kCombineKernelSignature),
        kCombineKernelName, "combine", static_cast<uint32_t>(context.layout.blockDim),
        &args, sizeof(args), static_cast<rtStream_t>(params.stream));
}

} // namespace TileXRMoonEp
