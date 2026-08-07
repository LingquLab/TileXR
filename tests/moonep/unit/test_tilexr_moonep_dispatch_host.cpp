#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "dispatch_host.h"
#include "dispatch_launch.h"
#include "moonep_peer_window.h"
#include "tilexr_types.h"

namespace TileXRMoonEp {
int TileXRMoonEpRunDispatchV1(
    const TileXRMoonEpDispatchArgsV1 *args, aclrtStream stream);
}

namespace {

int failures = 0;
int hostArgsReturn = TileXR::TILEXR_SUCCESS;
int devArgsReturn = TileXR::TILEXR_SUCCESS;
int magicReturn = TileXR::TILEXR_SUCCESS;
int launchReturn = TileXR::TILEXR_SUCCESS;
int hostArgsCalls = 0;
int devArgsCalls = 0;
int magicCalls = 0;
int launchCalls = 0;
int64_t nextMagic = 77;
TileXR::CommArgs commArgs {};
GM_ADDR devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
TileXRMoonEp::DispatchParams launchedParams {};
TileXRMoonEp::DispatchLaunchContext launchedContext {};

void Check(bool condition, const std::string &message)
{
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

void CheckStatus(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
        ++failures;
    }
}

TileXR::CommArgs ValidCommArgs()
{
    TileXR::CommArgs args {};
    args.rank = 0;
    args.localRank = 0;
    args.rankSize = 2;
    args.localRankSize = 2;
    args.extraFlag = TileXR::ExtraFlag::TOPO_910A5;
    args.peerMems[0] = reinterpret_cast<GM_ADDR>(uintptr_t {0x100000});
    args.peerMems[1] = reinterpret_cast<GM_ADDR>(uintptr_t {0x200000});
    return args;
}

void Reset()
{
    hostArgsReturn = devArgsReturn = magicReturn = launchReturn = TileXR::TILEXR_SUCCESS;
    hostArgsCalls = devArgsCalls = magicCalls = launchCalls = 0;
    nextMagic = 77;
    commArgs = ValidCommArgs();
    devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
    launchedParams = TileXRMoonEp::DispatchParams {};
    launchedContext = TileXRMoonEp::DispatchLaunchContext {};
}

TileXRMoonEpPlanV1 ValidPlan()
{
    TileXRMoonEpPlanV1 plan {};
    plan.structSize = sizeof(plan);
    plan.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    plan.n = 4;
    plan.r = 2;
    plan.e = 8;
    plan.b = 2;
    plan.nvS = 8;
    plan.k = 2;
    plan.dst = reinterpret_cast<void *>(uintptr_t {0x3000});
    plan.expertsToCopy = reinterpret_cast<void *>(uintptr_t {0x3100});
    plan.zeroFillRanges = reinterpret_cast<void *>(uintptr_t {0x3200});
    plan.remoteStats = reinterpret_cast<void *>(uintptr_t {0x3300});
    plan.dupGroups = reinterpret_cast<void *>(uintptr_t {0x3400});
    plan.dupLoffs = reinterpret_cast<void *>(uintptr_t {0x3500});
    plan.dupCounts = reinterpret_cast<void *>(uintptr_t {0x3600});
    plan.status = reinterpret_cast<void *>(uintptr_t {0x3700});
    return plan;
}

TileXRMoonEpTensorV1 Tensor(void *data, uint32_t dtype, uint32_t rank,
    int64_t dim0, int64_t dim1, uint64_t elements)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = data;
    tensor.elementCount = elements;
    tensor.dtype = dtype;
    tensor.rank = rank;
    tensor.shape[0] = dim0;
    tensor.shape[1] = dim1;
    return tensor;
}

TileXRMoonEpDispatchArgsV1 Args(const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *hiddenSh, const TileXRMoonEpTensorV1 *weightsSk,
    TileXRMoonEpTensorV1 *hiddenNvsh, TileXRMoonEpTensorV1 *weightsNvs, uint64_t flags)
{
    TileXRMoonEpDispatchArgsV1 args {};
    args.structSize = sizeof(args);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    args.comm = reinterpret_cast<TileXRCommPtr>(uintptr_t {0x1000});
    args.plan = plan;
    args.hiddenSh = hiddenSh;
    args.routeWeightsSk = weightsSk;
    args.hiddenNvsh = hiddenNvsh;
    args.routeWeightsNvs = weightsNvs;
    args.flags = flags;
    return args;
}

void TestPairedFreshLaunch()
{
    Reset();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 hiddenSh = Tensor(reinterpret_cast<void *>(uintptr_t {0x4000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 2, 17, 34);
    TileXRMoonEpTensorV1 hiddenNvsh = Tensor(reinterpret_cast<void *>(uintptr_t {0x5000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 8, 17, 136);
    TileXRMoonEpTensorV1 weightsSk = Tensor(reinterpret_cast<void *>(uintptr_t {0x6000}),
        TILEXR_MOONEP_DTYPE_FLOAT32, 2, 2, 2, 4);
    TileXRMoonEpTensorV1 weightsNvs = Tensor(reinterpret_cast<void *>(uintptr_t {0x7000}),
        TILEXR_MOONEP_DTYPE_FLOAT32, 1, 8, 0, 8);
    TileXRMoonEpDispatchArgsV1 args = Args(&plan, &hiddenSh, &weightsSk,
        &hiddenNvsh, &weightsNvs, TILEXR_MOONEP_FLAG_BUILD_DEDUP);
    aclrtStream stream = reinterpret_cast<aclrtStream>(uintptr_t {0x8000});

    CheckStatus("paired fresh", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_SUCCESS);
    Check(hostArgsCalls == 1 && devArgsCalls == 1 && magicCalls == 1 && launchCalls == 1,
        "paired launch call counts mismatch");
    Check(launchedParams.dst == plan.dst &&
        launchedParams.zeroFillRanges == plan.zeroFillRanges &&
        launchedParams.dupGroups == plan.dupGroups && launchedParams.dupLoffs == plan.dupLoffs &&
        launchedParams.dupCounts == plan.dupCounts && launchedParams.hiddenSh == hiddenSh.data &&
        launchedParams.hiddenNvsh == hiddenNvsh.data &&
        launchedParams.routeWeightsSk == weightsSk.data &&
        launchedParams.routeWeightsNvs == weightsNvs.data &&
        launchedParams.status == plan.status && launchedParams.flags == args.flags &&
        launchedParams.stream == stream, "paired launch pointers mismatch");
    Check(launchedContext.devArgs == devArgs && launchedContext.magic == nextMagic &&
        launchedContext.waitIterations == TileXRMoonEp::kMoonEpPeerWaitIterations,
        "paired launch context mismatch");
    Check(launchedContext.layout.n == 4 && launchedContext.layout.nvS == 8 &&
        launchedContext.layout.hiddenRowBytes == 34 &&
        launchedContext.layout.hiddenChunkStride == 64 &&
        launchedContext.layout.routeWeightsBytes == 32 &&
        launchedContext.layout.dedupParentsBytes == 32,
        "paired layout mismatch");
}

void TestReuseAndValidation()
{
    Reset();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 hiddenSh = Tensor(reinterpret_cast<void *>(uintptr_t {0x4000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 2, 16, 32);
    TileXRMoonEpTensorV1 hiddenNvsh = Tensor(reinterpret_cast<void *>(uintptr_t {0x5000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 8, 16, 128);
    TileXRMoonEpTensorV1 weightsSk = Tensor(reinterpret_cast<void *>(uintptr_t {0x6000}),
        TILEXR_MOONEP_DTYPE_FLOAT32, 2, 2, 2, 4);
    TileXRMoonEpDispatchArgsV1 args = Args(&plan, &hiddenSh, nullptr, &hiddenNvsh,
        nullptr, TILEXR_MOONEP_FLAG_SKIP_INTER_RANK_SYNC);
    aclrtStream stream = reinterpret_cast<aclrtStream>(uintptr_t {0x8000});

    plan.b = 1;
    CheckStatus("B below experts-per-rank", TileXRMoonEp::TileXRMoonEpRunDispatchV1(
        &args, stream), TILEXR_MOONEP_SUCCESS);
    Check(launchedParams.dupGroups == plan.dupGroups &&
        launchedContext.layout.dedupParentsBytes == 0,
        "reuse path did not preserve saved dedup contract");

    args.routeWeightsSk = &weightsSk;
    CheckStatus("unpaired weights", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.routeWeightsSk = nullptr;
    args.flags = TILEXR_MOONEP_FLAG_ZERO_COPY;
    CheckStatus("zero copy", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED);
    args.flags = UINT64_C(1) << 20;
    CheckStatus("unknown flag", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.flags = 0;
    plan.r = 4;
    CheckStatus("world mismatch", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    plan = ValidPlan();
    plan.dupGroups = nullptr;
    CheckStatus("missing duplicate plan", TileXRMoonEp::TileXRMoonEpRunDispatchV1(
        &args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    CheckStatus("null stream", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, nullptr),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
}

void TestTopologyAndFailurePropagation()
{
    Reset();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 hiddenSh = Tensor(reinterpret_cast<void *>(uintptr_t {0x4000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 2, 16, 32);
    TileXRMoonEpTensorV1 hiddenNvsh = Tensor(reinterpret_cast<void *>(uintptr_t {0x5000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 8, 16, 128);
    TileXRMoonEpDispatchArgsV1 args = Args(&plan, &hiddenSh, nullptr, &hiddenNvsh, nullptr, 0);
    aclrtStream stream = reinterpret_cast<aclrtStream>(uintptr_t {0x8000});

    commArgs.localRankSize = 1;
    CheckStatus("cross-node", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED);
    Reset();
    commArgs.extraFlag = 0;
    CheckStatus("non-A5", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED);
    Reset();
    commArgs.peerMems[1] = nullptr;
    CheckStatus("missing peer", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INTERNAL);
    Reset();
    hostArgsReturn = -41;
    CheckStatus("host args error", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream), -41);
    Reset();
    devArgs = nullptr;
    CheckStatus("missing device args", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INTERNAL);
    Reset();
    devArgsReturn = -42;
    CheckStatus("device args error", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream), -42);
    Reset();
    magicReturn = -43;
    CheckStatus("magic error", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream), -43);
    Reset();
    nextMagic = 0;
    CheckStatus("invalid magic", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INTERNAL);
    Reset();
    launchReturn = -44;
    CheckStatus("launch error", TileXRMoonEp::TileXRMoonEpRunDispatchV1(&args, stream), -44);
}

} // namespace

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&args)
{
    ++hostArgsCalls;
    args = hostArgsReturn == TileXR::TILEXR_SUCCESS ? &commArgs : nullptr;
    return hostArgsReturn;
}

extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &args)
{
    ++devArgsCalls;
    args = devArgsReturn == TileXR::TILEXR_SUCCESS ? devArgs : nullptr;
    return devArgsReturn;
}

extern "C" int TileXRCommNextMagic(TileXRCommPtr, int64_t *magic)
{
    ++magicCalls;
    if (magicReturn == TileXR::TILEXR_SUCCESS && magic != nullptr) {
        *magic = nextMagic;
    }
    return magicReturn;
}

namespace TileXRMoonEp {
int TileXRMoonEpLaunchDispatchKernel(
    const DispatchParams &params, const DispatchLaunchContext &context)
{
    ++launchCalls;
    launchedParams = params;
    launchedContext = context;
    return launchReturn;
}
} // namespace TileXRMoonEp

int main()
{
    TestPairedFreshLaunch();
    TestReuseAndValidation();
    TestTopologyAndFailurePropagation();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
