#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "combine_host.h"
#include "combine_launch.h"
#include "moonep_peer_window.h"
#include "tilexr_types.h"

namespace TileXRMoonEp {
int TileXRMoonEpRunCombineV1(
    const TileXRMoonEpCombineArgsV1 *args, aclrtStream stream);
}

namespace {

int failures = 0;
int hostReturn = TileXR::TILEXR_SUCCESS;
int devReturn = TileXR::TILEXR_SUCCESS;
int magicReturn = TileXR::TILEXR_SUCCESS;
int launchReturn = TileXR::TILEXR_SUCCESS;
int hostCalls = 0;
int devCalls = 0;
int magicCalls = 0;
int launchCalls = 0;
int64_t nextMagic = 91;
TileXR::CommArgs commArgs {};
GM_ADDR devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
TileXRMoonEp::CombineParams launchedParams {};
TileXRMoonEp::CombineLaunchContext launchedContext {};

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
    hostReturn = devReturn = magicReturn = launchReturn = TileXR::TILEXR_SUCCESS;
    hostCalls = devCalls = magicCalls = launchCalls = 0;
    nextMagic = 91;
    commArgs = ValidCommArgs();
    devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
    launchedParams = TileXRMoonEp::CombineParams {};
    launchedContext = TileXRMoonEp::CombineLaunchContext {};
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

TileXRMoonEpCombineArgsV1 Args(const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *hiddenNvsh, const TileXRMoonEpTensorV1 *weightsNvs,
    TileXRMoonEpTensorV1 *hiddenSh, TileXRMoonEpTensorV1 *weightsSk, uint64_t flags)
{
    TileXRMoonEpCombineArgsV1 args {};
    args.structSize = sizeof(args);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    args.comm = reinterpret_cast<TileXRCommPtr>(uintptr_t {0x1000});
    args.plan = plan;
    args.dstLocal = reinterpret_cast<const int32_t *>(uintptr_t {0x3800});
    args.hiddenNvsh = hiddenNvsh;
    args.routeWeightsNvs = weightsNvs;
    args.hiddenSh = hiddenSh;
    args.routeWeightsSk = weightsSk;
    args.flags = flags;
    return args;
}

void TestPairedLaunch()
{
    Reset();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 hiddenNvsh = Tensor(reinterpret_cast<void *>(uintptr_t {0x4000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 8, 17, 136);
    TileXRMoonEpTensorV1 hiddenSh = Tensor(reinterpret_cast<void *>(uintptr_t {0x5000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 2, 17, 34);
    TileXRMoonEpTensorV1 weightsNvs = Tensor(reinterpret_cast<void *>(uintptr_t {0x6000}),
        TILEXR_MOONEP_DTYPE_FLOAT32, 1, 8, 0, 8);
    TileXRMoonEpTensorV1 weightsSk = Tensor(reinterpret_cast<void *>(uintptr_t {0x7000}),
        TILEXR_MOONEP_DTYPE_FLOAT32, 2, 2, 2, 4);
    TileXRMoonEpCombineArgsV1 args = Args(&plan, &hiddenNvsh, &weightsNvs,
        &hiddenSh, &weightsSk, TILEXR_MOONEP_FLAG_NONE);
    aclrtStream stream = reinterpret_cast<aclrtStream>(uintptr_t {0x8000});

    CheckStatus("paired combine", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_SUCCESS);
    Check(hostCalls == 1 && devCalls == 1 && magicCalls == 1 && launchCalls == 1,
        "paired combine call counts mismatch");
    Check(launchedParams.dstLocal == args.dstLocal &&
        launchedParams.dst == plan.dst && launchedParams.dupGroups == plan.dupGroups &&
        launchedParams.dupLoffs == plan.dupLoffs && launchedParams.dupCounts == plan.dupCounts &&
        launchedParams.hiddenNvsh == hiddenNvsh.data && launchedParams.hiddenSh == hiddenSh.data &&
        launchedParams.routeWeightsNvs == weightsNvs.data &&
        launchedParams.routeWeightsSk == weightsSk.data && launchedParams.status == plan.status &&
        launchedParams.flags == args.flags && launchedParams.stream == stream,
        "paired combine pointers mismatch");
    Check(launchedContext.hostArgs == &commArgs && launchedContext.devArgs == devArgs &&
        launchedContext.magic == nextMagic &&
        launchedContext.waitIterations == TileXRMoonEp::kMoonEpPeerWaitIterations,
        "paired combine context mismatch");
    Check(launchedContext.layout.n == 4 && launchedContext.layout.nvS == 8 &&
        launchedContext.layout.hiddenRowBytes == 34 &&
        launchedContext.layout.hiddenChunkStride == 64 &&
        launchedContext.layout.routeWeightsBytes == 32,
        "paired combine layout mismatch");
}

void TestValidationAndFailures()
{
    Reset();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 hiddenNvsh = Tensor(reinterpret_cast<void *>(uintptr_t {0x4000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 8, 16, 128);
    TileXRMoonEpTensorV1 hiddenSh = Tensor(reinterpret_cast<void *>(uintptr_t {0x5000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 2, 16, 32);
    TileXRMoonEpTensorV1 weightsNvs = Tensor(reinterpret_cast<void *>(uintptr_t {0x6000}),
        TILEXR_MOONEP_DTYPE_FLOAT32, 1, 8, 0, 8);
    TileXRMoonEpCombineArgsV1 args = Args(&plan, &hiddenNvsh, nullptr, &hiddenSh, nullptr, 0);
    aclrtStream stream = reinterpret_cast<aclrtStream>(uintptr_t {0x8000});

    plan.b = 1;
    CheckStatus("B below experts-per-rank", TileXRMoonEp::TileXRMoonEpRunCombineV1(
        &args, stream), TILEXR_MOONEP_SUCCESS);
    args.routeWeightsNvs = &weightsNvs;
    CheckStatus("unpaired weights", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.routeWeightsNvs = nullptr;
    args.dstLocal = nullptr;
    CheckStatus("missing reverse route", TileXRMoonEp::TileXRMoonEpRunCombineV1(
        &args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.dstLocal = reinterpret_cast<const int32_t *>(uintptr_t {0x3800});
    args.flags = TILEXR_MOONEP_FLAG_BUILD_DEDUP;
    CheckStatus("build dedup", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.flags = TILEXR_MOONEP_FLAG_ZERO_COPY;
    CheckStatus("zero copy", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED);
    args.flags = 0;
    plan.r = 4;
    CheckStatus("world mismatch", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    plan = ValidPlan();
    plan.dupCounts = nullptr;
    CheckStatus("missing duplicate plan", TileXRMoonEp::TileXRMoonEpRunCombineV1(
        &args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    CheckStatus("null stream", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, nullptr),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);

    plan = ValidPlan();
    Reset();
    commArgs.localRankSize = 1;
    CheckStatus("cross-node", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_SUCCESS);
    Reset();
    commArgs.extraFlag = 0;
    CheckStatus("non-A5", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED);
    Reset();
    commArgs.peerMems[1] = nullptr;
    CheckStatus("missing peer", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_ERROR_INTERNAL);
    Reset();
    hostReturn = -51;
    CheckStatus("host args error", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream), -51);
    Reset();
    devArgs = nullptr;
    CheckStatus("missing device args", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_ERROR_INTERNAL);
    Reset();
    devReturn = -52;
    CheckStatus("device args error", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream), -52);
    Reset();
    magicReturn = -53;
    CheckStatus("magic error", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream), -53);
    Reset();
    nextMagic = 0;
    CheckStatus("invalid magic", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream),
        TILEXR_MOONEP_ERROR_INTERNAL);
    Reset();
    launchReturn = -54;
    CheckStatus("launch error", TileXRMoonEp::TileXRMoonEpRunCombineV1(&args, stream), -54);
}

void TestSplitPhaseRejected()
{
    Reset();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 hiddenNvsh = Tensor(reinterpret_cast<void *>(uintptr_t {0x4000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 8, 16, 128);
    TileXRMoonEpTensorV1 hiddenSh = Tensor(reinterpret_cast<void *>(uintptr_t {0x5000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 2, 16, 32);
    TileXRMoonEpCombineArgsV1 args = Args(&plan, &hiddenNvsh, nullptr, &hiddenSh, nullptr,
        TILEXR_MOONEP_FLAG_COMBINE_PUBLISH_ONLY);
    aclrtStream stream = reinterpret_cast<aclrtStream>(uintptr_t {0x8000});

    CheckStatus("publish-only launch", TileXRMoonEp::TileXRMoonEpRunCombineV1(
        &args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    Check(launchCalls == 0, "publish-only unexpectedly launched a kernel");

    args.flags = TILEXR_MOONEP_FLAG_COMBINE_CONSUME_ONLY;
    CheckStatus("consume-only launch", TileXRMoonEp::TileXRMoonEpRunCombineV1(
        &args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    Check(launchCalls == 0, "consume-only unexpectedly launched a kernel");
}

} // namespace

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&args)
{
    ++hostCalls;
    args = hostReturn == TileXR::TILEXR_SUCCESS ? &commArgs : nullptr;
    return hostReturn;
}

extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &args)
{
    ++devCalls;
    args = devReturn == TileXR::TILEXR_SUCCESS ? devArgs : nullptr;
    return devReturn;
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
int TileXRMoonEpLaunchCombineKernel(
    const CombineParams &params, const CombineLaunchContext &context)
{
    ++launchCalls;
    launchedParams = params;
    launchedContext = context;
    return launchReturn;
}
} // namespace TileXRMoonEp

int main()
{
    TestPairedLaunch();
    TestValidationAndFailures();
    TestSplitPhaseRejected();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
