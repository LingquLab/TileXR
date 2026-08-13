#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "acl/acl_rt.h"
#include "dispatch_host.h"
#include "dispatch_launch.h"
#include "tilexr_types.h"

namespace {

int failures = 0;
int hostArgsCalls = 0;
int devArgsCalls = 0;
int launchCalls = 0;
int memsetCalls = 0;
int synchronizeCalls = 0;
int launchReturn = TileXR::TILEXR_SUCCESS;
aclError memsetReturn = ACL_SUCCESS;
aclError synchronizeReturn = ACL_SUCCESS;
TileXR::CommArgs commArgs {};
GM_ADDR devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
TileXRMoonEp::DispatchUrmaLaunchParams launchedParams {};

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

void Reset()
{
    hostArgsCalls = devArgsCalls = launchCalls = 0;
    memsetCalls = synchronizeCalls = 0;
    launchReturn = TileXR::TILEXR_SUCCESS;
    memsetReturn = synchronizeReturn = ACL_SUCCESS;
    commArgs = TileXR::CommArgs {};
    commArgs.rank = 0;
    commArgs.localRank = 0;
    commArgs.rankSize = 1;
    commArgs.localRankSize = 1;
    devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
    launchedParams = TileXRMoonEp::DispatchUrmaLaunchParams {};
}

TileXRMoonEpPlanV1 ValidPlan()
{
    TileXRMoonEpPlanV1 plan {};
    plan.structSize = sizeof(plan);
    plan.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    plan.n = 4;
    plan.r = 1;
    plan.e = 8;
    plan.b = 2;
    plan.nvS = 4;
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

TileXRMoonEpDispatchArgsV2 Args(const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *hiddenInput,
    const TileXRMoonEpTensorV1 *weightInput,
    TileXRMoonEpTensorV1 *hiddenOutput,
    TileXRMoonEpTensorV1 *weightOutput)
{
    TileXRMoonEpDispatchArgsV2 args {};
    args.structSize = sizeof(args);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V2;
    args.comm = reinterpret_cast<TileXRCommPtr>(uintptr_t {0x1000});
    args.plan = plan;
    args.hiddenSh = hiddenInput;
    args.routeWeightsSk = weightInput;
    args.hiddenNvsh = hiddenOutput;
    args.routeWeightsNvs = weightOutput;
    args.flags = TILEXR_MOONEP_FLAG_RESET_STATUS;
    args.registeredWorkspace = reinterpret_cast<void *>(uintptr_t {0x200000});
    args.registeredWorkspaceBytes = UINT64_C(2) * 1024U * 1024U;
    return args;
}

void TestPairedSingleLaunch()
{
    Reset();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 hiddenInput = Tensor(
        reinterpret_cast<void *>(uintptr_t {0x4000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 2, 17, 34);
    TileXRMoonEpTensorV1 hiddenOutput = Tensor(
        reinterpret_cast<void *>(uintptr_t {0x5000}),
        TILEXR_MOONEP_DTYPE_BFLOAT16, 2, 4, 17, 68);
    TileXRMoonEpTensorV1 weightInput = Tensor(
        reinterpret_cast<void *>(uintptr_t {0x6000}),
        TILEXR_MOONEP_DTYPE_FLOAT32, 2, 2, 2, 4);
    TileXRMoonEpTensorV1 weightOutput = Tensor(
        reinterpret_cast<void *>(uintptr_t {0x7000}),
        TILEXR_MOONEP_DTYPE_FLOAT32, 1, 4, 0, 4);
    TileXRMoonEpDispatchArgsV2 args = Args(&plan, &hiddenInput, &weightInput,
        &hiddenOutput, &weightOutput);
    aclrtStream stream = reinterpret_cast<aclrtStream>(uintptr_t {0x8000});

    CheckStatus("paired", TileXRMoonEp::TileXRMoonEpRunDispatchUrmaV2(
        &args, stream), TILEXR_MOONEP_SUCCESS);
    Check(hostArgsCalls == 1 && devArgsCalls == 1 && launchCalls == 1,
        "paired Dispatch must make exactly one internal launch");
    Check(memsetCalls == 1 && synchronizeCalls == 0,
        "paired success must enqueue one reset without synchronizing");
    Check(launchedParams.hiddenInput == hiddenInput.data &&
        launchedParams.hiddenOutput == hiddenOutput.data &&
        launchedParams.weightInput == weightInput.data &&
        launchedParams.weightOutput == weightOutput.data,
        "paired launch did not carry all four payload pointers");
    Check(launchedParams.layout.weight.sourceOffset >=
        launchedParams.layout.hidden.scratchOffset +
            launchedParams.layout.hidden.scratchBytes,
        "paired launch layout overlaps Hidden and Weight active regions");
}

void TestHiddenOnlyAndFailureBoundaries()
{
    Reset();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 hiddenInput = Tensor(
        reinterpret_cast<void *>(uintptr_t {0x4000}),
        TILEXR_MOONEP_DTYPE_FLOAT16, 2, 2, 16, 32);
    TileXRMoonEpTensorV1 hiddenOutput = Tensor(
        reinterpret_cast<void *>(uintptr_t {0x5000}),
        TILEXR_MOONEP_DTYPE_FLOAT16, 2, 4, 16, 64);
    TileXRMoonEpDispatchArgsV2 args = Args(
        &plan, &hiddenInput, nullptr, &hiddenOutput, nullptr);
    aclrtStream stream = reinterpret_cast<aclrtStream>(uintptr_t {0x8000});

    CheckStatus("hidden", TileXRMoonEp::TileXRMoonEpRunDispatchUrmaV2(
        &args, stream), TILEXR_MOONEP_SUCCESS);
    Check(launchCalls == 1 && launchedParams.weightInput == nullptr &&
        launchedParams.weightOutput == nullptr,
        "hidden-only Dispatch must launch once without Weight pointers");

    Reset();
    launchReturn = TileXR::TILEXR_ERROR_MKIRT;
    CheckStatus("launch failure", TileXRMoonEp::TileXRMoonEpRunDispatchUrmaV2(
        &args, stream), TILEXR_MOONEP_ERROR_INTERNAL);
    Check(launchCalls == 1 && synchronizeCalls == 1,
        "same-stream reset must be synchronized after launch failure");

    Reset();
    memsetReturn = 1;
    CheckStatus("reset failure", TileXRMoonEp::TileXRMoonEpRunDispatchUrmaV2(
        &args, stream), TILEXR_MOONEP_ERROR_INTERNAL);
    Check(launchCalls == 0, "reset failure must prevent launch");
}

} // namespace

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&args)
{
    ++hostArgsCalls;
    args = &commArgs;
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &args)
{
    ++devArgsCalls;
    args = devArgs;
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRUDMAGetQpCount(TileXRCommPtr, uint32_t *qpCount)
{
    if (qpCount != nullptr) {
        *qpCount = 2U;
    }
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRGetUDMARegistryHost(
    TileXRCommPtr, const TileXR::TileXRUDMARegistry **registry)
{
    if (registry != nullptr) {
        *registry = nullptr;
    }
    return TileXR::TILEXR_SUCCESS;
}

extern "C" aclError aclrtMemsetAsync(
    void *, size_t, int32_t, size_t, aclrtStream)
{
    ++memsetCalls;
    return memsetReturn;
}

extern "C" aclError aclrtSynchronizeStream(aclrtStream)
{
    ++synchronizeCalls;
    return synchronizeReturn;
}

extern "C" aclError aclrtGetDevice(int32_t *deviceId)
{
    if (deviceId != nullptr) {
        *deviceId = 0;
    }
    return ACL_SUCCESS;
}

extern "C" aclError aclrtGetDeviceInfo(
    uint32_t, aclrtDevAttr, int64_t *value)
{
    if (value != nullptr) {
        *value = TileXRMoonEp::kDispatchAivCoreCount;
    }
    return ACL_SUCCESS;
}

namespace TileXRMoonEp {
int TileXRMoonEpLaunchDispatchUrmaKernel(const DispatchUrmaLaunchParams &params)
{
    ++launchCalls;
    launchedParams = params;
    return launchReturn;
}
} // namespace TileXRMoonEp

int main()
{
    TestPairedSingleLaunch();
    TestHiddenOnlyAndFailureBoundaries();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
