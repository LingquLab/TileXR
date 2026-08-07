#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "planner_launch.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

extern "C" {
extern const unsigned char TileXRMoonEpPlannerKernelBinaryData[] = {0x7f, 'E', 'L', 'F'};
extern const std::size_t TileXRMoonEpPlannerKernelBinarySize =
    sizeof(TileXRMoonEpPlannerKernelBinaryData);
}

namespace {

struct PlannerKernelArgs {
    GM_ADDR commArgs;
    GM_ADDR topkExpertIds;
    GM_ADDR tokensPerExpert;
    GM_ADDR workspace;
    GM_ADDR dst;
    GM_ADDR cuSeqlens;
    GM_ADDR expertsToCopy;
    GM_ADDR remoteStats;
    GM_ADDR plannerStatus;
    int64_t s;
    int64_t k;
    int64_t expertCount;
    int64_t expertsPerRank;
    int64_t routeCount;
    int64_t dispatchedCapacity;
    uint64_t waitIterations;
    uint64_t tpePrefixOffset;
    uint64_t blockHistogramOffset;
    uint64_t allocPrefixOffset;
    uint64_t expertOffsetsOffset;
    uint64_t zOffset;
    uint64_t groupTotalsOffset;
    int64_t magic;
};

int g_failures = 0;
int g_magicReturn = TileXR::TILEXR_SUCCESS;
int g_magicCalls = 0;
int64_t g_magic = 37;
rtError_t g_binaryRegisterReturn = RT_ERROR_NONE;
rtError_t g_binaryUnregisterReturn = RT_ERROR_NONE;
rtError_t g_functionRegisterReturn = RT_ERROR_NONE;
rtError_t g_launchReturn = RT_ERROR_NONE;
int g_binaryRegisterCalls = 0;
int g_binaryUnregisterCalls = 0;
int g_functionRegisterCalls = 0;
int g_launchCalls = 0;
bool g_binaryWasNull = false;
rtDevBinary_t g_binary {};
void *g_binaryHandle = reinterpret_cast<void *>(0xc000);
void *g_functionBinaryHandle = nullptr;
void *g_unregisteredBinaryHandle = nullptr;
const void *g_functionStub = nullptr;
std::string g_functionStubName;
std::string g_functionKernelInfo;
uint32_t g_functionMode = UINT32_MAX;
const void *g_launchKernel = nullptr;
uint32_t g_blockDim = 0;
rtSmDesc_t *g_smDesc = nullptr;
rtStream_t g_stream = nullptr;
uint32_t g_launchFlags = UINT32_MAX;
int g_schemMode = -1;
size_t g_argsSize = 0;
PlannerKernelArgs g_args {};

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

void TestLaunch()
{
    TileXRMoonEp::PlannerParams params {};
    params.comm = reinterpret_cast<TileXRCommPtr>(0x1000);
    params.topkExpertIds = reinterpret_cast<const int32_t *>(0x2000);
    params.tokensPerExpert = reinterpret_cast<const int32_t *>(0x3000);
    params.workspace = reinterpret_cast<void *>(0x4000);
    params.dst = reinterpret_cast<int32_t *>(0x5000);
    params.cuSeqlens = reinterpret_cast<int32_t *>(0x6000);
    params.expertsToCopy = reinterpret_cast<int32_t *>(0x7000);
    params.remoteStats = reinterpret_cast<int32_t *>(0x8000);
    params.plannerStatus = reinterpret_cast<int32_t *>(0x9000);
    params.waitIterations = 123;
    params.stream = reinterpret_cast<aclrtStream>(0xa000);

    TileXRMoonEp::PlannerLaunchContext context {};
    context.devArgs = reinterpret_cast<GM_ADDR>(0xb000);
    context.layout.s = 8;
    context.layout.k = 2;
    context.layout.expertCount = 4;
    context.layout.expertsPerRank = 2;
    context.layout.routeCount = 16;
    context.layout.dispatchedCapacity = 15;
    context.layout.blockDim = 64;
    context.layout.tpePrefixOffset = 1;
    context.layout.blockHistogramOffset = 2;
    context.layout.allocPrefixOffset = 3;
    context.layout.expertOffsetsOffset = 4;
    context.layout.zOffset = 5;
    context.layout.groupTotalsOffset = 6;

    g_binaryRegisterReturn = -10;
    Check(TileXRMoonEp::TileXRMoonEpLaunchKernel(params, context) ==
        TileXR::TILEXR_ERROR_MKIRT, "Planner binary registration error was not translated");
    Check(g_binaryRegisterCalls == 1 && g_functionRegisterCalls == 0 &&
        g_binaryUnregisterCalls == 0 && g_launchCalls == 0 && g_magicCalls == 0,
        "Planner binary registration failure did not stop launch");

    g_binaryRegisterReturn = RT_ERROR_NONE;
    g_functionRegisterReturn = -11;
    Check(TileXRMoonEp::TileXRMoonEpLaunchKernel(params, context) ==
        TileXR::TILEXR_ERROR_MKIRT, "Planner function registration error was not translated");
    Check(g_binaryRegisterCalls == 2 && g_functionRegisterCalls == 1 &&
        g_binaryUnregisterCalls == 1 && g_launchCalls == 0 && g_magicCalls == 0,
        "Planner function registration failure did not stop launch");
    Check(g_unregisteredBinaryHandle == g_binaryHandle,
        "Planner function registration failure did not release the binary handle");

    g_functionRegisterReturn = RT_ERROR_NONE;
    Check(TileXRMoonEp::TileXRMoonEpLaunchKernel(params, context) ==
        TileXR::TILEXR_SUCCESS, "Planner runtime launch failed");
    Check(g_binaryRegisterCalls == 3 && g_functionRegisterCalls == 2 &&
        g_binaryUnregisterCalls == 1 && g_launchCalls == 1 && g_magicCalls == 1,
        "Planner successful registration/launch counts mismatch");
    Check(!g_binaryWasNull && g_binary.data == TileXRMoonEpPlannerKernelBinaryData &&
        g_binary.length == TileXRMoonEpPlannerKernelBinarySize &&
        g_binary.magic == RT_DEV_BINARY_MAGIC_ELF_AIVEC && g_binary.version == 0,
        "Planner binary descriptor ABI mismatch");
    Check(g_functionBinaryHandle == g_binaryHandle && g_functionStub != nullptr &&
        g_functionStubName == "tilexr_moonep_planner_kernel" &&
        g_functionKernelInfo == "tilexr_moonep_planner_kernel" && g_functionMode == 0,
        "Planner function registration ABI mismatch");
    Check(g_launchKernel == g_functionStub &&
        std::string(static_cast<const char *>(g_launchKernel)) ==
            "tilexr_moonep_planner_kernel",
        "Planner launch did not reuse the registered stub");
    Check(g_blockDim == 64 && g_stream == params.stream && g_smDesc == nullptr &&
        g_launchFlags == 0 && g_schemMode == 1,
        "Planner runtime launch metadata mismatch");
    Check(g_argsSize == sizeof(PlannerKernelArgs), "Planner args size ABI mismatch");
    Check(g_args.commArgs == context.devArgs &&
        g_args.topkExpertIds == reinterpret_cast<GM_ADDR>(
            const_cast<int32_t *>(params.topkExpertIds)) &&
        g_args.tokensPerExpert == reinterpret_cast<GM_ADDR>(
            const_cast<int32_t *>(params.tokensPerExpert)) &&
        g_args.workspace == static_cast<GM_ADDR>(params.workspace) &&
        g_args.dst == reinterpret_cast<GM_ADDR>(params.dst) &&
        g_args.cuSeqlens == reinterpret_cast<GM_ADDR>(params.cuSeqlens) &&
        g_args.expertsToCopy == reinterpret_cast<GM_ADDR>(params.expertsToCopy) &&
        g_args.remoteStats == reinterpret_cast<GM_ADDR>(params.remoteStats) &&
        g_args.plannerStatus == reinterpret_cast<GM_ADDR>(params.plannerStatus),
        "Planner pointer args ABI mismatch");
    Check(g_args.s == 8 && g_args.k == 2 && g_args.expertCount == 4 &&
        g_args.expertsPerRank == 2 && g_args.routeCount == 16 &&
        g_args.dispatchedCapacity == 15 && g_args.waitIterations == 123 &&
        g_args.tpePrefixOffset == 1 && g_args.blockHistogramOffset == 2 &&
        g_args.allocPrefixOffset == 3 && g_args.expertOffsetsOffset == 4 &&
        g_args.zOffset == 5 && g_args.groupTotalsOffset == 6 &&
        g_args.magic == g_magic,
        "Planner scalar args ABI mismatch");

    g_launchReturn = -12;
    Check(TileXRMoonEp::TileXRMoonEpLaunchKernel(params, context) ==
        TileXR::TILEXR_ERROR_MKIRT, "Planner launch error was not translated");
    Check(g_binaryRegisterCalls == 3 && g_functionRegisterCalls == 2 &&
        g_binaryUnregisterCalls == 1 && g_launchCalls == 2 && g_magicCalls == 2,
        "Planner successful registration was not cached");

    g_launchReturn = RT_ERROR_NONE;
    g_magicReturn = -91;
    Check(TileXRMoonEp::TileXRMoonEpLaunchKernel(params, context) == -91,
        "Planner magic error was not propagated");
    Check(g_launchCalls == 2 && g_magicCalls == 3,
        "Planner magic failure did not stop launch");
}

} // namespace

extern "C" int TileXRCommNextMagic(TileXRCommPtr, int64_t *magic)
{
    ++g_magicCalls;
    if (magic != nullptr) {
        *magic = g_magic;
    }
    return g_magicReturn;
}

extern "C" rtError_t rtDevBinaryRegister(const rtDevBinary_t *binary, void **handle)
{
    ++g_binaryRegisterCalls;
    g_binaryWasNull = binary == nullptr;
    if (binary != nullptr) {
        g_binary = *binary;
    }
    if (g_binaryRegisterReturn == RT_ERROR_NONE && handle != nullptr) {
        *handle = g_binaryHandle;
    }
    return g_binaryRegisterReturn;
}

extern "C" rtError_t rtDevBinaryUnRegister(void *handle)
{
    ++g_binaryUnregisterCalls;
    g_unregisteredBinaryHandle = handle;
    return g_binaryUnregisterReturn;
}

extern "C" rtError_t rtFunctionRegister(void *binaryHandle, const void *stubFunc,
    const char_t *stubName, const void *kernelInfoExt, uint32_t functionMode)
{
    ++g_functionRegisterCalls;
    g_functionBinaryHandle = binaryHandle;
    g_functionStub = stubFunc;
    g_functionStubName = stubName == nullptr ? "" : stubName;
    g_functionKernelInfo = kernelInfoExt == nullptr ? "" :
        static_cast<const char *>(kernelInfoExt);
    g_functionMode = functionMode;
    return g_functionRegisterReturn;
}

extern "C" rtError_t rtKernelLaunchWithFlagV2(const void *kernel, uint32_t blockDim,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags,
    const rtTaskCfgInfo_t *cfgInfo)
{
    ++g_launchCalls;
    g_launchKernel = kernel;
    g_blockDim = blockDim;
    g_smDesc = smDesc;
    g_stream = stream;
    g_launchFlags = flags;
    g_schemMode = cfgInfo == nullptr ? -1 : cfgInfo->schemMode;
    g_argsSize = argsInfo == nullptr ? 0 : argsInfo->argsSize;
    if (argsInfo != nullptr && argsInfo->args != nullptr &&
        argsInfo->argsSize == sizeof(PlannerKernelArgs)) {
        g_args = *static_cast<const PlannerKernelArgs *>(argsInfo->args);
    }
    return g_launchReturn;
}

int main()
{
    TestLaunch();
    return g_failures == 0 ? 0 : 1;
}
