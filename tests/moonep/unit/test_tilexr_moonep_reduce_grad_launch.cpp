#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "reduce_grad_common.h"
#include "reduce_grad_host.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

#ifndef UINTPTR_C
#define UINTPTR_C(value) static_cast<uintptr_t>(value)
#endif

extern "C" {
extern const unsigned char TileXRMoonEpReduceGradKernelBinaryData[] = {0x7f, 'E', 'L', 'F'};
extern const std::size_t TileXRMoonEpReduceGradKernelBinarySize =
    sizeof(TileXRMoonEpReduceGradKernelBinaryData);
}

namespace {

int g_failures = 0;
int g_magicReturn = TileXR::TILEXR_SUCCESS;
int g_magicCalls = 0;
int64_t g_magic = 77;
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
void *g_binaryHandle = reinterpret_cast<void *>(UINTPTR_C(0xa000));
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
TileXRMoonEp::ReduceGradKernelArgs g_args {};

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

TileXRMoonEpTensorV1 Tensor(void *data)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.data = data;
    return tensor;
}

void TestLaunch()
{
    TileXRMoonEpPlanV1 plan {};
    plan.expertsToCopy = reinterpret_cast<void *>(UINTPTR_C(0x2000));
    TileXRMoonEpTensorV1 gate = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x3000)));
    TileXRMoonEpTensorV1 up = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x4000)));
    TileXRMoonEpTensorV1 down = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x5000)));
    TileXRMoonEpTensorV1 status = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x6000)));

    TileXRMoonEp::ReduceGradParams params {};
    params.comm = reinterpret_cast<TileXRCommPtr>(UINTPTR_C(0x1000));
    params.plan = &plan;
    params.gradients[0] = &gate;
    params.gradients[1] = &up;
    params.gradients[2] = &down;
    params.workspace = reinterpret_cast<void *>(UINTPTR_C(0x7000));
    params.status = &status;
    params.waitIterations = 123;
    params.stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x8000));

    TileXRMoonEp::ReduceGradLaunchContext context {};
    context.devArgs = reinterpret_cast<GM_ADDR>(UINTPTR_C(0x9000));
    auto &layout = context.layout;
    layout.rank = 1;
    layout.rankSize = 4;
    layout.expertCount = 8;
    layout.expertsPerRank = 2;
    layout.prefetchSlots = 1;
    layout.controlBlockCount = 3;
    layout.blockDim = 64;
    layout.rowElements[0] = 10;
    layout.rowElements[1] = 20;
    layout.rowElements[2] = 30;
    layout.rowBytes[0] = 40;
    layout.rowBytes[1] = 80;
    layout.rowBytes[2] = 120;
    layout.transports[0] = TileXRMoonEp::kReduceGradTransportPeer;
    layout.transports[1] = TileXRMoonEp::kReduceGradTransportUdma;
    layout.transports[2] = TileXRMoonEp::kReduceGradTransportPeer;
    layout.udmaQpCount = 3;
    layout.peerRecordBaseOffset = 1;
    layout.peerHalfBytes = 2;
    layout.peerSlotStrideBytes = 3;
    layout.peerChunkPayloadBytes = 4;
    layout.udmaStateOffset = 5;
    layout.udmaOutboundOffset = 6;
    layout.udmaInboundOffset = 7;
    layout.udmaChunkBytes = 8;
    layout.workspaceBytes = 9;

    g_binaryRegisterReturn = -10;
    Check(TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context) ==
        TileXR::TILEXR_ERROR_MKIRT,
        "ReduceGrad binary registration error was not translated");
    Check(g_binaryRegisterCalls == 1 && g_functionRegisterCalls == 0 &&
        g_binaryUnregisterCalls == 0 && g_launchCalls == 0 && g_magicCalls == 0,
        "ReduceGrad binary registration failure did not stop launch");

    g_binaryRegisterReturn = RT_ERROR_NONE;
    g_functionRegisterReturn = -11;
    Check(TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context) ==
        TileXR::TILEXR_ERROR_MKIRT,
        "ReduceGrad function registration error was not translated");
    Check(g_binaryRegisterCalls == 2 && g_functionRegisterCalls == 1 &&
        g_binaryUnregisterCalls == 1 && g_launchCalls == 0 && g_magicCalls == 0,
        "ReduceGrad function registration failure did not stop launch");
    Check(g_unregisteredBinaryHandle == g_binaryHandle,
        "ReduceGrad function registration failure did not release the binary handle");

    g_functionRegisterReturn = RT_ERROR_NONE;
    Check(TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context) ==
        TileXR::TILEXR_SUCCESS, "ReduceGrad runtime launch failed");
    Check(g_binaryRegisterCalls == 3 && g_functionRegisterCalls == 2 &&
        g_binaryUnregisterCalls == 1 && g_launchCalls == 1 && g_magicCalls == 1,
        "ReduceGrad successful registration/launch counts mismatch");
    Check(!g_binaryWasNull &&
        g_binary.data == TileXRMoonEpReduceGradKernelBinaryData &&
        g_binary.length == TileXRMoonEpReduceGradKernelBinarySize &&
        g_binary.magic == RT_DEV_BINARY_MAGIC_ELF_AIVEC && g_binary.version == 0,
        "ReduceGrad binary descriptor ABI mismatch");
    Check(g_functionBinaryHandle == g_binaryHandle && g_functionStub != nullptr &&
        g_functionStubName == "tilexr_moonep_reduce_grad_kernel" &&
        g_functionKernelInfo == "tilexr_moonep_reduce_grad_kernel" &&
        g_functionMode == 0,
        "ReduceGrad function registration ABI mismatch");
    Check(g_launchKernel == g_functionStub &&
        std::string(static_cast<const char *>(g_launchKernel)) ==
            "tilexr_moonep_reduce_grad_kernel",
        "ReduceGrad launch did not reuse the registered stub");
    Check(g_blockDim == 64 && g_stream == params.stream && g_smDesc == nullptr &&
        g_launchFlags == 0 && g_schemMode == 1,
        "ReduceGrad runtime launch metadata mismatch");
    Check(g_argsSize == sizeof(TileXRMoonEp::ReduceGradKernelArgs),
        "ReduceGrad args size ABI mismatch");
    Check(g_args.commArgs == context.devArgs &&
        g_args.expertsToCopy == plan.expertsToCopy && g_args.gate == gate.data &&
        g_args.up == up.data && g_args.down == down.data &&
        g_args.workspace == params.workspace && g_args.status == status.data,
        "ReduceGrad pointer args ABI mismatch");
    Check(g_args.rank == 1 && g_args.rankSize == 4 && g_args.expertCount == 8 &&
        g_args.expertsPerRank == 2 && g_args.prefetchSlots == 1 &&
        g_args.controlBlockCount == 3,
        "ReduceGrad rank args ABI mismatch");
    Check(g_args.gateRowElements == 10 && g_args.upRowElements == 20 &&
        g_args.downRowElements == 30 && g_args.gateRowBytes == 40 &&
        g_args.upRowBytes == 80 && g_args.downRowBytes == 120,
        "ReduceGrad row args ABI mismatch");
    Check(g_args.gateTransport == TileXRMoonEp::kReduceGradTransportPeer &&
        g_args.upTransport == TileXRMoonEp::kReduceGradTransportUdma &&
        g_args.downTransport == TileXRMoonEp::kReduceGradTransportPeer &&
        g_args.udmaQpCount == 3,
        "ReduceGrad transport args ABI mismatch");
    Check(g_args.peerRecordBaseOffset == 1 && g_args.peerHalfBytes == 2 &&
        g_args.peerSlotStrideBytes == 3 && g_args.peerChunkPayloadBytes == 4 &&
        g_args.udmaStateOffset == 5 && g_args.udmaOutboundOffset == 6 &&
        g_args.udmaInboundOffset == 7 && g_args.udmaChunkBytes == 8 &&
        g_args.workspaceBytes == 9 && g_args.waitIterations == 123 &&
        g_args.magic == g_magic,
        "ReduceGrad offset/wait args ABI mismatch");

    g_launchReturn = -12;
    Check(TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context) ==
        TileXR::TILEXR_ERROR_MKIRT, "ReduceGrad launch error was not translated");
    Check(g_binaryRegisterCalls == 3 && g_functionRegisterCalls == 2 &&
        g_binaryUnregisterCalls == 1 && g_launchCalls == 2 && g_magicCalls == 2,
        "ReduceGrad successful registration was not cached");

    g_launchReturn = RT_ERROR_NONE;
    g_magicReturn = -91;
    Check(TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context) == -91,
        "ReduceGrad magic error was not propagated");
    Check(g_launchCalls == 2 && g_magicCalls == 3,
        "ReduceGrad magic failure did not stop launch");
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
        argsInfo->argsSize == sizeof(TileXRMoonEp::ReduceGradKernelArgs)) {
        g_args = *static_cast<const TileXRMoonEp::ReduceGradKernelArgs *>(argsInfo->args);
    }
    return g_launchReturn;
}

int main()
{
    TestLaunch();
    return g_failures == 0 ? 0 : 1;
}
