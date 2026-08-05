#include <cstdint>
#include <cstring>
#include <iostream>

#include "reduce_grad_common.h"
#include "reduce_grad_host.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

#ifndef UINTPTR_C
#define UINTPTR_C(value) static_cast<uintptr_t>(value)
#endif

namespace {

int g_failures = 0;
int g_magicReturn = TileXR::TILEXR_SUCCESS;
int64_t g_magic = 77;
rtError_t g_launchReturn = RT_ERROR_NONE;
int g_launchCalls = 0;
const void *g_kernel = nullptr;
uint32_t g_blockDim = 0;
rtStream_t g_stream = nullptr;
uint32_t g_flags = 99;
int g_schemeMode = 0;
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
    layout.peerRecordBaseOffset = 1;
    layout.peerHalfBytes = 2;
    layout.peerSlotStrideBytes = 3;
    layout.peerChunkPayloadBytes = 4;
    layout.udmaStateOffset = 5;
    layout.udmaOutboundOffset = 6;
    layout.udmaInboundOffset = 7;
    layout.udmaChunkBytes = 8;
    layout.workspaceBytes = 9;

    Check(TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context) ==
        TileXR::TILEXR_SUCCESS, "ReduceGrad launch failed");
    Check(g_launchCalls == 1, "Runtime V2 must launch exactly once");
    Check(g_kernel != nullptr && g_blockDim == 64 && g_stream == params.stream &&
        g_flags == 0 && g_schemeMode == 1, "Runtime V2 launch metadata mismatch");
    Check(g_args.commArgs == context.devArgs && g_args.expertsToCopy == plan.expertsToCopy &&
        g_args.gate == gate.data && g_args.up == up.data && g_args.down == down.data &&
        g_args.workspace == params.workspace && g_args.status == status.data,
        "ReduceGrad pointer argument order mismatch");
    Check(g_args.rank == 1 && g_args.rankSize == 4 && g_args.expertCount == 8 &&
        g_args.expertsPerRank == 2 && g_args.controlBlockCount == 3,
        "ReduceGrad scalar argument order mismatch");
    Check(g_args.gateRowElements == 10 && g_args.upRowElements == 20 &&
        g_args.downRowElements == 30 && g_args.gateRowBytes == 40 &&
        g_args.upRowBytes == 80 && g_args.downRowBytes == 120,
        "ReduceGrad row argument order mismatch");
    Check(g_args.gateTransport == TileXRMoonEp::kReduceGradTransportPeer &&
        g_args.upTransport == TileXRMoonEp::kReduceGradTransportUdma &&
        g_args.downTransport == TileXRMoonEp::kReduceGradTransportPeer,
        "ReduceGrad transport argument order mismatch");
    Check(g_args.waitIterations == 123 && g_args.magic == g_magic,
        "ReduceGrad wait/magic argument mismatch");

    g_launchReturn = -1;
    Check(TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context) ==
        TileXR::TILEXR_ERROR_MKIRT, "Runtime launch error was not translated");
    g_launchReturn = RT_ERROR_NONE;
    g_magicReturn = -91;
    Check(TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context) == -91,
        "magic error was not propagated");
}

} // namespace

extern "C" int TileXRCommNextMagic(TileXRCommPtr, int64_t *magic)
{
    if (magic != nullptr) {
        *magic = g_magic;
    }
    return g_magicReturn;
}

extern "C" void tilexr_moonep_reduce_grad_kernel()
{
}

extern "C" rtError_t rtKernelLaunchWithFlagV2(const void *kernel, uint32_t blockDim,
    rtArgsEx_t *argsInfo, void *, rtStream_t stream, uint32_t flags,
    rtTaskCfgInfo_t *cfgInfo)
{
    ++g_launchCalls;
    g_kernel = kernel;
    g_blockDim = blockDim;
    g_stream = stream;
    g_flags = flags;
    g_schemeMode = cfgInfo == nullptr ? -1 : cfgInfo->schemMode;
    if (argsInfo != nullptr && argsInfo->args != nullptr &&
        argsInfo->argsSize == sizeof(g_args)) {
        std::memcpy(&g_args, argsInfo->args, sizeof(g_args));
    }
    return g_launchReturn;
}

int main()
{
    TestLaunch();
    return g_failures == 0 ? 0 : 1;
}
