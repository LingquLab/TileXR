#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#include "acl/acl_rt.h"
#include "comm_args.h"
#include "reduce_grad_host.h"
#include "tilexr_api.h"
#include "tilexr_moonep.h"
#include "tilexr_types.h"
#include "tilexr_udma_reg.h"

#ifndef UINTPTR_C
#define UINTPTR_C(value) static_cast<uintptr_t>(value)
#endif

namespace {

int g_failures = 0;
int g_launchCalls = 0;
int g_memsetCalls = 0;
int g_commHostReturn = TileXR::TILEXR_SUCCESS;
int g_commDevReturn = TileXR::TILEXR_SUCCESS;
int g_registryReturn = TileXR::TILEXR_SUCCESS;
int g_launchReturn = TileXR::TILEXR_SUCCESS;
TileXR::CommArgs g_commArgs {};
TileXR::TileXRUDMARegistry g_registry {};
GM_ADDR g_commDev = reinterpret_cast<GM_ADDR>(UINTPTR_C(0x800000));
TileXRMoonEp::ReduceGradLayout g_launchedLayout {};

void Check(bool condition, const std::string &message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

void CheckStatus(const std::string &label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

TileXRMoonEpPlanV1 Plan()
{
    TileXRMoonEpPlanV1 plan {};
    plan.structSize = sizeof(plan);
    plan.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    plan.s = 2;
    plan.k = 2;
    plan.e = 4;
    plan.b = 2;
    plan.rank = 0;
    plan.world = 2;
    plan.dispatchedCapacity = 4;
    plan.dst = reinterpret_cast<void *>(UINTPTR_C(0x11000));
    plan.cu = reinterpret_cast<void *>(UINTPTR_C(0x12000));
    plan.expertsToCopy = reinterpret_cast<void *>(UINTPTR_C(0x13000));
    plan.remoteStats = reinterpret_cast<void *>(UINTPTR_C(0x14000));
    plan.status = reinterpret_cast<void *>(UINTPTR_C(0x15000));
    return plan;
}

TileXRMoonEpTensorV1 Gradient(void *data, uint64_t rowElements)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = data;
    tensor.elementCount = 6 * rowElements;
    tensor.dtype = TILEXR_MOONEP_DTYPE_FLOAT32;
    tensor.rank = 2;
    tensor.shape[0] = 6;
    tensor.shape[1] = static_cast<int64_t>(rowElements);
    return tensor;
}

TileXRMoonEpTensorV1 Status()
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = reinterpret_cast<void *>(UINTPTR_C(0x16000));
    tensor.elementCount = 1;
    tensor.dtype = TILEXR_MOONEP_DTYPE_INT32;
    tensor.rank = 1;
    tensor.shape[0] = 1;
    return tensor;
}

void Reset()
{
    g_launchCalls = 0;
    g_memsetCalls = 0;
    g_commHostReturn = TileXR::TILEXR_SUCCESS;
    g_commDevReturn = TileXR::TILEXR_SUCCESS;
    g_registryReturn = TileXR::TILEXR_SUCCESS;
    g_launchReturn = TileXR::TILEXR_SUCCESS;
    g_launchedLayout = TileXRMoonEp::ReduceGradLayout {};
    g_commArgs = TileXR::CommArgs {};
    g_commArgs.rank = 0;
    g_commArgs.localRank = 0;
    g_commArgs.rankSize = 2;
    g_commArgs.localRankSize = 2;
    g_commArgs.extraFlag = TileXR::ExtraFlag::TOPO_910A5;
    g_commArgs.peerMems[0] = reinterpret_cast<GM_ADDR>(UINTPTR_C(0x200000));
    g_commArgs.peerMems[1] = reinterpret_cast<GM_ADDR>(UINTPTR_C(0x300000));
    g_registry = TileXR::TileXRUDMARegistry {};
    g_registry.rankSize = 2;
    g_registry.regionCount = 1;
}

TileXRMoonEpReduceGradWorkspaceInfoV2 Query(
    TileXRMoonEpPlanV1 *plan, TileXRMoonEpTensorV1 *gate,
    TileXRMoonEpTensorV1 *up, TileXRMoonEpTensorV1 *down, int expected)
{
    TileXRMoonEpReduceGradWorkspaceQueryV2 query {};
    query.structSize = sizeof(query);
    query.abiVersion = TILEXR_MOONEP_ABI_VERSION_V2;
    query.comm = reinterpret_cast<TileXRCommPtr>(UINTPTR_C(0x1000));
    query.plan = plan;
    query.gate = gate;
    query.up = up;
    query.down = down;
    TileXRMoonEpReduceGradWorkspaceInfoV2 info {};
    info.structSize = sizeof(info);
    info.abiVersion = TILEXR_MOONEP_ABI_VERSION_V2;
    CheckStatus("workspace query", TileXRMoonEpReduceGradGetWorkspaceSizeV2(&query, &info), expected);
    return info;
}

TileXRMoonEpReduceGradArgsV2 Args(TileXRMoonEpPlanV1 *plan,
    TileXRMoonEpTensorV1 *gate, TileXRMoonEpTensorV1 *up,
    TileXRMoonEpTensorV1 *down, TileXRMoonEpTensorV1 *status)
{
    TileXRMoonEpReduceGradArgsV2 args {};
    args.structSize = sizeof(args);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V2;
    args.comm = reinterpret_cast<TileXRCommPtr>(UINTPTR_C(0x1000));
    args.plan = plan;
    args.gate = gate;
    args.up = up;
    args.down = down;
    args.status = status;
    args.waitIterations = 1000;
    return args;
}

void TestPeerOnly()
{
    Reset();
    TileXRMoonEpPlanV1 plan = Plan();
    TileXRMoonEpTensorV1 gate = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x400000)), 64);
    TileXRMoonEpTensorV1 up = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x500000)), 128);
    TileXRMoonEpTensorV1 down = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x600000)), 256);
    TileXRMoonEpTensorV1 status = Status();
    const auto info = Query(&plan, &gate, &up, &down, TileXR::TILEXR_SUCCESS);
    Check(info.workspaceBytes == 0 && info.udmaChunkBytes == 0,
        "peer-only query must not request UDMA workspace");
    Check(info.transports[0] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER &&
        info.transports[1] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER &&
        info.transports[2] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER,
        "peer-only query selected the wrong transport");

    auto args = Args(&plan, &gate, &up, &down, &status);
    aclrtStream stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x700000));
    CheckStatus("peer launch", TileXRMoonEpReduceGradV2(&args, stream), TileXR::TILEXR_SUCCESS);
    Check(g_memsetCalls == 1 && g_launchCalls == 1,
        "peer launch must initialize status and launch exactly once");
}

void TestMixedUdma()
{
    Reset();
    g_commArgs.localRankSize = 1;
    TileXRMoonEpPlanV1 plan = Plan();
    const uint64_t thresholdElements = TILEXR_MOONEP_REDUCE_GRAD_UDMA_THRESHOLD_BYTES / sizeof(float);
    TileXRMoonEpTensorV1 gate = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x400000)), thresholdElements);
    TileXRMoonEpTensorV1 up = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x500000)), thresholdElements + 1);
    TileXRMoonEpTensorV1 down = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x600000)), 256);
    TileXRMoonEpTensorV1 status = Status();
    const auto info = Query(&plan, &gate, &up, &down, TileXR::TILEXR_SUCCESS);
    Check(info.transports[0] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER &&
        info.transports[1] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA,
        "mixed query threshold selection mismatch");
    Check(info.workspaceBytes > 0 && info.workspaceAlignment == 512,
        "mixed query must return aligned UDMA workspace");

    void *workspace = reinterpret_cast<void *>(UINTPTR_C(0x1000000));
    g_commArgs.extraFlag |= TileXR::ExtraFlag::UDMA;
    g_commArgs.udmaInfoPtr = reinterpret_cast<GM_ADDR>(UINTPTR_C(0x17000));
    g_commArgs.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(UINTPTR_C(0x18000));
    for (int rank = 0; rank < 2; ++rank) {
        g_registry.regions[rank].base = rank == 0 ? static_cast<GM_ADDR>(workspace) :
            reinterpret_cast<GM_ADDR>(UINTPTR_C(0x2000000));
        g_registry.regions[rank].bytes = info.workspaceBytes;
    }

    auto args = Args(&plan, &gate, &up, &down, &status);
    args.workspace = workspace;
    args.workspaceBytes = info.workspaceBytes;
    aclrtStream stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x700000));
    CheckStatus("mixed launch", TileXRMoonEpReduceGradV2(&args, stream), TileXR::TILEXR_SUCCESS);
    Check(g_launchCalls == 1 &&
        g_launchedLayout.transports[TileXRMoonEp::kReduceGradUp] ==
            TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA,
        "cross-node mixed launch did not preserve transport selection");

    g_commArgs.extraFlag &= ~TileXR::ExtraFlag::UDMA;
    CheckStatus("missing UDMA", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    Check(g_launchCalls == 1, "missing UDMA must not launch");
}

void TestValidation()
{
    Reset();
    TileXRMoonEpPlanV1 plan = Plan();
    TileXRMoonEpTensorV1 gate = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x400000)), 64);
    TileXRMoonEpTensorV1 up = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x500000)), 64);
    TileXRMoonEpTensorV1 down = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x600000)), 64);
    TileXRMoonEpTensorV1 status = Status();
    auto args = Args(&plan, &gate, &up, &down, &status);
    aclrtStream stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x700000));

    gate.dtype = TILEXR_MOONEP_DTYPE_FLOAT16;
    CheckStatus("gradient dtype", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    gate.dtype = TILEXR_MOONEP_DTYPE_FLOAT32;
    g_commArgs.peerMems[1] = nullptr;
    CheckStatus("missing peer", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    g_commArgs.peerMems[1] = reinterpret_cast<GM_ADDR>(UINTPTR_C(0x300000));
    g_commArgs.extraFlag = 0;
    CheckStatus("wrong architecture", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_NOT_SUPPORT);

    Reset();
    plan = Plan();
    plan.e = std::numeric_limits<int64_t>::max();
    plan.b = std::numeric_limits<int64_t>::max();
    plan.world = 1;
    g_commArgs.rankSize = 1;
    g_commArgs.localRankSize = 1;
    (void)Query(&plan, &gate, &up, &down, TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&commArgs)
{
    commArgs = g_commHostReturn == TileXR::TILEXR_SUCCESS ? &g_commArgs : nullptr;
    return g_commHostReturn;
}

extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &commArgs)
{
    commArgs = g_commDevReturn == TileXR::TILEXR_SUCCESS ? g_commDev : nullptr;
    return g_commDevReturn;
}

extern "C" int TileXRGetUDMARegistryHost(TileXRCommPtr,
    const TileXR::TileXRUDMARegistry **registry)
{
    if (registry != nullptr) {
        *registry = g_registryReturn == TileXR::TILEXR_SUCCESS ? &g_registry : nullptr;
    }
    return g_registryReturn;
}

extern "C" aclError aclrtMemsetAsync(void *, size_t, int32_t, size_t, aclrtStream)
{
    ++g_memsetCalls;
    return ACL_SUCCESS;
}

extern "C" aclError aclrtMemcpyAsync(void *, size_t, const void *, size_t,
    aclrtMemcpyKind, aclrtStream)
{
    return ACL_SUCCESS;
}

extern "C" aclError aclrtSynchronizeStream(aclrtStream)
{
    return ACL_SUCCESS;
}

namespace TileXRMoonEp {

int TileXRMoonEpLaunchReduceGradKernel(const ReduceGradParams &,
    const ReduceGradLaunchContext &context)
{
    ++g_launchCalls;
    g_launchedLayout = context.layout;
    return g_launchReturn;
}

} // namespace TileXRMoonEp

int main()
{
    TestPeerOnly();
    TestMixedUdma();
    TestValidation();
    return g_failures == 0 ? 0 : 1;
}
