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
int g_synchronizeCalls = 0;
aclError g_synchronizeReturn = ACL_SUCCESS;
int g_commHostReturn = TileXR::TILEXR_SUCCESS;
int g_commDevReturn = TileXR::TILEXR_SUCCESS;
int g_registryReturn = TileXR::TILEXR_SUCCESS;
int g_qpCountReturn = TileXR::TILEXR_SUCCESS;
int g_launchReturn = TileXR::TILEXR_SUCCESS;
int g_qpCountCalls = 0;
uint32_t g_qpCount = 3;
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
    plan.n = 4;
    plan.k = 2;
    plan.e = 4;
    plan.b = 2;
    plan.r = 2;
    plan.nvS = 4;
    plan.dst = reinterpret_cast<void *>(UINTPTR_C(0x11000));
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
    g_synchronizeCalls = 0;
    g_synchronizeReturn = ACL_SUCCESS;
    g_commHostReturn = TileXR::TILEXR_SUCCESS;
    g_commDevReturn = TileXR::TILEXR_SUCCESS;
    g_registryReturn = TileXR::TILEXR_SUCCESS;
    g_qpCountReturn = TileXR::TILEXR_SUCCESS;
    g_launchReturn = TileXR::TILEXR_SUCCESS;
    g_qpCountCalls = 0;
    g_qpCount = 3;
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
    Check(g_qpCountCalls == 0,
        "peer-only query and launch must not query UDMA QPs");
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
    Check(info.workspaceBytes > 0 &&
        info.workspaceBytes % TileXRMoonEp::kReduceGradUdmaWorkspaceAlignment == 0 &&
        info.workspaceAlignment == TileXRMoonEp::kReduceGradUdmaWorkspaceAlignment,
        "mixed query must return aligned UDMA workspace");
    Check(g_qpCountCalls == 1,
        "mixed query must obtain the negotiated UDMA QP count");

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
            TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA &&
        g_launchedLayout.udmaQpCount == g_qpCount,
        "cross-node mixed launch did not preserve transport selection");
    Check(g_qpCountCalls == 2,
        "mixed launch must obtain the negotiated UDMA QP count");

    g_registry.regions[1].base += 512;
    CheckStatus("misaligned peer workspace", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    Check(g_launchCalls == 1, "misaligned peer workspace must not launch");
    g_registry.regions[1].base -= 512;
    g_registry.regions[1].bytes += TileXRMoonEp::kReduceGradUdmaWorkspaceAlignment;
    CheckStatus("mismatched peer workspace bytes", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    Check(g_launchCalls == 1, "mismatched peer workspace bytes must not launch");
    g_registry.regions[1].bytes = info.workspaceBytes;

    g_commArgs.extraFlag &= ~TileXR::ExtraFlag::UDMA;
    CheckStatus("missing UDMA", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    Check(g_launchCalls == 1, "missing UDMA must not launch");

    g_commArgs.extraFlag |= TileXR::ExtraFlag::UDMA;
    const int qpCallsBeforeFailureChecks = g_qpCountCalls;
    g_qpCountReturn = -92;
    (void)Query(&plan, &gate, &up, &down, -92);
    Check(g_qpCountCalls == qpCallsBeforeFailureChecks + 1,
        "UDMA QP query failure must be observed by workspace preparation");

    g_qpCountReturn = TileXR::TILEXR_SUCCESS;
    g_qpCount = 0;
    (void)Query(&plan, &gate, &up, &down, TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    g_qpCount = TileXRMoonEp::kReduceGradMaxUdmaQpCount + 1;
    (void)Query(&plan, &gate, &up, &down, TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    Check(g_qpCountCalls == qpCallsBeforeFailureChecks + 3,
        "invalid UDMA QP counts must be validated after every query");
}

void TestSingleRankLargeRowsDoNotRequireUdma()
{
    Reset();
    g_commArgs.rankSize = 1;
    g_commArgs.localRankSize = 1;
    g_registryReturn = TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    TileXRMoonEpPlanV1 plan = Plan();
    plan.r = 1;
    plan.b = plan.e;
    const uint64_t largeRow =
        TILEXR_MOONEP_REDUCE_GRAD_UDMA_THRESHOLD_BYTES / sizeof(float) + 1;
    TileXRMoonEpTensorV1 gate = Gradient(
        reinterpret_cast<void *>(UINTPTR_C(0x400000)), largeRow);
    TileXRMoonEpTensorV1 up = Gradient(
        reinterpret_cast<void *>(UINTPTR_C(0x500000)), largeRow + 1);
    TileXRMoonEpTensorV1 down = Gradient(
        reinterpret_cast<void *>(UINTPTR_C(0x600000)), largeRow + 2);
    TileXRMoonEpTensorV1 *gradients[] = {&gate, &up, &down};
    for (TileXRMoonEpTensorV1 *tensor : gradients) {
        tensor->shape[0] = plan.e + plan.b;
        tensor->elementCount = static_cast<uint64_t>(tensor->shape[0]) *
            static_cast<uint64_t>(tensor->shape[1]);
    }
    TileXRMoonEpTensorV1 status = Status();
    const auto info = Query(&plan, &gate, &up, &down, TileXR::TILEXR_SUCCESS);
    Check(info.workspaceBytes == 0 && info.udmaChunkBytes == 0,
        "single-rank large-row query must not request UDMA workspace");
    Check(info.transports[0] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER &&
        info.transports[1] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER &&
        info.transports[2] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER,
        "single-rank large-row query must select the local peer path");

    auto args = Args(&plan, &gate, &up, &down, &status);
    aclrtStream stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x700000));
    CheckStatus("single-rank large-row launch", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_SUCCESS);
    Check(g_launchCalls == 1,
        "single-rank large-row launch must not consult an unavailable UDMA registry");
}

void TestLargeRankWorkspaceQuery()
{
    Reset();
    g_commArgs.rankSize = TileXR::TILEXR_MAX_RANK_SIZE;
    g_commArgs.localRankSize = 8;
    for (int rank = 0; rank < g_commArgs.rankSize; ++rank) {
        g_commArgs.peerMems[rank] = reinterpret_cast<GM_ADDR>(
            UINTPTR_C(0x200000) + static_cast<uintptr_t>(rank) * UINTPTR_C(0x100000));
    }

    TileXRMoonEpPlanV1 plan = Plan();
    plan.n = TileXR::TILEXR_MAX_RANK_SIZE;
    plan.k = 1;
    plan.e = TileXR::TILEXR_MAX_RANK_SIZE;
    plan.b = 1;
    plan.r = TileXR::TILEXR_MAX_RANK_SIZE;
    TileXRMoonEpTensorV1 gate = Gradient(
        reinterpret_cast<void *>(UINTPTR_C(0x400000)), 64);
    TileXRMoonEpTensorV1 up = Gradient(
        reinterpret_cast<void *>(UINTPTR_C(0x500000)), 128);
    TileXRMoonEpTensorV1 down = Gradient(
        reinterpret_cast<void *>(UINTPTR_C(0x600000)), 256);
    TileXRMoonEpTensorV1 *gradients[] = {&gate, &up, &down};
    for (TileXRMoonEpTensorV1 *tensor : gradients) {
        tensor->shape[0] = plan.e + plan.b;
        tensor->elementCount = static_cast<uint64_t>(tensor->shape[0]) *
            static_cast<uint64_t>(tensor->shape[1]);
    }

    const auto info = Query(&plan, &gate, &up, &down, TileXR::TILEXR_SUCCESS);
    Check(info.blockDim == TileXRMoonEp::kReduceGradMaxAivBlockCount,
        "128-rank workspace query must retain the 64-AIV launch limit");
}

void TestLaunchFailureDrainsEnqueuedStatusReset()
{
    Reset();
    TileXRMoonEpPlanV1 plan = Plan();
    TileXRMoonEpTensorV1 gate = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x400000)), 64);
    TileXRMoonEpTensorV1 up = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x500000)), 64);
    TileXRMoonEpTensorV1 down = Gradient(reinterpret_cast<void *>(UINTPTR_C(0x600000)), 64);
    TileXRMoonEpTensorV1 status = Status();
    auto args = Args(&plan, &gate, &up, &down, &status);
    aclrtStream stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x700000));

    g_launchReturn = TileXR::TILEXR_ERROR_NOT_SUPPORT;
    CheckStatus("launch failure", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_NOT_SUPPORT);
    Check(g_memsetCalls == 1 && g_launchCalls == 1 && g_synchronizeCalls == 1,
        "launch failure must drain the enqueued status reset");

    g_synchronizeReturn = -91;
    CheckStatus("launch failure with drain failure", TileXRMoonEpReduceGradV2(&args, stream),
        TileXR::TILEXR_ERROR_MKIRT);
    Check(g_memsetCalls == 2 && g_launchCalls == 2 && g_synchronizeCalls == 2,
        "stream drain failure must be reported after a failed launch");
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
    plan.r = 1;
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

extern "C" int TileXRUDMAGetQpCount(TileXRCommPtr, uint32_t *qpCount)
{
    ++g_qpCountCalls;
    if (qpCount != nullptr) {
        *qpCount = g_qpCount;
    }
    return g_qpCountReturn;
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
    ++g_synchronizeCalls;
    return g_synchronizeReturn;
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
    TestSingleRankLargeRowsDoNotRequireUdma();
    TestLargeRankWorkspaceQuery();
    TestLaunchFailureDrainsEnqueuedStatusReset();
    TestValidation();
    return g_failures == 0 ? 0 : 1;
}
