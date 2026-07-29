#include <cstdlib>
#include <cstdint>
#include <iostream>

#include "comm_args.h"
#include "ep_dispatch_host.h"
#include "ep_window.h"
#include "tilexr_udma_reg.h"
#include "tilexr_types.h"

namespace {

TileXR::CommArgs *g_commArgs = nullptr;
GM_ADDR g_devArgs = reinterpret_cast<GM_ADDR>(0x70000000);
const TileXR::TileXRUDMARegistry *g_registry = nullptr;
int g_registryCalls = 0;

} // namespace

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&commArgsPtr)
{
    commArgsPtr = g_commArgs;
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &commArgsPtr)
{
    commArgsPtr = g_devArgs;
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRGetUDMARegistryHost(TileXRCommPtr, const TileXR::TileXRUDMARegistry **registry)
{
    ++g_registryCalls;
    *registry = g_registry;
    return TileXR::TILEXR_SUCCESS;
}

namespace {

int g_failures = 0;

void SetTransportMode(const char *value)
{
#ifdef _WIN32
    _putenv_s("TILEXR_TRANSPORT_MODE", value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("TILEXR_TRANSPORT_MODE");
    } else {
        setenv("TILEXR_TRANSPORT_MODE", value, 1);
    }
#endif
}

void CheckInt(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

TileXREp::EpDispatchParams ValidParams()
{
    static uint16_t x[32] = {};
    static int32_t expertIds[8] = {};
    static uint16_t expandXOut[64] = {};
    static int64_t expertTokenNumsOut[4] = {};
    static int32_t epRecvCountsOut[2] = {};
    static int32_t assistInfoForCombineOut[32] = {};

    TileXREp::EpDispatchParams params {};
    params.x = x;
    params.expertIds = expertIds;
    params.comm = reinterpret_cast<TileXRCommPtr>(0x1000);
    params.bs = 4;
    params.h = 8;
    params.topK = 2;
    params.moeExpertNum = 8;
    params.expandXOut = expandXOut;
    params.expertTokenNumsOut = expertTokenNumsOut;
    params.epRecvCountsOut = epRecvCountsOut;
    params.assistInfoForCombineOut = assistInfoForCombineOut;
    params.dtype = TileXR::TILEXR_DATA_TYPE_FP16;
    params.stream = reinterpret_cast<aclrtStream>(0x2000);
    return params;
}

TileXREp::EpDispatchParams ValidV2Params()
{
    static bool xActiveMask[4] = {true, false, true, true};
    TileXREp::EpDispatchParams params = ValidParams();
    params.xActiveMask = xActiveMask;
    params.epWorldSize = 2;
    params.epRankId = 0;
    params.tpWorldSize = 1;
    params.tpRankId = 0;
    params.expertShardType = 0;
    params.sharedExpertNum = 0;
    params.sharedExpertRankNum = 0;
    params.quantMode = 0;
    params.globalBs = 8;
    params.expertTokenNumsType = 1;
    return params;
}

TileXREp::EpDispatchParams ValidStaticQuantParams()
{
    TileXREp::EpDispatchParams params = ValidV2Params();
    static int8_t expandXOutI8[64] = {};
    static float scales[1] = {1.0f};
    params.scales = scales;
    params.expandXOut = expandXOutI8;
    params.quantMode = 1;
    params.dtype = TileXR::TILEXR_DATA_TYPE_INT8;
    return params;
}

TileXREp::EpDispatchParams ValidPerTokenDynamicQuantParams()
{
    TileXREp::EpDispatchParams params = ValidV2Params();
    static int8_t expandXOutI8[64] = {};
    static float dynamicScalesOut[8] = {};
    params.expandXOut = expandXOutI8;
    params.dynamicScalesOut = dynamicScalesOut;
    params.quantMode = 2;
    params.dtype = TileXR::TILEXR_DATA_TYPE_INT8;
    return params;
}

TileXREp::EpCombineParams ValidCombineParams()
{
    static uint16_t expertOut[64] = {};
    static int32_t assistInfoForCombine[32] = {};
    static int32_t epRecvCounts[2] = {};
    static uint16_t yOut[32] = {};

    TileXREp::EpCombineParams params {};
    params.expertOut = expertOut;
    params.assistInfoForCombine = assistInfoForCombine;
    params.epRecvCounts = epRecvCounts;
    params.comm = reinterpret_cast<TileXRCommPtr>(0x1000);
    params.bs = 4;
    params.h = 8;
    params.topK = 2;
    params.moeExpertNum = 8;
    params.yOut = yOut;
    params.dtype = TileXR::TILEXR_DATA_TYPE_FP16;
    params.stream = reinterpret_cast<aclrtStream>(0x2000);
    return params;
}

TileXR::CommArgs ValidCommArgs()
{
    TileXR::CommArgs args {};
    args.rank = 0;
    args.localRank = 0;
    args.rankSize = 2;
    args.localRankSize = args.rankSize;
    args.peerMems[0] = reinterpret_cast<GM_ADDR>(0x10000000);
    args.peerMems[1] = reinterpret_cast<GM_ADDR>(0x20000000);
    return args;
}

TileXR::CommArgs CrossNodeCommArgs()
{
    TileXR::CommArgs args {};
    args.rank = 0;
    args.localRank = 0;
    args.rankSize = 8;
    args.localRankSize = 4;
    for (int rank = 0; rank < args.rankSize; ++rank) {
        args.peerMems[rank] = reinterpret_cast<GM_ADDR>(
            static_cast<uintptr_t>(0x10000000ULL + static_cast<uint64_t>(rank) * 0x10000000ULL));
    }
    return args;
}

TileXR::CommArgs CrossNodeUdmaCommArgs()
{
    TileXR::CommArgs args = CrossNodeCommArgs();
    args.extraFlag |= TileXR::ExtraFlag::UDMA;
    args.udmaInfoPtr = reinterpret_cast<GM_ADDR>(0x30000000);
    args.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(0x40000000);
    return args;
}

void TestBasicValidation()
{
    TileXREp::EpDispatchParams params = ValidParams();
    CheckInt("valid basic params", TileXREp::TileXREpValidateBasicDispatchParams(params), TileXR::TILEXR_SUCCESS);

    params = ValidParams();
    params.x = nullptr;
    CheckInt("null x", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.comm = nullptr;
    CheckInt("null comm", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.bs = 0;
    CheckInt("zero bs", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.dtype = TileXR::TILEXR_DATA_TYPE_FP32;
    CheckInt("unsupported fp32", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestCommValidation()
{
    TileXREp::EpDispatchParams params = ValidParams();
    TileXR::CommArgs commArgs = ValidCommArgs();
    TileXREp::EpWindowConfig window {};
    CheckInt("valid dispatch config", TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window),
        TileXR::TILEXR_SUCCESS);

    commArgs = ValidCommArgs();
    commArgs.rankSize = 0;
    CheckInt("rank size zero", TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    commArgs = ValidCommArgs();
    commArgs.peerMems[1] = nullptr;
    CheckInt("dispatch shape validation is transport independent",
        TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window), TileXR::TILEXR_SUCCESS);

    commArgs = CrossNodeCommArgs();
    CheckInt("cross-node dispatch supports peer memory without udma",
        TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window), TileXR::TILEXR_SUCCESS);

    commArgs = CrossNodeUdmaCommArgs();
    params = ValidParams();
    static uint8_t workspace[1024] = {};
    params.workspace = workspace;
    CheckInt("cross-node dispatch config", TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window),
        TileXR::TILEXR_SUCCESS);

    commArgs = ValidCommArgs();
    params = ValidParams();
    params.moeExpertNum = 7;
    CheckInt("non-divisible experts", TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestCombineValidation()
{
    TileXREp::EpCombineParams params = ValidCombineParams();
    CheckInt("valid combine basic params", TileXREp::TileXREpValidateBasicCombineParams(params),
        TileXR::TILEXR_SUCCESS);

    params = ValidCombineParams();
    params.expertOut = nullptr;
    CheckInt("combine null expertOut", TileXREp::TileXREpValidateBasicCombineParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidCombineParams();
    params.assistInfoForCombine = nullptr;
    CheckInt("combine null assist", TileXREp::TileXREpValidateBasicCombineParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidCombineParams();
    params.epRecvCounts = nullptr;
    CheckInt("combine null recv counts", TileXREp::TileXREpValidateBasicCombineParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidCombineParams();
    params.yOut = nullptr;
    CheckInt("combine null yOut", TileXREp::TileXREpValidateBasicCombineParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidCombineParams();
    TileXR::CommArgs commArgs = ValidCommArgs();
    TileXREp::EpWindowConfig window {};
    CheckInt("valid combine config", TileXREp::TileXREpValidateCombineConfig(params, commArgs, &window),
        TileXR::TILEXR_SUCCESS);

    commArgs = ValidCommArgs();
    commArgs.peerMems[1] = nullptr;
    CheckInt("combine shape validation is transport independent",
        TileXREp::TileXREpValidateCombineConfig(params, commArgs, &window), TileXR::TILEXR_SUCCESS);

    commArgs = CrossNodeCommArgs();
    CheckInt("cross-node combine supports peer memory without udma",
        TileXREp::TileXREpValidateCombineConfig(params, commArgs, &window), TileXR::TILEXR_SUCCESS);

    commArgs = CrossNodeUdmaCommArgs();
    CheckInt("cross-node combine shape validation is transport independent",
        TileXREp::TileXREpValidateCombineConfig(params, commArgs, &window), TileXR::TILEXR_SUCCESS);

    params.workspace = reinterpret_cast<void *>(0x50000000);
    CheckInt("cross-node combine config", TileXREp::TileXREpValidateCombineConfig(params, commArgs, &window),
        TileXR::TILEXR_SUCCESS);
}

void FillRegistry(TileXR::TileXRUDMARegistry *registry, const TileXR::CommArgs &args, void *localWorkspace)
{
    *registry = TileXR::TileXRUDMARegistry {};
    registry->rankSize = static_cast<uint32_t>(args.rankSize);
    registry->regionCount = 1;
    for (int rank = 0; rank < args.rankSize; ++rank) {
        registry->regions[rank].base = rank == args.rank ? static_cast<GM_ADDR>(localWorkspace) :
            reinterpret_cast<GM_ADDR>(
                static_cast<uintptr_t>(0x80000000ULL + static_cast<uint64_t>(rank) * 0x10000000ULL));
        registry->regions[rank].bytes = 1024ULL * 1024ULL * 1024ULL;
    }
}

void TestLaunchContextUsesResolvedTransport()
{
    TileXREp::EpDispatchParams params = ValidParams();
    TileXR::CommArgs args = CrossNodeUdmaCommArgs();
    g_commArgs = &args;
    g_registry = nullptr;
    g_registryCalls = 0;
    SetTransportMode("memory");

    TileXREp::EpHostLaunchContext context {};
    CheckInt("forced memory launch context", TileXREp::TileXREpPrepareLaunchContext(params, &context),
        TileXR::TILEXR_SUCCESS);
    CheckInt("forced memory saved route", static_cast<int>(context.transport),
        static_cast<int>(TileXR::TileXRTransportKind::MEMORY));
    CheckInt("forced memory skips registry", g_registryCalls, 0);

    args.peerMems[7] = nullptr;
    CheckInt("forced memory requires every peer mapping", TileXREp::TileXREpPrepareLaunchContext(params, &context),
        TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    args.peerMems[7] = reinterpret_cast<GM_ADDR>(0x80000000);

    SetTransportMode("direct_urma");
    args = CrossNodeCommArgs();
    g_commArgs = &args;
    CheckInt("forced direct requires capability", TileXREp::TileXREpPrepareLaunchContext(params, &context),
        TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    CheckInt("missing capability skips registry", g_registryCalls, 0);

    args = CrossNodeUdmaCommArgs();
    args.peerMems[4] = nullptr;
    g_commArgs = &args;
    static uint8_t workspace[1024] = {};
    params.workspace = workspace;
    TileXR::TileXRUDMARegistry registry {};
    FillRegistry(&registry, args, workspace);
    g_registry = &registry;
    TileXREp::EpWindowConfig directWindow {};
    CheckInt("build direct dispatch window", TileXREp::TileXREpValidateDispatchConfig(params, args, &directWindow),
        TileXR::TILEXR_SUCCESS);
    const uint64_t legacyDispatchBytes = static_cast<uint64_t>(
        TileXREp::TileXREpAlignUp(directWindow.totalBytes, TileXREp::kEpWindowAlignmentBytes) * 3);
    for (int rank = 0; rank < args.rankSize; ++rank) {
        registry.regions[rank].bytes = legacyDispatchBytes;
    }
    CheckInt("direct dispatch rejects workspace without relay and status area",
        TileXREp::TileXREpPrepareLaunchContext(params, &context), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    FillRegistry(&registry, args, workspace);
    g_registryCalls = 0;
    CheckInt("forced direct launch context", TileXREp::TileXREpPrepareLaunchContext(params, &context),
        TileXR::TILEXR_SUCCESS);
    CheckInt("forced direct saved route", static_cast<int>(context.transport),
        static_cast<int>(TileXR::TileXRTransportKind::DIRECT_URMA));
    CheckInt("forced direct checks registry", g_registryCalls, 1);

    args.peerMems[1] = nullptr;
    g_registryCalls = 0;
    CheckInt("forced direct requires local peer mapping", TileXREp::TileXREpPrepareLaunchContext(params, &context),
        TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    CheckInt("missing local peer skips registry", g_registryCalls, 0);

    SetTransportMode("invalid");
    CheckInt("invalid transport mode", TileXREp::TileXREpPrepareLaunchContext(params, &context),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    SetTransportMode(nullptr);
}

void TestCombineContextUsesRouteAwarePeerMappings()
{
    TileXREp::EpCombineParams params = ValidCombineParams();
    static uint8_t workspace[1024] = {};
    params.workspace = workspace;
    TileXR::CommArgs args = CrossNodeUdmaCommArgs();
    args.peerMems[4] = nullptr;
    g_commArgs = &args;
    TileXR::TileXRUDMARegistry registry {};
    FillRegistry(&registry, args, workspace);
    g_registry = &registry;
    g_registryCalls = 0;
    SetTransportMode("direct_urma");

    TileXREp::EpHostLaunchContext context {};
    CheckInt("direct combine ignores remote peer mapping",
        TileXREp::TileXREpPrepareCombineLaunchContext(params, &context), TileXR::TILEXR_SUCCESS);
    CheckInt("direct combine checks registry", g_registryCalls, 1);

    SetTransportMode("memory");
    g_registryCalls = 0;
    CheckInt("memory combine requires remote peer mapping",
        TileXREp::TileXREpPrepareCombineLaunchContext(params, &context), TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    CheckInt("memory combine skips registry", g_registryCalls, 0);
    SetTransportMode(nullptr);
}

void TestDirectTransportChecksRegistryOnSameNode()
{
    TileXREp::EpDispatchParams params = ValidParams();
    static uint8_t workspace[1024] = {};
    params.workspace = workspace;
    TileXR::CommArgs args = ValidCommArgs();
    args.extraFlag |= TileXR::ExtraFlag::UDMA;
    args.udmaInfoPtr = reinterpret_cast<GM_ADDR>(0x30000000);
    args.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(0x40000000);
    g_commArgs = &args;
    TileXR::TileXRUDMARegistry registry {};
    FillRegistry(&registry, args, workspace);
    g_registry = &registry;
    g_registryCalls = 0;
    SetTransportMode("direct_urma");

    TileXREp::EpHostLaunchContext context {};
    CheckInt("same-node direct launch context", TileXREp::TileXREpPrepareLaunchContext(params, &context),
        TileXR::TILEXR_SUCCESS);
    CheckInt("same-node direct checks registry", g_registryCalls, 1);
    CheckInt("same-node direct saved route", static_cast<int>(context.transport),
        static_cast<int>(TileXR::TileXRTransportKind::DIRECT_URMA));
    SetTransportMode(nullptr);
}

void TestAutoWithoutWorkspaceStaysOnMemory()
{
    TileXREp::EpDispatchParams params = ValidParams();
    params.bs = 64;
    params.h = 128;
    params.workspace = nullptr;
    TileXR::CommArgs args = CrossNodeUdmaCommArgs();
    g_commArgs = &args;
    g_registry = nullptr;
    g_registryCalls = 0;
    SetTransportMode("auto");

    TileXREp::EpHostLaunchContext context {};
    CheckInt("auto without workspace uses memory",
        TileXREp::TileXREpPrepareLaunchContext(params, &context), TileXR::TILEXR_SUCCESS);
    CheckInt("auto without workspace saves memory route", static_cast<int>(context.transport),
        static_cast<int>(TileXR::TileXRTransportKind::MEMORY));
    CheckInt("auto without workspace skips registry", g_registryCalls, 0);

    TileXREp::EpCombineParams combineParams = ValidCombineParams();
    combineParams.bs = 64;
    combineParams.h = 128;
    combineParams.workspace = nullptr;
    g_registryCalls = 0;
    CheckInt("auto combine without workspace uses memory",
        TileXREp::TileXREpPrepareCombineLaunchContext(combineParams, &context), TileXR::TILEXR_SUCCESS);
    CheckInt("auto combine without workspace saves memory route", static_cast<int>(context.transport),
        static_cast<int>(TileXR::TileXRTransportKind::MEMORY));
    CheckInt("auto combine without workspace skips registry", g_registryCalls, 0);
    SetTransportMode(nullptr);
}

void TestV2CapabilityValidation()
{
    TileXREp::EpDispatchParams params = ValidV2Params();
    TileXR::CommArgs commArgs = ValidCommArgs();
    CheckInt("valid v2 params", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs), TileXR::TILEXR_SUCCESS);

    params = ValidV2Params();
    params.tpWorldSize = 0;
    CheckInt("v2 no tp world zero", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_SUCCESS);

    params = ValidV2Params();
    static int32_t tpRecvCountsOut[2] = {};
    params.tpRecvCountsOut = tpRecvCountsOut;
    CheckInt("v2 supports tp recv counts output for tp size one",
        TileXREp::TileXREpValidateDispatchV2Config(params, commArgs), TileXR::TILEXR_SUCCESS);

    params = ValidV2Params();
    params.expertTokenNumsType = 0;
    CheckInt("v2 supports prefix-sum expert token counts", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_SUCCESS);

    params = ValidV2Params();
    params.expertTokenNumsType = 2;
    CheckInt("v2 rejects unsupported expert token count type", TileXREp::TileXREpValidateDispatchV2Config(params,
        commArgs), TileXR::TILEXR_ERROR_NOT_SUPPORT);

    params = ValidV2Params();
    params.quantMode = 1;
    CheckInt("v2 static quant requires int8 output", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidStaticQuantParams();
    CheckInt("v2 supports static quant int8", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_SUCCESS);

    params = ValidStaticQuantParams();
    params.scales = nullptr;
    CheckInt("v2 static quant requires scales", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidPerTokenDynamicQuantParams();
    CheckInt("v2 supports per-token dynamic quant int8",
        TileXREp::TileXREpValidateDispatchV2Config(params, commArgs), TileXR::TILEXR_SUCCESS);

    params = ValidPerTokenDynamicQuantParams();
    params.dynamicScalesOut = nullptr;
    CheckInt("v2 per-token dynamic quant requires dynamic scales output",
        TileXREp::TileXREpValidateDispatchV2Config(params, commArgs), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidPerTokenDynamicQuantParams();
    params.scales = reinterpret_cast<void *>(0x5000);
    CheckInt("v2 per-token dynamic quant does not take input scales",
        TileXREp::TileXREpValidateDispatchV2Config(params, commArgs), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidV2Params();
    params.quantMode = 3;
    CheckInt("v2 rejects unsupported quant modes after per-token dynamic quant",
        TileXREp::TileXREpValidateDispatchV2Config(params, commArgs), TileXR::TILEXR_ERROR_NOT_SUPPORT);

    params = ValidV2Params();
    params.tpWorldSize = 2;
    params.epWorldSize = 1;
    static int32_t tpRecvCountsOutForTp[2] = {};
    params.tpRecvCountsOut = tpRecvCountsOutForTp;
    CheckInt("v2 supports same-node tp size two", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_SUCCESS);

    params = ValidV2Params();
    params.tpWorldSize = 2;
    params.epWorldSize = 1;
    params.tpRankId = 2;
    params.tpRecvCountsOut = tpRecvCountsOutForTp;
    CheckInt("v2 rejects invalid tp rank", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidV2Params();
    params.tpWorldSize = 2;
    params.epWorldSize = 1;
    params.tpRecvCountsOut = nullptr;
    CheckInt("v2 tp size two requires tp recv counts", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidV2Params();
    static int32_t tpRecvCountsOutForTp4[4] = {};
    params.tpWorldSize = 4;
    params.epWorldSize = 1;
    params.globalBs = 16;
    params.tpRecvCountsOut = tpRecvCountsOutForTp4;
    commArgs.rankSize = 4;
    commArgs.localRankSize = 4;
    commArgs.peerMems[2] = reinterpret_cast<GM_ADDR>(0x30000000);
    commArgs.peerMems[3] = reinterpret_cast<GM_ADDR>(0x40000000);
    CheckInt("v2 supports same-node tp size four", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_SUCCESS);

    params = ValidV2Params();
    params.sharedExpertNum = 1;
    CheckInt("v2 rejects shared expert", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_ERROR_NOT_SUPPORT);

    params = ValidV2Params();
    params.epWorldSize = 4;
    params.globalBs = 16;
    params.sharedExpertNum = 1;
    params.sharedExpertRankNum = 1;
    params.moeExpertNum = 6;
    commArgs.rankSize = 4;
    CheckInt("v2 supports shared expert on leading ep ranks",
        TileXREp::TileXREpValidateDispatchV2Config(params, commArgs), TileXR::TILEXR_SUCCESS);

    params = ValidV2Params();
    static int32_t tpRecvCountsOutForSharedTp[8] = {};
    params.epWorldSize = 4;
    params.globalBs = 32;
    params.tpWorldSize = 2;
    params.tpRankId = 0;
    params.tpRecvCountsOut = tpRecvCountsOutForSharedTp;
    params.sharedExpertNum = 1;
    params.sharedExpertRankNum = 1;
    params.moeExpertNum = 6;
    commArgs.rankSize = 8;
    commArgs.localRankSize = 8;
    for (int rank = 0; rank < commArgs.rankSize; ++rank) {
        commArgs.peerMems[rank] = reinterpret_cast<GM_ADDR>(
            static_cast<uintptr_t>(0x10000000ULL + static_cast<uint64_t>(rank) * 0x10000000ULL));
    }
    CheckInt("v2 supports tp with shared expert",
        TileXREp::TileXREpValidateDispatchV2Config(params, commArgs), TileXR::TILEXR_SUCCESS);

    params = ValidV2Params();
    params.expertShardType = 1;
    CheckInt("v2 rejects shard type", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_ERROR_NOT_SUPPORT);

    params = ValidV2Params();
    params.epWorldSize = 4;
    CheckInt("v2 rejects ep world mismatch", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidV2Params();
    params.epRankId = 1;
    CheckInt("v2 rejects ep rank mismatch", TileXREp::TileXREpValidateDispatchV2Config(params, commArgs),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestBasicValidation();
    TestCommValidation();
    TestCombineValidation();
    TestLaunchContextUsesResolvedTransport();
    TestCombineContextUsesRouteAwarePeerMappings();
    TestDirectTransportChecksRegistryOnSameNode();
    TestAutoWithoutWorkspaceStaysOnMemory();
    TestV2CapabilityValidation();
    return g_failures == 0 ? 0 : 1;
}
