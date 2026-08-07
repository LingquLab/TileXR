#include <cstdint>
#include <iostream>

#include "comm_args.h"
#include "planner_host.h"
#include "planner_layout.h"
#include "tilexr_types.h"

namespace {

int failures = 0;
TileXR::CommArgs *gHostArgs = nullptr;
GM_ADDR gDeviceArgs = nullptr;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++failures;
    }
}

TileXR::CommArgs MakeCommArgs(int32_t rankSize, int32_t rank,
    int32_t localRankSize, int32_t localRank)
{
    TileXR::CommArgs args {};
    args.rankSize = rankSize;
    args.rank = rank;
    args.localRankSize = localRankSize;
    args.localRank = localRank;
    args.extraFlag = TileXR::ExtraFlag::TOPO_910A5;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        args.peerMems[peer] = reinterpret_cast<GM_ADDR>(
            static_cast<uintptr_t>(0x100000 + peer * 0x1000));
    }
    return args;
}

TileXRMoonEp::PlannerParams MakeParams(TileXRCommPtr comm, uint64_t workspaceBytes)
{
    TileXRMoonEp::PlannerParams params {};
    params.topkExpertIds = reinterpret_cast<const int32_t *>(0x1000);
    params.tokensPerExpert = reinterpret_cast<const int32_t *>(0x2000);
    params.comm = comm;
    params.s = 64;
    params.k = 4;
    params.expertCount = 64;
    params.b = 8;
    params.tokenPadding = 4;
    params.workspace = reinterpret_cast<void *>(0x3000);
    params.workspaceBytes = workspaceBytes;
    params.dst = reinterpret_cast<int32_t *>(0x4000);
    params.cuSeqlens = reinterpret_cast<int32_t *>(0x5000);
    params.expertsToCopy = reinterpret_cast<int32_t *>(0x6000);
    params.zeroFillRanges = reinterpret_cast<int32_t *>(0x6800);
    params.remoteStats = reinterpret_cast<int32_t *>(0x7000);
    params.dupCounts = reinterpret_cast<int32_t *>(0x7800);
    params.plannerStatus = reinterpret_cast<int32_t *>(0x8000);
    params.waitIterations = 1000;
    params.stream = reinterpret_cast<aclrtStream>(0x9000);
    return params;
}

void TestSameNodeOnlyValidation()
{
    const TileXRCommPtr comm = reinterpret_cast<TileXRCommPtr>(0x1);
    TileXRMoonEp::PlannerLayout layout {};

    TileXR::CommArgs sameNode = MakeCommArgs(8, 3, 8, 3);
    gHostArgs = &sameNode;
    Check(TileXRMoonEp::TileXRMoonEpPrepareLayout(comm, 64, 4, 64, 8, 4, &layout) ==
        TileXR::TILEXR_SUCCESS, "same-node Planner layout rejected");

    TileXR::CommArgs crossNode = MakeCommArgs(16, 9, 8, 1);
    gHostArgs = &crossNode;
    Check(TileXRMoonEp::TileXRMoonEpPrepareLayout(comm, 64, 4, 64, 4, 4, &layout) ==
        TileXR::TILEXR_ERROR_NOT_SUPPORT, "cross-node Planner layout accepted");

    crossNode = MakeCommArgs(16, 9, 8, 8);
    gHostArgs = &crossNode;
    Check(TileXRMoonEp::TileXRMoonEpPrepareLayout(comm, 64, 4, 64, 4, 4, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "invalid locality accepted");

    crossNode = MakeCommArgs(16, 9, 8, 1);
    crossNode.extraFlag = 0;
    gHostArgs = &crossNode;
    Check(TileXRMoonEp::TileXRMoonEpPrepareLayout(comm, 64, 4, 64, 4, 4, &layout) ==
        TileXR::TILEXR_ERROR_NOT_SUPPORT, "non-A5 communicator accepted");
}

void TestLaunchParameterValidation()
{
    const TileXRCommPtr comm = reinterpret_cast<TileXRCommPtr>(0x1);
    TileXR::CommArgs args = MakeCommArgs(8, 0, 8, 0);
    TileXRMoonEp::PlannerLayout layout {};
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(8, 64, 4, 64, 8, 4, &layout) ==
        TileXR::TILEXR_SUCCESS, "validation layout setup failed");

    TileXRMoonEp::PlannerParams params = MakeParams(comm, layout.workspaceBytes);
    Check(TileXRMoonEp::TileXRMoonEpValidateParams(params, args, layout) ==
        TileXR::TILEXR_SUCCESS, "valid Planner parameters rejected");

    params.waitIterations = 0;
    Check(TileXRMoonEp::TileXRMoonEpValidateParams(params, args, layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "zero wait budget accepted");
    params.waitIterations = 1000;

    params.plannerStatus = nullptr;
    Check(TileXRMoonEp::TileXRMoonEpValidateParams(params, args, layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "null status accepted");
    params.plannerStatus = reinterpret_cast<int32_t *>(0x8000);

    params.zeroFillRanges = nullptr;
    Check(TileXRMoonEp::TileXRMoonEpValidateParams(params, args, layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "null zero-fill ranges accepted");
    params.zeroFillRanges = reinterpret_cast<int32_t *>(0x6800);

    params.workspaceBytes = layout.workspaceBytes - 1;
    Check(TileXRMoonEp::TileXRMoonEpValidateParams(params, args, layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "undersized workspace accepted");

    params.workspaceBytes = layout.workspaceBytes;
    params.workspace = reinterpret_cast<void *>(0x3004);
    Check(TileXRMoonEp::TileXRMoonEpValidateParams(params, args, layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "unaligned workspace accepted");
    params.workspace = reinterpret_cast<void *>(0x3000);

    args.peerMems[7] = nullptr;
    Check(TileXRMoonEp::TileXRMoonEpValidateParams(params, args, layout) ==
        TileXR::TILEXR_ERROR_NOT_INITIALIZED, "missing launch peer window accepted");
}

void TestPrepareLaunchContext()
{
    const TileXRCommPtr comm = reinterpret_cast<TileXRCommPtr>(0x1);
    TileXR::CommArgs args = MakeCommArgs(16, 9, 8, 1);
    TileXRMoonEp::PlannerLayout layout {};
    Check(TileXRMoonEp::TileXRMoonEpBuildPlannerLayout(16, 64, 4, 64, 4, 4, &layout) ==
        TileXR::TILEXR_SUCCESS, "context layout setup failed");
    TileXRMoonEp::PlannerParams params = MakeParams(comm, layout.workspaceBytes);
    params.b = 4;

    gHostArgs = &args;
    gDeviceArgs = reinterpret_cast<GM_ADDR>(0xa000);
    TileXRMoonEp::PlannerLaunchContext context {};
    Check(TileXRMoonEp::TileXRMoonEpPrepareLaunchContext(params, &context) ==
        TileXR::TILEXR_ERROR_NOT_SUPPORT, "cross-node launch context accepted");
    Check(context.hostArgs == nullptr && context.devArgs == nullptr,
        "failed cross-node launch retained CommArgs");
}

} // namespace

int TileXRGetCommArgsHost(TileXRCommPtr comm, TileXR::CommArgs *&commArgsPtr)
{
    if (comm == nullptr || gHostArgs == nullptr) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    commArgsPtr = gHostArgs;
    return TileXR::TILEXR_SUCCESS;
}

int TileXRGetCommArgsDev(TileXRCommPtr comm, GM_ADDR &commArgsPtr)
{
    if (comm == nullptr || gDeviceArgs == nullptr) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    commArgsPtr = gDeviceArgs;
    return TileXR::TILEXR_SUCCESS;
}

int main()
{
    TestSameNodeOnlyValidation();
    TestLaunchParameterValidation();
    TestPrepareLaunchContext();
    return failures == 0 ? 0 : 1;
}
