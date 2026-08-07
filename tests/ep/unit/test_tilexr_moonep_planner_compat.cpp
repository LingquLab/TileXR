#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl_rt.h"
#include "ep_plan_host.h"
#include "planner_launch.h"
#include "tilexr_moonep_planner.h"
#include "tilexr_types.h"

namespace {
int failures = 0;
TileXR::CommArgs commArgs {};
int launchCalls = 0;
int validateCalls = 0;
int syncMemcpyCalls = 0;
int validateRet = TileXR::TILEXR_SUCCESS;
TileXRMoonEPPlanConfig capturedConfig {};
TileXRMoonEPPlanDesc capturedPlan {};
TileXREp::Plan::PlanHostArguments capturedArguments {};
int64_t nextMagic = 100;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        ++failures;
    }
}

void *Align64(std::vector<uint8_t> &storage)
{
    uintptr_t value = reinterpret_cast<uintptr_t>(storage.data());
    value = (value + 63U) & ~static_cast<uintptr_t>(63U);
    return reinterpret_cast<void *>(value);
}
} // namespace

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&out)
{
    out = &commArgs;
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &out)
{
    out = reinterpret_cast<GM_ADDR>(0x1000);
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRCommNextMagic(TileXRCommPtr, int64_t *magic)
{
    *magic = ++nextMagic;
    return TileXR::TILEXR_SUCCESS;
}

extern "C" aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src,
    size_t count, aclrtMemcpyKind, aclrtStream)
{
    if (dst == nullptr || src == nullptr || count > destMax) return 1;
    std::memcpy(dst, src, count);
    return 0;
}

extern "C" aclError aclrtMemcpy(void *dst, size_t destMax, const void *src,
    size_t count, aclrtMemcpyKind)
{
    ++syncMemcpyCalls;
    if (dst == nullptr || src == nullptr || count > destMax) return 1;
    std::memcpy(dst, src, count);
    return 0;
}

namespace TileXREp { namespace Plan {
int ValidatePlanHostArguments(const PlanHostArguments &arguments, const PlanRuntimeMetadata &runtime,
    PlanHostContext *context)
{
    ++validateCalls;
    capturedArguments = arguments;
    capturedConfig = *arguments.config;
    capturedPlan = *arguments.plan;
    Check(runtime.hostCommArgs == &commArgs, "compat wrapper did not forward Host CommArgs");
    Check(runtime.deviceCommArgs == reinterpret_cast<GM_ADDR>(0x1000),
        "compat wrapper did not forward Device CommArgs");
    Check(syncMemcpyCalls == 0, "compat wrapper copied rank ids before validation");
    *context = PlanHostContext {};
    return validateRet;
}

int LaunchPlanKernel(TileXRCommPtr, const PlanHostArguments &, const PlanHostContext &)
{
    ++launchCalls;
    return TileXR::TILEXR_SUCCESS;
}
} }

int main()
{
    commArgs = TileXR::CommArgs {};
    commArgs.rankSize = 8;
    commArgs.rank = 3;
    commArgs.localRankSize = 2;
    commArgs.localRank = 1;
    commArgs.extraFlag = TileXR::ExtraFlag::TOPO_910A5;
    for (int rank = 0; rank < commArgs.rankSize; ++rank) {
        commArgs.peerMems[rank] = reinterpret_cast<GM_ADDR>(
            static_cast<uintptr_t>(0x100000 + rank * 0x10000));
    }

    const TileXRCommPtr comm = reinterpret_cast<TileXRCommPtr>(0x1);
    uint64_t workspaceBytes = 0;
    int64_t capacity = 0;
    Check(TileXRMoonEpPlannerGetWorkspaceSizeV2(comm, 64, 4, 64,
        &workspaceBytes, &capacity) == TileXR::TILEXR_SUCCESS, "workspace query failed");
    Check(workspaceBytes > 0 && workspaceBytes % 64 == 0, "workspace size/alignment invalid");
    Check(capacity == 256, "dispatched capacity mismatch");

    std::vector<uint8_t> storage(static_cast<size_t>(workspaceBytes + 64), 0);
    void *workspace = Align64(storage);
    std::vector<int32_t> topk(256, 0), tpe(64, 0), dst(256, 0), cu(72, 0);
    std::vector<int32_t> copy(64, -1), stats(2, 0), status(1, -1);
    Check(TileXRMoonEpPlannerV2(topk.data(), tpe.data(), comm, 64, 4, 64,
        workspace, workspaceBytes, dst.data(), cu.data(), copy.data(), stats.data(), status.data(),
        1000, reinterpret_cast<aclrtStream>(0x2)) == TileXR::TILEXR_SUCCESS,
        "compat launch failed");
    Check(validateCalls == 1, "compat wrapper did not validate exactly once");
    Check(syncMemcpyCalls == 1, "compat wrapper did not synchronously copy rank ids exactly once");
    Check(launchCalls == 1, "optimized planner was not launched exactly once");
    Check(capturedConfig.prefetchSlots == 8 && capturedConfig.rankTokenCapacity == 256 &&
        capturedConfig.nvS == 256 && capturedConfig.tokenPadding == 1,
        "compatibility config mismatch");
    Check(capturedPlan.r == 8 && capturedPlan.e == 64 && capturedPlan.b == 8 &&
        capturedPlan.cap == 256 && capturedPlan.epoch != 0,
        "compatibility descriptor mismatch");
    Check(capturedArguments.waitIterations == 1000, "compat wait budget was not forwarded");
    Check(capturedArguments.globalRankIds != nullptr, "physical rank map missing");
    const int32_t expectedRankIds[8] = {0, 1, 8, 9, 16, 17, 24, 25};
    for (int rank = 0; rank < 8; ++rank) {
        Check(capturedArguments.globalRankIds[rank] == expectedRankIds[rank],
            "physical rank map does not match PR91-compatible locality mapping");
    }
    Check(capturedArguments.registeredMetaWorkspace != nullptr &&
        capturedArguments.registeredMetaBytes > 0,
        "compat wrapper did not carve metadata workspace from caller memory");

    validateRet = TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    syncMemcpyCalls = 0;
    launchCalls = 0;
    Check(TileXRMoonEpPlannerV2(topk.data(), tpe.data(), comm, 64, 4, 64,
        workspace, workspaceBytes, dst.data(), cu.data(), copy.data(), stats.data(), status.data(),
        1000, reinterpret_cast<aclrtStream>(0x2)) == TileXR::TILEXR_ERROR_NOT_INITIALIZED,
        "compat launch must return validation failure");
    Check(syncMemcpyCalls == 0, "validation failure must not copy rank ids");
    Check(launchCalls == 0, "validation failure must not launch the kernel");

    commArgs.extraFlag = 0;
    Check(TileXRMoonEpPlannerGetWorkspaceSizeV2(comm, 64, 4, 64,
        &workspaceBytes, &capacity) == TileXR::TILEXR_ERROR_NOT_SUPPORT,
        "compat workspace query must reject non-A5 communicators");
    return failures == 0 ? 0 : 1;
}
