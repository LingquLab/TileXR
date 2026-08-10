#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef TILEXR_SOURCE_ROOT
#error TILEXR_SOURCE_ROOT must be defined
#endif

namespace {

std::string Read(const std::string &path)
{
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

bool Require(const std::string &text, const std::string &needle, const char *message)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << message << std::endl;
        return false;
    }
    return true;
}

bool Reject(const std::string &text, const std::string &needle, const char *message)
{
    if (text.find(needle) != std::string::npos) {
        std::cerr << message << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main()
{
    const std::string root = TILEXR_SOURCE_ROOT;
    const std::string cmake = Read(root + "/CMakeLists.txt");
    const std::string moonepCmake = Read(root + "/src/moonep/CMakeLists.txt");
    const std::string plannerCmake = Read(root + "/src/moonep/planner_v2/CMakeLists.txt");
    const std::string plannerV3Cmake = Read(root + "/src/moonep/planner/CMakeLists.txt");
    const std::string kernelCmake = Read(root + "/src/moonep/cmake/MoonEpKernel.cmake");
    const std::string registration = Read(
        root + "/src/moonep/common/moonep_kernel_registration.cpp");
    const std::string kernelLaunch = Read(
        root + "/src/moonep/common/moonep_kernel_launch.h");
    const std::string kernel = Read(
        root + "/src/moonep/planner_v2/kernels/tilexr_moonep_planner_kernel.cpp");
    const std::string host = Read(root + "/src/moonep/planner_v2/host/planner_host.cpp");
    const std::string hostLaunch = Read(root + "/src/moonep/planner_v2/host/planner_launch.cpp");
    const std::string entry = Read(
        root + "/src/moonep/planner_v2/host/tilexr_moonep_planner.cpp");
    const std::string layout = Read(root + "/src/moonep/planner_v2/host/planner_layout.cpp");
    const std::string publicHeader = Read(root + "/src/include/tilexr_moonep_planner.h");
    const std::string planHeader = Read(root + "/src/include/tilexr_ep_plan.h");
    const std::string v3Entry = Read(
        root + "/src/moonep/planner/host/tilexr_moonep_planner.cpp");
    const std::string v3Layout = Read(
        root + "/src/moonep/planner/host/planner_layout.cpp");
    const std::string v3Kernel = Read(
        root + "/src/moonep/planner/kernels/tilexr_moonep_planner_kernel.cpp");
    const std::string moonEpHost = Read(root + "/src/moonep/host/tilexr_moonep.cpp");

    bool ok = true;
    ok &= Require(cmake, "TILEXR_BUILD_MOONEP_PLANNER", "root planner option missing");
    const size_t plannerV2Position = moonepCmake.find("add_subdirectory(planner_v2)");
    const size_t plannerV3Position = moonepCmake.find("add_subdirectory(planner)");
    if (plannerV2Position == std::string::npos || plannerV3Position == std::string::npos ||
        plannerV2Position >= plannerV3Position) {
        std::cerr << "Planner V2 must be configured before the V3 compatibility backend" << std::endl;
        ok = false;
    }
    ok &= Require(plannerCmake, "add_library(tilexr-moonep-planner SHARED",
        "PR96 Planner V2 does not own shared-library creation");
    ok &= Require(plannerV3Cmake, "add_library(tilexr-moonep-planner-v3-objects OBJECT",
        "PR103 Planner V3 is not an object-only compatibility backend");
    ok &= Require(plannerV3Cmake, "$<TARGET_OBJECTS:tilexr-moonep-planner-v3-objects>",
        "PR103 Planner V3 objects are not appended to the V2-owned target");
    ok &= Require(plannerCmake, "dav-c310-vec", "A5 compiler target missing");
    ok &= Require(plannerCmake, "tilexr_moonep_planner_kernel.cpp",
        "planner_v2 kernel source missing");
    ok &= Require(kernelCmake, "--cce-aicore-only",
        "Planner pure AICore compilation missing");
    ok &= Require(kernelCmake, "embed_moonep_kernel.cmake",
        "Planner embedded kernel binary missing");
    ok &= Require(plannerCmake, "INSTALL_RPATH \"$ORIGIN\"", "planner RPATH contract missing");
    ok &= Require(plannerCmake, "SOVERSION 2", "Planner SONAME 2 missing");
    ok &= Require(kernel, "RunPlanAlgorithm", "optimized planner algorithm missing");
    ok &= Require(kernel, "PublishInputs", "cross-rank input publication missing");
    ok &= Require(kernel, "GatherInputs", "cross-rank input gather missing");
    ok &= Require(kernel, "CollectiveBarrier", "bounded collective barrier missing");
    ok &= Require(kernel, "IPC_DATA_OFFSET", "peer-memory payload offset missing");
    ok &= Require(kernel, "ReduceGlobalPlanStatus", "global status reduction missing");
    ok &= Require(kernel, "PublishBarrierFlags", "peer mailbox publish missing");
    ok &= Require(kernel, "peerMems", "peerMems access missing");
    ok &= Require(registration, "rtDevBinaryRegister",
        "Planner explicit binary registration missing");
    ok &= Require(registration, "rtFunctionRegister",
        "Planner explicit function registration missing");
    ok &= Require(kernelLaunch, "rtKernelLaunchWithFlagV2",
        "Planner Runtime V2 launch helper missing");
    ok &= Require(hostLaunch, "LaunchRegisteredMoonEpKernel",
        "Planner V2 does not use the registered Runtime V2 launch path");
    ok &= Require(hostLaunch, "kPlannerV2KernelSignature",
        "Planner V2 stable function signature missing");
    ok &= Require(layout, "TILEXR_MOONEP_PLANNER_BLOCK_DIM", "PR96 blockDim override missing");
    ok &= Require(host, "PeerWindowsReady", "all-rank peer-window validation missing");
    ok &= Reject(host, "commArgs->localRankSize != commArgs->rankSize",
        "Planner layout still rejects cross-node communicators");
    ok &= Reject(host, "commArgs.localRankSize != commArgs.rankSize",
        "Planner launch still rejects cross-node communicators");
    ok &= Require(entry, "TileXRMoeEpPlanV2", "optimized Planner V2 wrapper missing");
    ok &= Require(entry, "TileXRMoonEpPlannerV2", "historical Planner V2 wrapper missing");
    ok &= Require(planHeader, "TileXRMoeEpPlanV2WithMetadata",
        "optimized Planner metadata ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerGetWorkspaceSizeV2",
        "Planner workspace V2 ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerV2", "Planner V2 ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerGetWorkspaceSizeV3",
        "Planner workspace V3 ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerV3", "Planner V3 ABI missing");
    ok &= Require(v3Entry, "TileXRMoonEpV3::PlannerLayout",
        "Planner V3 internals are not isolated from PR96 V2");
    ok &= Reject(v3Layout, "selected < rankSize",
        "Planner V3 still requires one AIV block per rank");
    ok &= Require(v3Kernel, "sourceRankValue += blockCount_",
        "Planner V3 gather does not stride over ranks");
    ok &= Require(v3Kernel, "ownerValue += blockCount_",
        "Planner V3 allocation does not stride over ranks");
    ok &= Require(v3Kernel, "destValue += blockCount_",
        "Planner V3 expert layout does not stride over ranks");
    ok &= Require(v3Kernel,
        "PublishCrossRankReady();\n        AscendC::SyncAll<true>();\n\n        CrossRankReady();",
        "Planner V3 does not publish readiness before its local waiters start");
    ok &= Require(v3Kernel, "peerOffset = blockIdx_; peerOffset < rankSize_",
        "Planner V3 readiness wait is not partitioned across AIV blocks");
    ok &= Require(v3Kernel, "peerOffset += blockCount_",
        "Planner V3 readiness wait does not stride over peers");
    ok &= Require(v3Kernel,
        "CrossRankReady();\n        AscendC::SyncAll<true>();\n        FinalizeCrossRankReady();",
        "Planner V3 does not reduce per-block readiness results");
    ok &= Require(v3Kernel, "groupTotalsGm_[blockIdx_]",
        "Planner V3 readiness waiters do not report through block-local slots");
    ok &= Require(moonEpHost, "TileXRMoonEpPlannerV3",
        "MoonEP V1 no longer routes planning through the V3 compatibility backend");

    ok &= Reject(kernel, "TileXRUDMA", "Planner kernel contains forbidden UDMA call");
    ok &= Reject(kernel, "udma", "Planner kernel contains forbidden lowercase UDMA reference");
    ok &= Reject(host, "UDMA", "Planner Host contains forbidden UDMA dependency");
    ok &= Reject(entry, "UDMA", "Planner entry contains forbidden UDMA dependency");
    ok &= Reject(kernel, "zeroFillRanges", "Planner V2 kernel writes V3 cleanup ranges");
    ok &= Reject(planHeader, "zeroFillRanges", "Planner V2 ABI exposes V3 cleanup ranges");
    ok &= Reject(publicHeader, "TileXRMoonEpPlanner(", "legacy unversioned Planner ABI present");
    ok &= Reject(hostLaunch, "launch_tilexr_moonep_planner_kernel",
        "Planner Host still calls a Bisheng launch wrapper");
    ok &= Reject(kernel, "launch_tilexr_moonep_planner_kernel",
        "Planner kernel still defines a Bisheng launch wrapper");
    ok &= Reject(kernel, "<<<", "Planner kernel still contains Host launch syntax");
    ok &= Reject(hostLaunch, "<<<", "Planner Host still contains compiler launch syntax");
    ok &= Reject(plannerCmake, "libtilexr_moonep_planner_kernel.so",
        "Planner still builds or links a Bisheng Host kernel library");
    ok &= Reject(plannerCmake, "/devlib", "Planner V2 links the toolkit stub devlib");
    ok &= Reject(plannerV3Cmake, "/devlib", "Planner V3 links the toolkit stub devlib");
    ok &= Reject(plannerV3Cmake, "add_library(tilexr-moonep-planner SHARED",
        "Planner V3 still creates the shared library");
    ok &= Reject(plannerV3Cmake, "install(TARGETS tilexr-moonep-planner",
        "Planner V3 still owns shared-library installation");

    if (kernel.find("src/ep") != std::string::npos || host.find("src/ep") != std::string::npos ||
        entry.find("src/ep") != std::string::npos || plannerCmake.find("src/ep") != std::string::npos) {
        std::cerr << "planner depends on forbidden src/ep path" << std::endl;
        ok = false;
    }
    return ok ? 0 : 1;
}
