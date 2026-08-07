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
    const std::string plannerCmake = Read(root + "/src/moonep/planner/CMakeLists.txt");
    const std::string kernelCmake = Read(root + "/src/moonep/cmake/MoonEpKernel.cmake");
    const std::string registration = Read(
        root + "/src/moonep/common/moonep_kernel_registration.cpp");
    const std::string kernel = Read(
        root + "/src/moonep/planner/kernels/tilexr_moonep_planner_kernel.cpp");
    const std::string host = Read(root + "/src/moonep/planner/host/planner_host.cpp");
    const std::string hostLaunch = Read(root + "/src/moonep/planner/host/planner_launch.cpp");
    const std::string layout = Read(root + "/src/moonep/planner/host/planner_layout.cpp");
    const std::string publicHeader = Read(root + "/src/include/tilexr_moonep_planner.h");
    const std::string launcher = Read(root + "/tests/moonep_planner/demo/run_a5.sh");
    const size_t readyBegin = kernel.find("bool WaitReadyFlag(int32_t peer)");
    const size_t readyEnd = kernel.find("void CrossRankReady()", readyBegin);
    const std::string readyPolling = readyBegin == std::string::npos || readyEnd == std::string::npos ?
        std::string() : kernel.substr(readyBegin, readyEnd - readyBegin);

    bool ok = true;
    ok &= Require(cmake, "TILEXR_BUILD_MOONEP_PLANNER", "root planner option missing");
    ok &= Require(plannerCmake, "dav-c310-vec", "A5 compiler target missing");
    ok &= Require(kernelCmake, "--cce-aicore-only",
        "Planner pure AICore compilation missing");
    ok &= Require(kernelCmake, "embed_moonep_kernel.cmake",
        "Planner embedded kernel binary missing");
    ok &= Require(plannerCmake, "INSTALL_RPATH \"$ORIGIN\"", "planner RPATH contract missing");
    ok &= Require(plannerCmake, "SOVERSION 2", "Planner SONAME 2 missing");
    ok &= Require(kernel, "SyncAll<true>()", "A5 hardware SyncAll missing");
    ok &= Require(kernel, "BuildLocalHistogram", "planner histogram phase missing");
    ok &= Require(kernel, "BuildExpertLayout", "planner layout phase missing");
    ok &= Require(kernel, "BuildDst", "planner dst phase missing");
    ok &= Require(registration, "rtDevBinaryRegister",
        "Planner explicit binary registration missing");
    ok &= Require(registration, "rtFunctionRegister",
        "Planner explicit function registration missing");
    ok &= Require(hostLaunch, "rtKernelLaunchWithFlagV2",
        "Planner direct runtime launch missing");
    ok &= Require(hostLaunch, "kPlannerKernelSignature",
        "Planner stable function signature missing");
    ok &= Require(kernel, "WaitReadyFlag", "bounded planner ready poll missing");
    ok &= Require(readyPolling, "DataCacheCleanAndInvalid",
        "Planner ready poll cache invalidation missing");
    ok &= Require(readyPolling, "readyGm.GetValue(0)",
        "Planner ready poll scalar peer-memory read missing");
    ok &= Require(kernel, "sortedExpertsScratch", "remote-expert sorting scratch missing");
    ok &= Require(kernel, "DataCopyPad", "internal DMA tail handling missing");
    ok &= Require(layout, "TILEXR_MOONEP_PLANNER_BLOCK_DIM", "blockDim override missing");
    ok &= Require(host, "PeerWindowsReady", "all-rank peer-window validation missing");
    ok &= Require(host, "commArgs->localRankSize != commArgs->rankSize",
        "Planner layout does not reject cross-node communicators");
    ok &= Require(host, "commArgs.localRankSize != commArgs.rankSize",
        "Planner launch does not reject cross-node communicators");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerGetWorkspaceSizeV2",
        "Planner workspace V2 ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerV2", "Planner V2 ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerGetWorkspaceSizeV3",
        "Planner workspace V3 ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerV3", "Planner V3 ABI missing");
    ok &= Require(publicHeader, "tokenPadding", "Planner public ABI token padding missing");
    ok &= Require(publicHeader, "zeroFillRanges", "Planner public ABI cleanup ranges missing");
    ok &= Require(publicHeader, "dupCounts", "Planner public ABI duplicate counts missing");
    ok &= Require(publicHeader, "nvS", "Planner public ABI padded token count missing");
    ok &= Require(layout, "tokenPadding", "Planner layout token padding missing");
    ok &= Require(kernel, "tokenPadding", "Planner kernel token padding missing");
    ok &= Require(kernel, "zeroFillRanges", "Planner kernel cleanup ranges missing");
    ok &= Require(kernel, "zeroRange.SetValue(0, 0)",
        "Planner kernel zero-fill default start missing");
    ok &= Require(kernel, "zeroRange.SetValue(1, 0)",
        "Planner kernel zero-fill default count missing");
    ok &= Require(kernel, "zeroRange.SetValue(1, end - realEnd)",
        "Planner kernel zero-fill row count missing");
    ok &= Require(kernel, "dupCounts", "Planner kernel duplicate counts missing");
    ok &= Require(kernel, "nvS", "Planner kernel padded token count missing");

    ok &= Reject(kernel, "WaitSyncFlag", "Planner still uses unbounded generic ready wait");
    ok &= Reject(readyPolling, "DataCopy",
        "Planner ready poll still uses MTE data movement");
    ok &= Reject(kernel, "TileXRUDMA", "Planner kernel contains forbidden UDMA call");
    ok &= Reject(kernel, "udma", "Planner kernel contains forbidden lowercase UDMA reference");
    ok &= Reject(host, "UDMA", "Planner Host contains forbidden UDMA dependency");
    ok &= Reject(publicHeader, "TileXRMoonEpPlanner(", "legacy unversioned Planner ABI present");
    ok &= Reject(kernel, "truncated", "Planner kernel contains a truncated migration artifact");
    ok &= Reject(kernel, "zeroRange.SetValue(1, end);",
        "Planner kernel writes padded end instead of zero-fill row count");
    ok &= Reject(hostLaunch, "launch_tilexr_moonep_planner_kernel",
        "Planner Host still calls a Bisheng launch wrapper");
    ok &= Reject(kernel, "launch_tilexr_moonep_planner_kernel",
        "Planner kernel still defines a Bisheng launch wrapper");
    ok &= Reject(kernel, "<<<", "Planner kernel still contains Host launch syntax");
    ok &= Reject(plannerCmake, "libtilexr_moonep_planner_kernel.so",
        "Planner still builds or links a Bisheng Host kernel library");

    ok &= Require(launcher, "rank % physical_device_count",
        "logical-rank to physical-device mapping missing");
    ok &= Require(launcher, "${SCRIPT_DIR}/tilexr_moonep_flow_demo",
        "installed launcher path detection missing");
    ok &= Require(launcher, "TILEXR_MOONEP_PLANNER_BLOCK_DIM",
        "oversubscribed blockDim selection missing");
    ok &= Require(launcher, "oversubscribed=", "oversubscription metadata missing");
    ok &= Require(launcher, "performance_valid=false",
        "oversubscribed performance invalidation missing");

    if (kernel.find("src/ep") != std::string::npos || host.find("src/ep") != std::string::npos ||
        plannerCmake.find("src/ep") != std::string::npos) {
        std::cerr << "planner depends on forbidden src/ep path" << std::endl;
        ok = false;
    }
    return ok ? 0 : 1;
}
