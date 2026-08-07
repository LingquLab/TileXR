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
    ok &= Require(plannerCmake, "--cce-aicore-only",
        "Planner AICore-only compile option missing");
    ok &= Require(plannerCmake, "LANGUAGE CCE",
        "Planner kernel is not compiled with the CCE language driver");
    ok &= Require(plannerCmake, "add_library(tilexr_moonep_planner_kernel_obj OBJECT",
        "Planner ASC object target missing");
    ok &= Require(plannerCmake, "-m aicorelinux -Ttext=0 --static",
        "Planner AICore static link missing");
    ok &= Require(plannerCmake, "ld.lld", "Planner AICore linker missing");
    ok &= Require(plannerCmake, "embed_moonep_kernel.cmake",
        "Planner binary embedding step missing");
    ok &= Require(plannerCmake, "TileXRMoonEpPlannerKernelBinary",
        "Planner embedded binary symbol missing");
    ok &= Require(plannerCmake, "$<TARGET_OBJECTS:tilexr_moonep_planner_kernel_obj>",
        "Planner AICore relink dependency missing");
    ok &= Require(plannerCmake, "INSTALL_RPATH \"$ORIGIN\"", "planner RPATH contract missing");
    ok &= Require(plannerCmake, "SOVERSION 2", "Planner SONAME 2 missing");
    ok &= Require(kernel, "SyncAll<true>()", "A5 hardware SyncAll missing");
    ok &= Require(kernel, "BuildLocalHistogram", "planner histogram phase missing");
    ok &= Require(kernel, "BuildExpertLayout", "planner layout phase missing");
    ok &= Require(kernel, "BuildDst", "planner dst phase missing");
    ok &= Require(hostLaunch, "TileXRMoonEpPlannerKernelBinaryData",
        "Planner binary data symbol missing");
    ok &= Require(hostLaunch, "TileXRMoonEpPlannerKernelBinarySize",
        "Planner binary size symbol missing");
    ok &= Require(hostLaunch, "rtDevBinaryRegister",
        "Planner binary registration missing");
    ok &= Require(hostLaunch, "rtDevBinaryUnRegister",
        "Planner failed-registration cleanup missing");
    ok &= Require(hostLaunch, "rtFunctionRegister",
        "Planner function registration missing");
    ok &= Require(hostLaunch, "rtKernelLaunchWithFlagV2",
        "Planner V2 runtime launch missing");
    ok &= Require(hostLaunch, "cfgInfo.schemMode = 1",
        "Planner batch scheduling mode missing");
    ok &= Require(kernel, "WaitReadyFlag", "bounded planner ready poll missing");
    ok &= Require(readyPolling, "DataCacheCleanAndInvalid",
        "Planner ready poll cache invalidation missing");
    ok &= Require(readyPolling, "sync_.GetFlag(readyAddr)",
        "Planner ready poll synchronized peer-memory read missing");
    ok &= Require(kernel, "sortedExpertsScratch", "remote-expert sorting scratch missing");
    ok &= Require(kernel, "DataCopyPad", "internal DMA tail handling missing");
    ok &= Require(layout, "TILEXR_MOONEP_PLANNER_BLOCK_DIM", "blockDim override missing");
    ok &= Require(host, "PeerWindowsReady", "all-rank peer-window validation missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerGetWorkspaceSizeV2",
        "Planner workspace V2 ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerV2", "Planner V2 ABI missing");

    ok &= Reject(kernel, "WaitSyncFlag", "Planner still uses unbounded generic ready wait");
    ok &= Reject(readyPolling, "readyGm.GetValue(0)",
        "Planner ready poll still uses a direct cached GM scalar read");
    ok &= Reject(kernel, "TileXRUDMA", "Planner kernel contains forbidden UDMA call");
    ok &= Reject(kernel, "udma", "Planner kernel contains forbidden lowercase UDMA reference");
    ok &= Reject(host, "UDMA", "Planner Host contains forbidden UDMA dependency");
    ok &= Reject(host, "IsCrossNode", "Planner still rejects cross-node communicators");
    ok &= Reject(kernel, "tokenPadding", "Planner kernel still exposes logical padding");
    ok &= Reject(layout, "tokenPadding", "Planner layout still exposes logical padding");
    ok &= Reject(publicHeader, "tokenPadding", "Planner public ABI still exposes logical padding");
    ok &= Reject(kernel, "zeroFillRanges", "Planner kernel still writes cleanup ranges");
    ok &= Reject(publicHeader, "zeroFillRanges", "Planner public ABI still exposes cleanup ranges");
    ok &= Reject(publicHeader, "TileXRMoonEpPlanner(", "legacy unversioned Planner ABI present");
    ok &= Reject(kernel, "truncated", "Planner kernel contains a truncated migration artifact");
    ok &= Reject(kernel, "<<<", "Planner kernel contains compiler launch syntax");
    ok &= Reject(kernel, "rtKernelLaunch", "Planner kernel directly launches runtime kernels");
    ok &= Reject(plannerCmake, "libtilexr_moonep_planner_kernel.so",
        "Planner still builds a Host kernel shared object");
    ok &= Reject(plannerCmake, "-l:libtilexr_moonep_planner_kernel.so",
        "Planner Host library still links the old kernel shared object");
    ok &= Reject(plannerCmake, "LANGUAGE ASC",
        "Planner kernel still uses the incompatible ASC language driver");
    ok &= Reject(plannerCmake, "devlib",
        "Planner CMake still exposes CANN stub libraries to runtime RPATH");

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
