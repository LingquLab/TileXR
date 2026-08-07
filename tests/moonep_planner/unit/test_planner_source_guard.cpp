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
    const std::string plannerCmake = Read(root + "/src/moonep/planner_v2/CMakeLists.txt");
    const std::string kernel = Read(root + "/src/moonep/planner_v2/kernels/tilexr_moonep_planner_kernel.cpp");
    const std::string host = Read(root + "/src/moonep/planner_v2/host/planner_host.cpp");
    const std::string entry = Read(root + "/src/moonep/planner_v2/host/tilexr_moonep_planner.cpp");
    const std::string planHeader = Read(root + "/src/include/tilexr_ep_plan.h");
    const std::string layout = Read(root + "/src/moonep/planner_v2/host/planner_layout.cpp");
    const std::string publicHeader = Read(root + "/src/include/tilexr_moonep_planner.h");

    bool ok = true;
    ok &= Require(cmake, "TILEXR_BUILD_MOONEP_PLANNER", "root planner option missing");
    ok &= Require(plannerCmake, "tilexr_moonep_planner_kernel.cpp", "planner_v2 kernel source missing");
    ok &= Require(plannerCmake, "INSTALL_RPATH \"$ORIGIN\"", "planner RPATH contract missing");
    ok &= Require(kernel, "RunPlanAlgorithm", "optimized planner algorithm missing");
    ok &= Require(kernel, "PublishInputs", "cross-rank input publication missing");
    ok &= Require(kernel, "GatherInputs", "cross-rank input gather missing");
    ok &= Require(kernel, "CollectiveBarrier", "bounded collective barrier missing");
    ok &= Require(kernel, "IPC_DATA_OFFSET", "peer-memory payload offset missing");
    ok &= Require(kernel, "ReduceGlobalPlanStatus", "global status reduction missing");
    ok &= Require(kernel, "PublishBarrierFlags", "memory-semantics peer mailbox publish missing");
    ok &= Require(kernel, "peerMems", "memory-semantics peerMems access missing");
    ok &= Reject(kernel, "TileXRUDMA", "Planner kernel contains forbidden UDMA call");
    ok &= Reject(kernel, "UDMAPut", "Planner kernel contains forbidden UDMA put");
    ok &= Reject(kernel, "UDMAQuiet", "Planner kernel contains forbidden UDMA quiet");
    ok &= Reject(host, "UDMA", "Planner Host contains forbidden UDMA dependency");
    ok &= Reject(entry, "UDMA", "Planner entry contains forbidden UDMA dependency");
    ok &= Reject(kernel, "zeroFillRanges", "Planner kernel still writes cleanup ranges");
    ok &= Reject(planHeader, "zeroFillRanges", "optimized Planner ABI still exposes cleanup ranges");
    ok &= Require(entry, "TileXRMoeEpPlanV2", "optimized Planner V2 wrapper missing");
    ok &= Require(entry, "TileXRMoonEpPlannerV2", "PR91-compatible Planner V2 wrapper missing");
    ok &= Require(layout, "TILEXR_MOONEP_PLANNER_BLOCK_DIM", "PR91 blockDim override missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerGetWorkspaceSizeV2", "Planner workspace V2 ABI missing");
    ok &= Require(publicHeader, "TileXRMoonEpPlannerV2", "Planner V2 ABI missing");
    ok &= Reject(kernel, "src/ep/planner", "stale src/ep/planner dependency present");
    ok &= Reject(host, "src/ep/planner", "stale src/ep/planner dependency present");
    ok &= Reject(entry, "src/ep/planner", "stale src/ep/planner dependency present");
    ok &= Reject(publicHeader, "TileXRMoonEpPlanner(", "legacy unversioned Planner ABI present");
    return ok ? 0 : 1;
}
