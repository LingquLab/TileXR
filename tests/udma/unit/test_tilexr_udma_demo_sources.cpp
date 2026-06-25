#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string &path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string &path)
{
    const std::string fullPath = RepoPath(path);
    std::ifstream input(fullPath.c_str());
    if (!input) {
        std::cerr << "failed to open " << fullPath << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string &path, const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << path << " missing: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string &path, const std::string &text, const std::string &needle)
{
    if (text.find(needle) != std::string::npos) {
        std::cerr << path << " should not contain: " << needle << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    const std::string demoPath = "tests/udma/demo/tilexr_udma_demo.cpp";
    const std::string scriptPath = "tests/udma/demo/run_tilexr_udma_data_channel_probe_mpi.sh";
    const std::string hccpDefsPath = "src/comm/udma/tilexr_hccp_defs.h";
    const std::string demo = ReadFile(demoPath);
    const std::string script = ReadFile(scriptPath);
    const std::string hccpDefs = ReadFile(hccpDefsPath);

    CheckContains(demoPath, demo, "TILEXR_DEMO_BARRIER_ADDR");
    CheckContains(demoPath, demo, "INADDR_ANY");
    CheckContains(demoPath, demo, "TileXRSDMAAvailable");
    CheckContains(demoPath, demo, "TileXRGetSDMAWorkspaceDev");
    CheckContains(demoPath, demo, "TILEXR_DEMO_REQUIRE_SDMA");
    CheckContains(demoPath, demo, "launch_tilexr_udma_slot_signal_get_probe");
    CheckContains(demoPath, demo, "testType == 2");

    const std::string kernelPath = "tests/udma/demo/tilexr_udma_demo_kernel.cpp";
    const std::string kernel = ReadFile(kernelPath);
    CheckContains(kernelPath, kernel, "tilexr_udma_slot_signal_get_probe_kernel");
    CheckContains(kernelPath, kernel, "UDMAPutSignalNbi<int32_t>");
    CheckContains(kernelPath, kernel, "UDMAGetNbi<int32_t>");

    CheckContains(scriptPath, script, "mpirun");
    CheckContains(scriptPath, script, "--hosts");
    CheckContains(scriptPath, script, "TILEXR_COMM_ID");
    CheckContains(scriptPath, script, "TILEXR_DEMO_BARRIER_ADDR");
    CheckContains(scriptPath, script, "--require-sdma");

    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_LOCAL_ONLY = 1");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_READ = (1 << 1)");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_WRITE = (1 << 2)");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_ATOMIC = (1 << 3)");
    CheckContains(hccpDefsPath, hccpDefs,
        "MEM_SEG_ACCESS_DEFAULT = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC");

    const std::string transportPath = "src/comm/udma/tilexr_udma_transport.cpp";
    const std::string transport = ReadFile(transportPath);
    CheckContains(transportPath, transport, "TILEXR_UDMA_TOPO_ROUTE");
    CheckContains(transportPath, transport, "const bool useTopoRoutes");
    CheckContains(transportPath, transport, "ResolveLocalEidRoute(rootInfo, topoEdges, localId, allLocalIds[peer], localEid)");
    CheckContains(transportPath, transport, "queuesImported_");
    CheckContains(transportPath, transport, "if (!queuesImported_) {\n        ret = ImportQueues();");
    CheckContains(transportPath, transport, "queuesImported_ = true;\n    }\n    ret = ExchangeAndImportMemory();");
    CheckNotContains(transportPath, transport,
        "ret = ImportQueues();\n    if (ret != TILEXR_SUCCESS) {\n        Shutdown();\n        return ret;\n    }\n    ret = RefreshUDMAInfo();");
    CheckContains(transportPath, transport, "mrInfo.in.ub.flags.bs.cacheable = 0");
    CheckContains(transportPath, transport, "mrInfo.in.ub.flags.bs.nonPin = 0");
    CheckContains(transportPath, transport, "mrInfo.in.ub.flags.bs.userIova = 0");

    if (g_failures == 0) {
        std::cout << "TileXR UDMA demo source checks passed" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
