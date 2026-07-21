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
    const std::string hccpDefsPath = "src/comm/udma/tilexr_hccp_defs.h";
    const std::string demo = ReadFile(demoPath);
    const std::string hccpDefs = ReadFile(hccpDefsPath);
    CheckContains(demoPath, demo, "TileXRUDMARegister");
    CheckContains(demoPath, demo, "TileXRUDMAUnregister");
    CheckContains(demoPath, demo, "ExtraFlag::UDMA");
    CheckContains(demoPath, demo, "launch_tilexr_udma_all_gather");
    CheckContains(demoPath, demo, "launch_tilexr_udma_put_signal");
    CheckContains(demoPath, demo, "DemoBarrierAll");
    CheckNotContains(demoPath, demo, "aclshmem");
    CheckNotContains(demoPath, demo, "shmem_");

    const std::string kernelPath = "tests/udma/demo/tilexr_udma_demo_kernel.cpp";
    const std::string kernel = ReadFile(kernelPath);
    CheckContains(kernelPath, kernel, "UDMARegistryEnabled");
    CheckContains(kernelPath, kernel, "UDMAPutNbi<int32_t>");
    CheckContains(kernelPath, kernel, "UDMAPutSignalNbi<int32_t>");
    CheckContains(kernelPath, kernel, "UDMAQuiet");
    CheckNotContains(kernelPath, kernel, "aclshmem");
    CheckNotContains(kernelPath, kernel, "shmem_");

    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_LOCAL_ONLY = 1");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_READ = (1 << 1)");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_WRITE = (1 << 2)");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_ATOMIC = (1 << 3)");
    CheckContains(hccpDefsPath, hccpDefs,
        "MEM_SEG_ACCESS_DEFAULT = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC");

    const std::string transportPath = "src/comm/udma/tilexr_udma_transport.cpp";
    const std::string transport = ReadFile(transportPath);
    CheckContains(transportPath, transport, "ResolveLocalEidRoute(rootInfo, topoEdges, localId, allLocalIds[peer], localEid)");

    if (g_failures == 0) {
        std::cout << "TileXR UDMA demo source checks passed" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
