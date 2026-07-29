#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string& path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string& path, bool required = true)
{
    std::ifstream input(RepoPath(path).c_str());
    if (!input.is_open()) {
        if (required) {
            std::cerr << "failed to open " << RepoPath(path) << std::endl;
            ++g_failures;
        }
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckNoNeedle(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "unexpected dependency in " << path << ": " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void CheckNeedle(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected text not found in " << path << ": " << needle << std::endl;
        ++g_failures;
    }
}

void TestCommSourcesDoNotUseShmem()
{
    const std::vector<std::string> paths = {
        "src/comm/CMakeLists.txt",
        "src/comm/tilexr_comm.cpp",
        "src/comm/comm_wrap.cpp",
        "src/comm/tilexr_comm.h",
        "src/comm/sdma/tilexr_sdma_transport.cpp",
        "src/comm/sdma/tilexr_sdma_transport.h",
    };
    const std::vector<std::string> forbidden = {
        "shmem",
        "shmem.h",
        "libshmem",
        "aclshmem",
        "ACLSHMEM",
    };
    for (const auto& path : paths) {
        const auto text = ReadFile(path);
        for (const auto& needle : forbidden) {
            CheckNoNeedle(path, text, needle);
        }
    }
}

void TestOnlyCompatIncludesSdmaIntrinsics()
{
    const std::vector<std::string> paths = {
        "src/include/tilexr_sdma.h",
        "src/include/tilexr_sdma_compat.h",
        "src/include/tilexr_sdma_types.h",
        "src/include/comm_args.h",
        "src/comm/tilexr_comm.cpp",
        "src/comm/tilexr_comm.h",
        "src/comm/comm_wrap.cpp",
        "src/comm/sdma/tilexr_sdma_transport.cpp",
        "src/comm/sdma/tilexr_sdma_transport.h",
    };
    const std::string ptoPrefix = "pto/npu/comm/async/sdma/";
    const std::string intrinHeader = ptoPrefix + "sdma_async_intrin.hpp";
    const std::string workspaceHeader = ptoPrefix + "sdma_workspace_manager.hpp";
    for (const auto& path : paths) {
        const bool required = path != "src/include/tilexr_sdma.h";
        const auto text = ReadFile(path, required);
        if (path == "src/include/tilexr_sdma_compat.h") {
            CheckNeedle(path, text, intrinHeader);
            CheckNoNeedle(path, text, workspaceHeader);
            continue;
        }
        if (path == "src/comm/sdma/tilexr_sdma_transport.cpp") {
            CheckNeedle(path, text, workspaceHeader);
            CheckNoNeedle(path, text, intrinHeader);
            continue;
        }
        CheckNoNeedle(path, text, ptoPrefix);
    }
}

void TestBuildSelectsExplicitSoc()
{
    const std::string buildPath = "tests/sdma/build.sh";
    const auto buildText = ReadFile(buildPath);
    CheckNeedle(buildPath, buildText,
        "SDMA_SOC_TYPE=\"${2:-${TILEXR_SDMA_DEMO_SOC_TYPE:-Ascend910B}}\"");
    CheckNeedle(buildPath, buildText,
        "-DTILEXR_SDMA_DEMO_SOC_TYPE=\"${SDMA_SOC_TYPE}\"");

    const std::string cmakePath = "tests/sdma/CMakeLists.txt";
    const auto cmakeText = ReadFile(cmakePath);
    CheckNeedle(cmakePath, cmakeText,
        "TILEXR_SDMA_DEMO_SOC_TYPE STREQUAL \"Ascend950\"");
    CheckNeedle(cmakePath, cmakeText,
        "TILEXR_SDMA_NPU_ARCH \"dav-3510\"");
    CheckNeedle(cmakePath, cmakeText,
        "TILEXR_SDMA_AICORE_ARCH \"--cce-aicore-arch=dav-c310-vec\"");
    CheckNeedle(cmakePath, cmakeText,
        "Unsupported TILEXR_SDMA_DEMO_SOC_TYPE=");
}

void TestAscend950UsesOwnedDirectBackend()
{
    const std::string transportPath = "src/comm/sdma/tilexr_sdma_transport.cpp";
    const auto transport = ReadFile(transportPath);
    CheckNeedle(transportPath, transport, "TileXRA5SDMABackend");
    CheckNeedle(transportPath, transport, "ClassifySDMABackend(socName)");

    const std::string backendPath = "src/comm/sdma/tilexr_sdma_a5_backend.cpp";
    const auto backend = ReadFile(backendPath);
    CheckNeedle(backendPath, backend, "aclnnShmemSdmaStarsQuery");
    CheckNeedle(backendPath, backend, "RES_ADDR_TYPE_STARS_RTSQ");
    CheckNeedle(backendPath, backend, "TILEXR_SDMA_A5_CHANNEL_COUNT");
    CheckNeedle(backendPath, backend, "kExpectedAicpuQueryFailure");
    CheckNeedle(backendPath, backend,
        "LoadSymbol(RTLD_DEFAULT, \"halResAddrMap\", mapResource)");
    CheckNeedle(backendPath, backend, "pendingQueries");
    CheckNoNeedle(backendPath, backend, "halResAddrMap(");
    CheckNoNeedle(backendPath, backend, "halResAddrUnmap(");

    const std::string devicePath = "src/include/tilexr_sdma_a5.h";
    const auto device = ReadFile(devicePath);
    CheckNeedle(devicePath, device, "while (A5SdmaReadCompletion(completion) != generation)");

    const std::string demoPath = "tests/sdma/demo/tilexr_sdma_demo.cpp";
    const auto demo = ReadFile(demoPath);
    CheckNeedle(demoPath, demo, "aclrtSynchronizeStream(stream)");
}

void TestSdmaFailureIsScopedToOneCommunicator()
{
    const std::string commPath = "src/comm/tilexr_comm.cpp";
    const auto comm = ReadFile(commPath);
    CheckNoNeedle(commPath, comm, "g_sdmaUnavailable");
    CheckNeedle(commPath, comm, "sdmaTransport_->Init(options)");
    CheckNeedle(commPath, comm, "bool TileXRComm::PrepareDestroy()");
    CheckNeedle(commPath, comm, "if (UpdateCommArgsDev() != TILEXR_SUCCESS)");

    const std::string wrapPath = "src/comm/comm_wrap.cpp";
    const auto wrap = ReadFile(wrapPath);
    CheckNeedle(wrapPath, wrap, "if (!c->PrepareDestroy())");
    CheckNeedle(wrapPath, wrap, "return TILEXR_ERROR_INTERNAL;");
}

} // namespace

int main()
{
    TestCommSourcesDoNotUseShmem();
    TestOnlyCompatIncludesSdmaIntrinsics();
    TestBuildSelectsExplicitSoc();
    TestAscend950UsesOwnedDirectBackend();
    TestSdmaFailureIsScopedToOneCommunicator();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA source guard checks passed" << std::endl;
    return 0;
}
