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
    const auto compat = ReadFile("src/include/tilexr_sdma_compat.h");
    CheckNeedle("src/include/tilexr_sdma_compat.h", compat,
                "pto/npu/comm/async/sdma/sdma_async_intrin.hpp");
    const auto transport = ReadFile("src/comm/sdma/tilexr_sdma_transport.cpp");
    CheckNeedle("src/comm/sdma/tilexr_sdma_transport.cpp", transport,
                "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp");
    const std::vector<std::string> disallowed = {
        "src/include/tilexr_sdma.h",
        "src/include/comm_args.h",
        "src/comm/tilexr_comm.cpp",
    };
    for (const auto& path : disallowed) {
        const bool required = path != "src/include/tilexr_sdma.h";
        const auto text = ReadFile(path, required);
        CheckNoNeedle(path, text, "pto/npu/comm/async/sdma/");
    }
}

} // namespace

int main()
{
    TestCommSourcesDoNotUseShmem();
    TestOnlyCompatIncludesSdmaIntrinsics();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA source guard checks passed" << std::endl;
    return 0;
}
