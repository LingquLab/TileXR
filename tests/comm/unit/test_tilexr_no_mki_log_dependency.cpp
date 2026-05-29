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

std::string ReadFile(const std::string& path)
{
    const std::string fullPath = RepoPath(path);
    std::ifstream input(fullPath.c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << fullPath << std::endl;
        ++g_failures;
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
        std::cerr << "unexpected MKI log dependency in " << path << ": " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void TestCommSourcesDoNotUseMkiLog()
{
    const std::vector<std::string> paths = {
        "src/comm/tilexr_log.h",
        "src/comm/tilexr_comm.cpp",
        "src/comm/comm_wrap.cpp",
        "src/comm/tilexr_internal.cpp",
        "src/comm/tools/socket/tilexr_sock_exchange.h",
        "src/comm/tools/socket/tilexr_sock_exchange.cpp",
        "src/comm/udma/tilexr_hccp_loader.cpp",
        "src/comm/udma/tilexr_udma_transport.cpp",
    };
    const std::vector<std::string> forbidden = {
        "mki/utils/log/log.h",
        "MKI_LOG",
    };
    for (const auto& path : paths) {
        const auto text = ReadFile(path);
        for (const auto& needle : forbidden) {
            CheckNoNeedle(path, text, needle);
        }
    }
}

} // namespace

int main()
{
    TestCommSourcesDoNotUseMkiLog();
    if (g_failures != 0) {
        std::cerr << g_failures << " MKI log dependency checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR comm sources have no MKI log dependency" << std::endl;
    return 0;
}
