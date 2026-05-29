#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

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
        std::cerr << "unexpected shmem dependency in " << path << ": " << needle
                  << " at byte " << pos << std::endl;
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
    };
    const std::vector<std::string> forbidden = {
        "shmem",
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

} // namespace

int main()
{
    TestCommSourcesDoNotUseShmem();
    if (g_failures != 0) {
        std::cerr << g_failures << " no-shmem dependency checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR comm sources have no shmem dependency" << std::endl;
    return 0;
}
