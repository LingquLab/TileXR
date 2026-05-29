#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

void CheckContains(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos == std::string::npos) {
        std::cerr << "expected " << path << " to contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckDoesNotContain(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "expected " << path << " not to contain: " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void TestCollectivesHeaderDeclaresPublicApis()
{
    const std::string path = "src/include/tilexr_collectives.h";
    const auto text = ReadFile(path);
    CheckContains(path, text, "#ifdef __cplusplus");
    CheckContains(path, text, "extern \"C\"");
    CheckContains(path, text, "int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,");
    CheckContains(path, text, "int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,");
}

void TestCoreApiHeaderDoesNotDeclareCollectives()
{
    const std::string path = "src/include/tilexr_api.h";
    const auto text = ReadFile(path);
    CheckDoesNotContain(path, text, "TileXRAllGather");
    CheckDoesNotContain(path, text, "TileXRAllToAll");
}

void TestCommBuildDoesNotReferenceCollectives()
{
    const std::string path = "src/comm/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckDoesNotContain(path, text, "src/collectives");
    CheckDoesNotContain(path, text, "tilexr_collectives");
}

void TestCollectivesBuildDefinesSeparateSharedLibrary()
{
    const std::string path = "src/collectives/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "add_library(tilexr-collectives SHARED");
    CheckContains(path, text, "target_link_libraries(tilexr-collectives\n        PRIVATE");
    CheckContains(path, text, "target_link_directories(tilexr-collectives\n        PRIVATE");
    CheckDoesNotContain(path, text, "target_link_libraries(tilexr-collectives tile-comm");
    CheckContains(path, text, "${CMAKE_SOURCE_DIR}/src/include/tilexr_collectives.h");
    CheckContains(path, text, "${CMAKE_SOURCE_DIR}/src/include/tilexr_api.h");
    CheckContains(path, text, "${CMAKE_SOURCE_DIR}/src/include/tilexr_types.h");
    CheckContains(path, text, "${CMAKE_SOURCE_DIR}/src/include/comm_args.h");
    CheckContains(path, text, "DESTINATION ${CMAKE_INSTALL_PREFIX}/include");
}

void TestCollectivesTestBuildUsesExplicitLibraryHint()
{
    const std::string path = "tests/collectives/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "set(TILEXR_INSTALL_PREFIX \"${TILEXR_ROOT}/install\" CACHE PATH");
    CheckContains(path, text, "${TILEXR_INSTALL_PREFIX}/include");
    CheckContains(path, text, "${TILEXR_INSTALL_PREFIX}/include/tilexr_collectives.h");
    CheckContains(path, text, "set(TILEXR_COLLECTIVES_LIB \"\" CACHE FILEPATH");
    CheckContains(path, text, "message(FATAL_ERROR");
    CheckDoesNotContain(path, text, "${TILEXR_ROOT}/src/include");
    CheckDoesNotContain(path, text, "/tmp/tilexr-install-split-collectives");
    CheckDoesNotContain(path, text, "/tmp/tilexr-build-split-collectives");
}

} // namespace

int main()
{
    TestCollectivesHeaderDeclaresPublicApis();
    TestCoreApiHeaderDoesNotDeclareCollectives();
    TestCommBuildDoesNotReferenceCollectives();
    TestCollectivesBuildDefinesSeparateSharedLibrary();
    TestCollectivesTestBuildUsesExplicitLibraryHint();
    if (g_failures != 0) {
        std::cerr << g_failures << " collectives API split checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR collectives API split checks passed" << std::endl;
    return 0;
}
