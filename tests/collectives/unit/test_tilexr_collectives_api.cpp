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

void TestCommBuildInstallsPublicHeadersAndKeepsLinksPrivate()
{
    const std::string path = "src/comm/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "include(GNUInstallDirs)");
    CheckContains(path, text, "target_link_directories(tile-comm\n        PRIVATE");
    CheckContains(path, text, "target_link_libraries(tile-comm\n        PRIVATE");
    CheckDoesNotContain(path, text, "target_link_libraries(tile-comm ascendcl");
    CheckContains(path, text, "LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}");
    CheckContains(path, text, "DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}");
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_api.h");
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_types.h");
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/comm_args.h");
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_sync.h");
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_udma.h");
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_udma_reg.h");
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_udma_types.h");
    CheckDoesNotContain(path, text, "${CMAKE_SOURCE_DIR}/src/include/");
    CheckDoesNotContain(path, text, "${CMAKE_SOURCE_DIR}/src/include/tilexr_collectives.h");
    CheckDoesNotContain(path, text, "FILES_MATCHING PATTERN \"*.h\"");
}

void TestCollectivesBuildDefinesSeparateSharedLibrary()
{
    const std::string path = "src/collectives/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "include(GNUInstallDirs)");
    CheckContains(path, text, "add_library(tilexr-collectives SHARED");
    CheckContains(path, text, "${ASCEND_DRIVER_PATH}/kernel/inc\n        PRIVATE");
    CheckContains(path, text, "target_link_libraries(tilexr-collectives\n        PUBLIC\n        tile-comm\n        PRIVATE");
    CheckContains(path, text, "target_link_directories(tilexr-collectives\n        PRIVATE");
    CheckDoesNotContain(path, text, "target_link_libraries(tilexr-collectives tile-comm");
    CheckContains(path, text, "LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}");
    CheckContains(path, text, "${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_collectives.h");
    CheckContains(path, text, "DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}");
    CheckDoesNotContain(path, text, "${CMAKE_SOURCE_DIR}/src/include/");
    CheckDoesNotContain(path, text, "tilexr_api.h");
    CheckDoesNotContain(path, text, "comm_args.h");
    CheckDoesNotContain(path, text, "${CMAKE_INSTALL_PREFIX}/lib");
    CheckDoesNotContain(path, text, "${CMAKE_INSTALL_PREFIX}/include");
}

void TestCollectivesTestBuildUsesExplicitLibraryHint()
{
    const std::string path = "tests/collectives/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "include(GNUInstallDirs)");
    CheckContains(path, text, "enable_testing()");
    CheckContains(path, text, "set(TILEXR_INSTALL_PREFIX \"${TILEXR_ROOT}/install\" CACHE PATH");
    CheckContains(path, text, "set(TILEXR_INSTALL_LIBDIR \"${CMAKE_INSTALL_LIBDIR}\" CACHE STRING");
    CheckContains(path, text, "if(IS_ABSOLUTE \"${TILEXR_INSTALL_LIBDIR}\")");
    CheckContains(path, text, "set(TILEXR_INSTALL_LIB_SEARCH_DIR \"${TILEXR_INSTALL_LIBDIR}\")");
    CheckContains(path, text, "set(TILEXR_INSTALL_LIB_SEARCH_DIR \"${TILEXR_INSTALL_PREFIX}/${TILEXR_INSTALL_LIBDIR}\")");
    CheckContains(path, text, "${TILEXR_INSTALL_PREFIX}/include");
    CheckContains(path, text, "${TILEXR_INSTALL_PREFIX}/include/tilexr_collectives.h");
    CheckContains(path, text, "set(TILEXR_COLLECTIVES_LIB \"\" CACHE FILEPATH");
    CheckContains(path, text, "set(TILEXR_LIB \"\" CACHE FILEPATH");
    CheckContains(path, text, "${TILEXR_INSTALL_LIB_SEARCH_DIR}");
    CheckContains(path, text, "add_library(tilexr-comm-installed SHARED IMPORTED)");
    CheckContains(path, text, "add_library(tilexr-collectives-installed SHARED IMPORTED)");
    CheckContains(path, text, "IMPORTED_LOCATION \"${TILEXR_LIB}\"");
    CheckContains(path, text, "IMPORTED_LOCATION \"${TILEXR_COLLECTIVES_LIB}\"");
    CheckContains(path, text, "INTERFACE_LINK_LIBRARIES tilexr-comm-installed");
    CheckContains(path, text, "target_link_libraries(test_tilexr_collectives_header_compile\n    tilexr-collectives-installed\n)");
    CheckContains(path, text, "target_link_libraries(test_tilexr_collectives_stub_behavior\n    tilexr-collectives-installed\n)");
    CheckContains(path, text, "RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}");
    CheckContains(path, text, "add_executable(test_tilexr_collectives_stub_behavior");
    CheckContains(path, text, "unit/test_tilexr_collectives_stub_behavior.cpp");
    CheckContains(path, text, "add_test(NAME test_tilexr_collectives_api COMMAND test_tilexr_collectives_api)");
    CheckContains(path, text, "add_test(NAME test_tilexr_collectives_header_compile COMMAND test_tilexr_collectives_header_compile)");
    CheckContains(path, text, "add_test(NAME test_tilexr_collectives_stub_behavior COMMAND test_tilexr_collectives_stub_behavior)");
    CheckContains(path, text, "message(FATAL_ERROR");
    CheckDoesNotContain(path, text, "${TILEXR_ROOT}/src/include");
    CheckDoesNotContain(path, text, "${TILEXR_ROOT}/3rdparty");
    CheckDoesNotContain(path, text, "${TILEXR_ROOT}/build");
    CheckDoesNotContain(path, text, "${CMAKE_INSTALL_PREFIX}/bin");
    CheckDoesNotContain(path, text, "${TILEXR_INSTALL_PREFIX}/lib");
    CheckDoesNotContain(path, text, "target_include_directories(test_tilexr_collectives_header_compile");
    CheckDoesNotContain(path, text, "target_include_directories(test_tilexr_collectives_stub_behavior");
    CheckDoesNotContain(path, text, "${TILEXR_COLLECTIVES_LIB}\n    ${TILEXR_LIB}");
    CheckDoesNotContain(path, text, "CACHE PATH \"TileXR install library directory");
    CheckDoesNotContain(path, text, "/tmp/tilexr-install-split-collectives");
    CheckDoesNotContain(path, text, "/tmp/tilexr-build-split-collectives");
}

} // namespace

int main()
{
    TestCollectivesHeaderDeclaresPublicApis();
    TestCoreApiHeaderDoesNotDeclareCollectives();
    TestCommBuildDoesNotReferenceCollectives();
    TestCommBuildInstallsPublicHeadersAndKeepsLinksPrivate();
    TestCollectivesBuildDefinesSeparateSharedLibrary();
    TestCollectivesTestBuildUsesExplicitLibraryHint();
    if (g_failures != 0) {
        std::cerr << g_failures << " collectives API split checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR collectives API split checks passed" << std::endl;
    return 0;
}
