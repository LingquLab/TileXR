#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#ifdef TILEXR_SOURCE_ROOT
const char *kSourceRoot = TILEXR_SOURCE_ROOT;
#else
const char *kSourceRoot = ".";
#endif

std::string JoinPath(const std::string &base, const std::string &path)
{
    if (base.empty() || base[base.size() - 1] == '/') {
        return base + path;
    }
    return base + "/" + path;
}

bool ReadFile(const std::string &relativePath, std::string *contents)
{
    const std::string fullPath = JoinPath(kSourceRoot, relativePath);
    std::ifstream stream(fullPath.c_str());
    if (!stream.is_open()) {
        std::cerr << "missing file: " << relativePath << std::endl;
        ++g_failures;
        return false;
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    *contents = buffer.str();
    return true;
}

void CheckContains(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) == std::string::npos) {
        std::cerr << label << " missing: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) != std::string::npos) {
        std::cerr << label << " contains forbidden string: " << needle << std::endl;
        ++g_failures;
    }
}

void TestPublicHeader()
{
    std::string contents;
    if (!ReadFile("src/include/tilexr_ep.h", &contents)) {
        return;
    }

    CheckContains("src/include/tilexr_ep.h", contents, "#ifdef __cplusplus");
    CheckContains("src/include/tilexr_ep.h", contents, "extern \"C\"");
    CheckContains("src/include/tilexr_ep.h", contents, "int TileXRMoeEpDispatch(");
    CheckContains("src/include/tilexr_ep.h", contents, "TileXRCommPtr comm");
    CheckContains("src/include/tilexr_ep.h", contents, "TileXR::TileXRDataType dtype");
    CheckContains("src/include/tilexr_ep.h", contents, "aclrtStream stream");
}

void TestBuildPlacement()
{
    std::string rootCmake;
    if (ReadFile("CMakeLists.txt", &rootCmake)) {
        CheckContains("CMakeLists.txt", rootCmake,
            "option(TILEXR_BUILD_EP \"Build TileXR EP communication library\" OFF)");
        CheckContains("CMakeLists.txt", rootCmake, "add_subdirectory(src/ep)");
    }

    std::string epCmake;
    if (ReadFile("src/ep/CMakeLists.txt", &epCmake)) {
        CheckContains("src/ep/CMakeLists.txt", epCmake, "add_library(tilexr-ep SHARED");
        CheckContains("src/ep/CMakeLists.txt", epCmake, "tile-comm");
        CheckContains("src/ep/CMakeLists.txt", epCmake, "tilexr_ep.h");
        CheckContains("src/ep/CMakeLists.txt", epCmake, "install(TARGETS tilexr-ep");
    }
}

void TestBishengVersionDateParsing()
{
    std::string epCmake;
    if (!ReadFile("src/ep/CMakeLists.txt", &epCmake)) {
        return;
    }

    CheckContains("src/ep/CMakeLists.txt", epCmake,
        "([0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9])");
    CheckContains("src/ep/CMakeLists.txt", epCmake,
        "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])");
    CheckContains("src/ep/CMakeLists.txt", epCmake, "${CMAKE_MATCH_1}${CMAKE_MATCH_2}${CMAKE_MATCH_3}");
    CheckContains("src/ep/CMakeLists.txt", epCmake, "TILEXR_EP_BISHENG_DATE GREATER_EQUAL 20250428");
    CheckNotContains("src/ep/CMakeLists.txt", epCmake, "{8}");
    CheckNotContains("src/ep/CMakeLists.txt", epCmake, "{4}");
}

void TestNoForbiddenDependencies()
{
    const std::vector<std::string> paths = {
        "src/include/tilexr_ep.h",
        "src/ep/CMakeLists.txt",
        "src/ep/host/ep_layout.h",
        "src/ep/host/ep_layout.cpp",
    };
    const std::vector<std::string> forbidden = {
        "src/mc2",
        "3rdparty/ops-transformer",
        "GetHcclContext",
        "TileXRUDMARegister",
        "UDMAPut",
        "shmem",
    };

    for (std::vector<std::string>::const_iterator path = paths.begin(); path != paths.end(); ++path) {
        std::string contents;
        if (!ReadFile(*path, &contents)) {
            continue;
        }
        for (std::vector<std::string>::const_iterator needle = forbidden.begin(); needle != forbidden.end(); ++needle) {
            CheckNotContains(*path, contents, *needle);
        }
    }
}

} // namespace

int main()
{
    TestPublicHeader();
    TestBuildPlacement();
    TestBishengVersionDateParsing();
    TestNoForbiddenDependencies();
    return g_failures == 0 ? 0 : 1;
}
