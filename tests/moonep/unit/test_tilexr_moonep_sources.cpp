#include <algorithm>
#include <cctype>
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

std::string ReadFile(const std::string &relativePath)
{
    const std::string path = std::string(kSourceRoot) + "/" + relativePath;
    std::ifstream stream(path.c_str());
    if (!stream.is_open()) {
        std::cerr << "missing file: " << relativePath << std::endl;
        ++g_failures;
        return std::string();
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void Contains(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) == std::string::npos) {
        std::cerr << label << " missing: " << needle << std::endl;
        ++g_failures;
    }
}

void Excludes(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) != std::string::npos) {
        std::cerr << label << " contains forbidden text: " << needle << std::endl;
        ++g_failures;
    }
}

std::string Lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

void CheckOrdered(const std::string &label, const std::string &contents,
                  const std::vector<std::string> &needles)
{
    size_t cursor = 0;
    for (std::vector<std::string>::const_iterator it = needles.begin();
         it != needles.end(); ++it) {
        const size_t position = contents.find(*it, cursor);
        if (position == std::string::npos) {
            std::cerr << label << " missing ordered item: " << *it << std::endl;
            ++g_failures;
            return;
        }
        cursor = position + it->size();
    }
}

} // namespace

int main()
{
    const std::string header = ReadFile("src/include/tilexr_moonep.h");
    const std::string host = ReadFile("src/moonep/host/tilexr_moonep.cpp");
    const std::string cmake = ReadFile("src/moonep/CMakeLists.txt");
    const std::string testCmake = ReadFile("tests/moonep/CMakeLists.txt");
    const std::string flowDemo =
        ReadFile("tests/moonep/demo/tilexr_moonep_flow_demo.cpp");
    const std::string flowRunner = ReadFile("tests/moonep/demo/run_a5.sh");

    Contains("public header", header, "extern \"C\"");
    Contains("public header", header, "TileXRMoonEpTensorV1");
    Contains("public header", header, "TileXRMoonEpPlanV1");
    Excludes("public header", header, "tilexr_api.h");
    Excludes("public header", header, "std::");

    Contains("host", host, "TileXRMoonEpPlannerGetWorkspaceSizeV2");
    Contains("host", host, "TileXRMoonEpPlannerV2");
    Contains("host", host, "aclrtMemsetAsync");
    Contains("host", host, "aclrtMemcpyAsync");
    Contains("host", host, "ACL_MEMCPY_DEVICE_TO_DEVICE");
    Excludes("host", host, "aclrtSynchronizeStream");

    Contains("CMake", cmake, "add_library(tilexr-moonep SHARED");
    Contains("CMake", cmake, "SOVERSION 1");
    Contains("CMake", cmake, "INSTALL_RPATH \"$ORIGIN\"");
    Contains("CMake", cmake, "tilexr-moonep-planner");
    Excludes("CMake", cmake, "devlib");

    Excludes("public header", Lower(header), "hccl");
    Excludes("host", Lower(host), "hccl");
    Excludes("CMake", Lower(cmake), "hccl");

    Contains("test CMake", testCmake, "if(TARGET tilexr-moonep)");
    Contains("test CMake", testCmake, "add_executable(tilexr_moonep_flow_demo");
    Contains("test CMake", testCmake, "tilexr-moonep");
    Contains("flow demo", flowDemo, "TileXRCommInitRankLocal");
    Contains("flow demo", flowDemo, "dispatchedCapacity != static_cast<int64_t>(routeCount)");
    Contains("flow demo", flowDemo, "statusHost[0] != TILEXR_MOONEP_PLANNER_STATUS_SUCCESS");
    Contains("flow demo", flowDemo, "cuHost.back()");
    Contains("flow demo", flowDemo, "CheckPrefixAndZeroTail");
    Contains("flow demo", flowDemo, "torch_validated=false");
    Contains("flow demo", flowDemo, "transport_performance_valid=false");
    CheckOrdered("native flow", flowDemo, {
        "TileXRMoonEpPlanningV1(&planning",
        "TileXRMoonEpDispatchV1(&forwardDispatch",
        "TileXRMoonEpPrefetchWeightV1(&prefetch",
        "TileXRMoonEpCombineV1(&forwardCombine",
        "TileXRMoonEpDispatchV1(&backwardDispatch",
        "TileXRMoonEpCombineV1(&backwardCombine",
        "TileXRMoonEpReduceGradV1(&reduceGrad",
        "aclrtSynchronizeStream(resources->stream)",
    });
    CheckOrdered("native flow cleanup", flowDemo, {
        "\"local completion synchronize\"",
        "DemoBarrierAll(options.rank, options.world)",
        "Cleanup(&resources)",
    });
    Contains("flow demo", flowDemo, "TILEXR_MOONEP_FLOW_BARRIER_ADDR");
    Contains("flow runner", flowRunner, "block_dim=$((64 / ranks_per_device))");
    Contains("flow runner", flowRunner, "${SCRIPT_DIR}/tilexr_moonep_flow_demo");
    Contains("flow runner", flowRunner, "device=$((rank % physical_device_count))");
    Contains("flow runner", flowRunner, "TILEXR_MOONEP_PLANNER_BLOCK_DIM");
    Contains("flow runner", flowRunner, "torch_validated=false");
    Contains("flow runner", flowRunner, "transport_performance_valid=false");
    Excludes("flow demo", Lower(flowDemo), "hccl");
    Excludes("flow runner", Lower(flowRunner), "hccl");
    return g_failures == 0 ? 0 : 1;
}
