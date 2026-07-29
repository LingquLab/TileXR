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
    if (text.find(needle) == std::string::npos) {
        std::cerr << path << " does not contain required text: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << path << " unexpectedly contains text: " << needle << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void CheckAppearsBefore(
    const std::string& path, const std::string& text, const std::string& beforeNeedle, const std::string& afterNeedle)
{
    const auto beforePos = text.find(beforeNeedle);
    const auto afterPos = text.find(afterNeedle);
    if (beforePos == std::string::npos || afterPos == std::string::npos || beforePos > afterPos) {
        std::cerr << path << " expected " << beforeNeedle << " before " << afterNeedle << std::endl;
        ++g_failures;
    }
}

void CheckBlockNotContains(const std::string& path, const std::string& text, const std::string& start,
    const std::string& end, const std::string& needle)
{
    const auto startPos = text.find(start);
    const auto endPos = startPos == std::string::npos ? std::string::npos : text.find(end, startPos);
    if (startPos == std::string::npos || endPos == std::string::npos) {
        std::cerr << path << " missing block: " << start << std::endl;
        ++g_failures;
        return;
    }
    CheckNotContains(path, text.substr(startPos, endPos - startPos), needle);
}

void TestMemoryDemoKernelUsesPeerMemorySemantics()
{
    const std::string path = "tests/memory/demo/tilexr_memory_demo_kernel.cpp";
    const std::string text = ReadFile(path);
    CheckAppearsBefore(path, text, "#include \"comm_args.h\"", "#include \"kernel_operator.h\"");
    CheckContains(path, text, "tilexr_memory_push_kernel");
    CheckContains(path, text, "tilexr_memory_collect_kernel");
    CheckContains(path, text, "peerMems");
    CheckContains(path, text, "AscendC::DataCopyPad");
    CheckContains(path, text, "IPC_DATA_OFFSET");
    CheckNotContains(path, text, "WaitRankInnerFlag");
    CheckNotContains(path, text, "SyncCollectives");
    CheckNotContains(path, text, "TileXRUDMARegister");
    CheckNotContains(path, text, "UDMAPut");
    CheckNotContains(path, text, "UDMAGet");
}

void TestCommArgsOwnsAicoreGmAddrCompatibility()
{
    const std::string path = "src/include/comm_args.h";
    const std::string text = ReadFile(path);
    CheckContains(path, text, "defined(__CCE__)");
    CheckContains(path, text, "defined(__CCE_IS_AICORE__)");
    CheckContains(path, text, "TILEXR_ASCENDC_AICORE_COMPILE");
    CheckNotContains(path, text, "__DAV_C220_VEC__");
    CheckNotContains(path, text, "__DAV_C310__");
    CheckAppearsBefore(path, text, "#include \"kernel_operator.h\"", "#ifndef GM_ADDR");
}

void TestMemoryDemoHostAndRunnerExist()
{
    const std::string hostPath = "tests/memory/demo/tilexr_memory_demo.cpp";
    const std::string hostText = ReadFile(hostPath);
    CheckContains(hostPath, hostText, "TileXRCommInitRankLocal");
    CheckContains(hostPath, hostText, "TileXRGetCommArgsDev");
    CheckContains(hostPath, hostText, "PushInputToPeerWindowsOnHost");
    CheckContains(hostPath, hostText, "CollectLocalWindowOnHost");
    CheckContains(hostPath, hostText, "ExchangeInputSegmentsOnHost");
    CheckContains(hostPath, hostText, "TILEXR_MEMORY_DEMO_HOST_STAGING\", 0");
    CheckContains(hostPath, hostText, "TILEXR_MEMORY_DEMO_HOST_COPY\", 0");
    CheckContains(hostPath, hostText, "diagnostic host staging");
    CheckNotContains(hostPath, hostText, "host staging memory fallback");
    CheckContains(hostPath, hostText, "std::string host");
    CheckContains(hostPath, hostText, "inet_pton(AF_INET, endpoint.host.c_str()");
    CheckContains(hostPath, hostText, "htonl(INADDR_ANY)");
    CheckNotContains(hostPath, hostText, "htonl(INADDR_LOOPBACK)");
    CheckContains(hostPath, hostText, "ACL_MEMCPY_DEVICE_TO_DEVICE");
    CheckContains(hostPath, hostText, "diagnostic host peer-memory copy");
    CheckContains(hostPath, hostText, "launch memory push kernel");
    CheckContains(hostPath, hostText, "launch_tilexr_memory_push");
    CheckContains(hostPath, hostText, "launch_tilexr_memory_collect");
    CheckContains(hostPath, hostText, "all ranks completed memory push kernels");
    CheckContains(hostPath, hostText, "all ranks completed memory collect kernels");
    CheckNotContains(hostPath, hostText, "TileXRUDMARegister");

    const std::string runPath = "tests/memory/demo/run_tilexr_memory_demo.sh";
    const std::string runText = ReadFile(runPath);
    CheckContains(runPath, runText, "tilexr_memory_demo");
    CheckContains(runPath, runText, "${TILEXR_ROOT}/install/lib64");
}

void TestMemoryHostTargetsExcludeCannDevlib()
{
    const std::string path = "tests/memory/CMakeLists.txt";
    const std::string text = ReadFile(path);
    CheckBlockNotContains(path, text, "link_directories(", "add_executable(test_tilexr_memory_demo_sources",
        "-linux/devlib");
}

} // namespace

int main()
{
    TestMemoryDemoKernelUsesPeerMemorySemantics();
    TestCommArgsOwnsAicoreGmAddrCompatibility();
    TestMemoryDemoHostAndRunnerExist();
    TestMemoryHostTargetsExcludeCannDevlib();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR memory demo source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR memory demo source checks passed" << std::endl;
    return 0;
}
