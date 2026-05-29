#include <sys/stat.h>

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

bool FileExists(const std::string &path)
{
    struct stat st {};
    return stat(RepoPath(path).c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string ReadFile(const std::string &path)
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

void CheckFileExists(const std::string &path)
{
    if (!FileExists(path)) {
        std::cerr << "expected " << RepoPath(path) << " to exist" << std::endl;
        ++g_failures;
    }
}

void CheckContains(const std::string &path, const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected " << path << " to contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckDoesNotContain(const std::string &path, const std::string &text, const std::string &needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "expected " << path << " not to contain: " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void TestCorrectnessRunnerSource()
{
    const std::string path = "tests/collectives/integration/test_tilexr_collectives_correctness.cpp";
    CheckFileExists(path);
    const auto text = ReadFile(path);
    CheckContains(path, text, "TileXRCommInitRankLocal");
    CheckContains(path, text, "TileXRAllGather");
    CheckContains(path, text, "TileXRAllToAll");
    CheckContains(path, text, "TILEXR_DATA_TYPE_INT32");
    CheckContains(path, text, "--rank-size");
    CheckContains(path, text, "--rank");
    CheckContains(path, text, "--count");
    CheckContains(path, text, "--first-npu");
    CheckContains(path, text, "--op");
    CheckContains(path, text, "aclrtSetDevice");
    CheckContains(path, text, "aclrtCreateStream");
    CheckContains(path, text, "aclrtMalloc");
    CheckContains(path, text, "aclrtMemcpy");
    CheckContains(path, text, "aclrtSynchronizeStream");
    CheckContains(path, text, "ExpectedAllGatherValue");
    CheckContains(path, text, "ExpectedAllToAllValue");
}

void TestPerfToolSource()
{
    const std::string path = "tests/collectives/tilexr-tests/tilexr_collective_perf.cpp";
    CheckFileExists(path);
    const auto text = ReadFile(path);
    CheckContains(path, text, "--op");
    CheckContains(path, text, "--min-bytes");
    CheckContains(path, text, "--max-bytes");
    CheckContains(path, text, "--step-factor");
    CheckContains(path, text, "--iters");
    CheckContains(path, text, "--warmup-iters");
    CheckContains(path, text, "--datatype");
    CheckContains(path, text, "--rank-size");
    CheckContains(path, text, "--rank");
    CheckContains(path, text, "--first-npu");
    CheckContains(path, text, "--check");
    CheckContains(path, text, "--csv");
    CheckContains(path, text, "--min-algbw");
    CheckContains(path, text, "--max-latency-us");
    CheckContains(path, text, "aclrtCreateEvent");
    CheckContains(path, text, "aclrtRecordEvent");
    CheckContains(path, text, "aclrtEventElapsedTime");
    CheckContains(path, text, "algbw(GB/s)");
    CheckContains(path, text, "busbw(GB/s)");
    CheckContains(path, text, "avg(us)");
    CheckContains(path, text, "min(us)");
    CheckContains(path, text, "max(us)");
    CheckContains(path, text, "ParseDataType");
    CheckContains(path, text, "int8");
    CheckContains(path, text, "int16");
    CheckContains(path, text, "int32");
    CheckContains(path, text, "int64");
    CheckContains(path, text, "fp16");
    CheckContains(path, text, "fp32");
    CheckContains(path, text, "bf16");
    CheckContains(path, text, "ComputeAlgBandwidthGbps");
    CheckContains(path, text, "ComputeBusBandwidthGbps");
    CheckContains(path, text, "ValidateInt32");
}

void TestLauncherScripts()
{
    const std::string correctnessPath = "tests/collectives/run_collectives_correctness.sh";
    CheckFileExists(correctnessPath);
    const auto correctness = ReadFile(correctnessPath);
    CheckContains(correctnessPath, correctness, "rank_size");
    CheckContains(correctnessPath, correctness, "count");
    CheckContains(correctnessPath, correctness, "first_npu");
    CheckContains(correctnessPath, correctness, "bin_dir");
    CheckContains(correctnessPath, correctness, "collectives_correctness_rank${rank}.log");
    CheckContains(correctnessPath, correctness, "TILEXR_SKIP_IF_INSUFFICIENT_NPUS");
    CheckContains(correctnessPath, correctness, "npu-smi info -l");
    CheckContains(correctnessPath, correctness, "tail -n");
    CheckContains(correctnessPath, correctness, "test_tilexr_collectives_correctness");

    const std::string perfPath = "tests/collectives/run_collective_perf.sh";
    CheckFileExists(perfPath);
    const auto perf = ReadFile(perfPath);
    CheckContains(perfPath, perf, "tilexr_collective_perf");
    CheckContains(perfPath, perf, "--rank-size");
    CheckContains(perfPath, perf, "--rank");
    CheckContains(perfPath, perf, "--first-npu");
    CheckContains(perfPath, perf, "collective_perf_rank${rank}.log");
    CheckContains(perfPath, perf, "TILEXR_SKIP_IF_INSUFFICIENT_NPUS");
}

void TestCMakeWiring()
{
    const std::string path = "tests/collectives/CMakeLists.txt";
    const auto text = ReadFile(path);
    CheckContains(path, text, "integration/test_tilexr_collectives_correctness.cpp");
    CheckContains(path, text, "tilexr-tests/tilexr_collective_perf.cpp");
    CheckContains(path, text, "add_executable(test_tilexr_collectives_correctness");
    CheckContains(path, text, "add_executable(tilexr_collective_perf");
    CheckContains(path, text, "target_link_libraries(test_tilexr_collectives_correctness");
    CheckContains(path, text, "target_link_libraries(tilexr_collective_perf");
    CheckContains(path, text, "test_tilexr_collectives_tools_sources");
    CheckContains(path, text, "run_collectives_correctness.sh");
    CheckContains(path, text, "run_collective_perf.sh");
    CheckDoesNotContain(path, text, "add_test(NAME test_tilexr_collectives_correctness");
    CheckDoesNotContain(path, text, "add_test(NAME tilexr_collective_perf");
}

void TestReadmeDocumentsManualRuns()
{
    const std::string path = "tests/collectives/README.md";
    CheckFileExists(path);
    const auto text = ReadFile(path);
    CheckContains(path, text, "libtilexr-collectives");
    CheckContains(path, text, "tile-comm");
    CheckContains(path, text, "test_tilexr_collectives_correctness");
    CheckContains(path, text, "tilexr_collective_perf");
    CheckContains(path, text, "run_collectives_correctness.sh");
    CheckContains(path, text, "run_collective_perf.sh");
    CheckContains(path, text, "--rank-size");
    CheckContains(path, text, "--rank");
    CheckContains(path, text, "--first-npu");
    CheckContains(path, text, "--op");
    CheckContains(path, text, "--datatype");
    CheckContains(path, text, "--min-bytes");
    CheckContains(path, text, "--max-bytes");
    CheckContains(path, text, "--check");
    CheckContains(path, text, "--csv");
    CheckContains(path, text, "algbw(GB/s)");
    CheckContains(path, text, "busbw(GB/s)");
    CheckContains(path, text, "bytes * iters");
    CheckContains(path, text, "rank_size - 1");
    CheckContains(path, text, "TILEXR_SKIP_IF_INSUFFICIENT_NPUS");
    CheckContains(path, text, "CTest");
    CheckContains(path, text, "manual");
}

} // namespace

int main()
{
    TestCorrectnessRunnerSource();
    TestPerfToolSource();
    TestLauncherScripts();
    TestCMakeWiring();
    TestReadmeDocumentsManualRuns();
    if (g_failures != 0) {
        std::cerr << g_failures << " collectives tools source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR collectives tools source checks passed" << std::endl;
    return 0;
}
