#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include "../common/int32_pattern.h"

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
    CheckContains(path, text, "CanUseCollisionFreeInt32Pattern");
    CheckContains(path, text, "validationCount");
    CheckContains(path, text, "CollectiveOp::REDUCESCATTER");
    CheckContains(path, text, "../common/int32_pattern.h");
    CheckDoesNotContain(path, text, "srcRank * 1000000 + index");
    CheckDoesNotContain(path, text, "dstRank * 1000 + index");
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
    CheckContains(path, text, "--profile");
    CheckContains(path, text, "--profile-dir");
    CheckContains(path, text, "--profile-ai-prompt");
    CheckContains(path, text, "--profile-sample-every");
    CheckContains(path, text, "--op allgather|alltoall|allreduce|reducescatter|broadcast");
    CheckContains(path, text, "TileXRAllReduce");
    CheckContains(path, text, "TileXRReduceScatter");
    CheckContains(path, text, "TileXRBroadcast");
    CheckContains(path, text, "reducescatter: count * rank_size * dtype_size");
    CheckContains(path, text, "broadcast: count * dtype_size");
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
    CheckContains(path, text, "CheckedMulInt64");
    CheckContains(path, text, "CheckedBytesForElements");
    CheckContains(path, text, "kMaxHostBufferBytes");
    CheckContains(path, text, "FitsHostBufferLimit");
    CheckContains(path, text, "AdvanceBytes");
    CheckContains(path, text, "aclrtMemcpy H2D devRecv sentinel");
    CheckContains(path, text, "actualSendBytesPerRank");
    CheckContains(path, text, "CanUseCollisionFreeInt32Pattern");
    CheckContains(path, text, "../common/int32_pattern.h");
    CheckContains(path, text, "JoinPath");
    CheckContains(path, text, "ResolveProfileOutputDir");
    CheckContains(path, text, "\"run/prof/collectives\"");
    CheckContains(path, text, "\"rank\" + std::to_string(options.rank)");
    CheckContains(path, text, "\"launch\" + std::to_string(profileLaunchIndex)");
    CheckContains(path, text, "TileXRCollectivePerfSessionCreate");
    CheckContains(path, text, "TileXRCollectivePerfSetActiveSession");
    CheckContains(path, text, "TileXRCollectivePerfWriteReport");
    CheckContains(path, text, "TileXRCollectivePerfSessionDestroy");
    CheckContains(path, text, "aclrtSynchronizeStream before perf report");
    CheckContains(path, text, "ProfileThisLaunch");
    CheckContains(path, text, "profileLaunchIndex");
    CheckContains(path, text, "TileXRCollectivePerfSetActiveSession(perfSession)");
    CheckContains(path, text, "TileXRCollectivePerfSetActiveSession(nullptr)");
    CheckContains(path, text, "skipWriteReport");
    CheckContains(path, text, "TileXRCollectivePerfSetActiveSession clear failed");
    CheckContains(path, text, "TileXRCollectivePerfSessionDestroy failed");
    CheckDoesNotContain(path, text, "sendElements * static_cast<int64_t>(options.dtype.bytes)");
    CheckDoesNotContain(path, text, "static_cast<int64_t>(static_cast<double>(bytes) * options.stepFactor)");
    CheckDoesNotContain(path, text, "srcRank * 1000000 + index");
    CheckDoesNotContain(path, text, "dstRank * 1000 + index");
}

void TestInt32PatternHasNoKnownCollisions()
{
    using TileXRCollectivesTest::CanUseCollisionFreeInt32Pattern;
    using TileXRCollectivesTest::ExpectedAllGatherValue;
    using TileXRCollectivesTest::ExpectedAllToAllValue;

    const int rankSize = 8;
    const int64_t reviewedCollisionA = 9646;
    const int64_t reviewedCollisionB = 76425;
    if (ExpectedAllGatherValue(rankSize, 0, reviewedCollisionA) ==
        ExpectedAllGatherValue(rankSize, 0, reviewedCollisionB)) {
        std::cerr << "reviewed allgather collision was reintroduced" << std::endl;
        ++g_failures;
    }

    if (!CanUseCollisionFreeInt32Pattern(128, 262144)) {
        std::cerr << "pattern should cover 1MiB INT32 validation at rank_size=128" << std::endl;
        ++g_failures;
    }

    std::set<int32_t> seen;
    const int64_t sampleCount = 4096;
    for (int src = 0; src < rankSize; ++src) {
        for (int64_t i = 0; i < sampleCount; ++i) {
            const int32_t value = ExpectedAllGatherValue(rankSize, src, i);
            if (!seen.insert(value).second) {
                std::cerr << "allgather int32 pattern collision src=" << src << " index=" << i << std::endl;
                ++g_failures;
                return;
            }
        }
    }

    seen.clear();
    for (int src = 0; src < rankSize; ++src) {
        for (int dst = 0; dst < rankSize; ++dst) {
            for (int64_t i = 0; i < sampleCount; ++i) {
                const int32_t value = ExpectedAllToAllValue(rankSize, src, dst, i);
                if (!seen.insert(value).second) {
                    std::cerr << "alltoall int32 pattern collision src=" << src
                              << " dst=" << dst << " index=" << i << std::endl;
                    ++g_failures;
                    return;
                }
            }
        }
    }
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
    CheckContains(correctnessPath, correctness, "TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC");
    CheckContains(correctnessPath, correctness, "kill_remaining_children");
    CheckContains(correctnessPath, correctness, "tail_logs");
    CheckContains(correctnessPath, correctness, "Timed out after");
    CheckContains(correctnessPath, correctness, "watchdog_pid");
    CheckContains(correctnessPath, correctness, "sleep \"${timeout_sec}\" >/dev/null 2>&1 &");
    CheckContains(correctnessPath, correctness, "wait \"${watchdog_pid}\"");
    CheckContains(correctnessPath, correctness, "wait -n");
    CheckContains(correctnessPath, correctness, "completed_count");
    CheckDoesNotContain(correctnessPath, correctness, "wait -n -p completed_pid");
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
    CheckContains(perfPath, perf, "TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC");
    CheckContains(perfPath, perf, "kill_remaining_children");
    CheckContains(perfPath, perf, "tail_logs");
    CheckContains(perfPath, perf, "Timed out after");
    CheckContains(perfPath, perf, "watchdog_pid");
    CheckContains(perfPath, perf, "sleep \"${timeout_sec}\" >/dev/null 2>&1 &");
    CheckContains(perfPath, perf, "wait \"${watchdog_pid}\"");
    CheckContains(perfPath, perf, "wait -n");
    CheckContains(perfPath, perf, "completed_count");
    CheckDoesNotContain(perfPath, perf, "wait -n -p completed_pid");
    CheckContains(perfPath, perf, "parse_profile_args");
    CheckContains(perfPath, perf, "write_profile_report_if_enabled");
    CheckContains(perfPath, perf, "tilexr_collective_profile_report.py");
    CheckContains(perfPath, perf, "--warmup-iters");
    CheckContains(perfPath, perf, "--profile-sample-every");
    CheckContains(perfPath, perf, "is_true_bool");
    CheckContains(perfPath, perf, "yes");
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
    CheckContains(path, text, "find_package(Python3 COMPONENTS Interpreter)");
    CheckContains(path, text, "test_collective_profile_report");
    CheckContains(path, text, "tilexr_collective_profile_report.py");
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
    CheckContains(path, text, "actual send bytes per rank");
    CheckContains(path, text, "--op allgather|alltoall|allreduce|reducescatter|broadcast");
    CheckContains(path, text,
        "Message-size semantics: allgather/allreduce/broadcast: count * dtype_size; alltoall/reducescatter: count * rank_size * dtype_size.");
    CheckContains(path, text, "Checked");
    CheckContains(path, text, "allreduce/reducescatter runs require `--datatype int32`");
    CheckContains(path, text, "rank_size - 1");
    CheckContains(path, text, "trace_index.json");
    CheckContains(path, text, "root-level report.html");
    CheckContains(path, text, "tilexr_collective_profile_report.py");
    CheckContains(path, text, "zoomable chronological timeline");
    CheckContains(path, text, "warmup launches are not profiled");
    CheckContains(path, text, "TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC");
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
    TestInt32PatternHasNoKnownCollisions();
    if (g_failures != 0) {
        std::cerr << g_failures << " collectives tools source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR collectives tools source checks passed" << std::endl;
    return 0;
}
