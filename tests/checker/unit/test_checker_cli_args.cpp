#include <cstdint>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

#include <cstdio>

#include <unistd.h>

#include "tilexr/checker/cli.h"
#include "tilexr/checker/diagnostics.h"

namespace tilexr {
namespace checker {

CheckerStatus NormalizeExecutorCliStatus(const RunResult &result);

}  // namespace checker
}  // namespace tilexr

namespace {

int g_failures = 0;

void ExpectTrue(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << "\n";
        ++g_failures;
    }
}

void ExpectEqInt(int actual, int expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

void ExpectEqString(const std::string &actual, const std::string &expected,
                    const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

std::string MakeTempDir() {
    char dir_template[] = "/tmp/tilexr-checker-cli-XXXXXX";
    char *created = mkdtemp(dir_template);
    if (created == nullptr) {
        std::cerr << "mkdtemp failed\n";
        ++g_failures;
        return std::string();
    }
    return std::string(created);
}

std::string MakeTempFile() {
    char file_template[] = "/tmp/tilexr-checker-cli-file-XXXXXX";
    const int fd = mkstemp(file_template);
    if (fd < 0) {
        std::cerr << "mkstemp failed\n";
        ++g_failures;
        return std::string();
    }
    close(fd);
    return std::string(file_template);
}

tilexr::checker::CliOptions MakeValidAllGatherOptions() {
    tilexr::checker::CliOptions options;
    options.test_case.op = tilexr::checker::CollectiveOp::kAllGather;
    options.test_case.rank_size = 4;
    options.test_case.count = 16;
    options.test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    options.test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    options.test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    options.test_case.magic = 0x1234;
    options.output_dir = MakeTempDir();
    return options;
}

void TestParseAllGatherRoundRobin() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "allgather",
        "--rank-size", "4",
        "--count", "16",
        "--datatype", "int32",
        "--scheduler", "round_robin",
        "--output-dir", "/tmp/checker-out",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allgather parse status");
    ExpectEqInt(static_cast<int>(options.test_case.op),
                static_cast<int>(tilexr::checker::CollectiveOp::kAllGather),
                "allgather op");
    ExpectEqInt(options.test_case.rank_size, 4, "allgather rank size");
    ExpectEqInt(static_cast<int>(options.test_case.count), 16, "allgather count");
    ExpectEqInt(static_cast<int>(options.test_case.scheduler),
                static_cast<int>(tilexr::checker::SchedulerMode::kRoundRobin),
                "allgather scheduler");
    ExpectEqString(options.output_dir, "/tmp/checker-out", "allgather output dir");
    ExpectTrue(!options.inject_read_before_copy, "allgather inject read-before-copy default");
    ExpectTrue(!options.inject_peer_user_read, "allgather inject peer-user-read default");
    ExpectTrue(!options.show_help, "allgather show help default");
}

void TestParseAllReduceSumWithInjectionFlags() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "allreduce",
        "--reduce-op", "sum",
        "--rank-size", "2",
        "--count", "16",
        "--datatype", "int32",
        "--inject-read-before-copy",
        "--inject-peer-user-read",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allreduce parse status");
    ExpectEqInt(static_cast<int>(options.test_case.op),
                static_cast<int>(tilexr::checker::CollectiveOp::kAllReduce),
                "allreduce op");
    ExpectEqInt(static_cast<int>(options.test_case.reduce_op),
                static_cast<int>(TileXR::TILEXR_REDUCE_SUM),
                "allreduce reduce op");
    ExpectTrue(options.inject_read_before_copy, "allreduce inject read-before-copy set");
    ExpectTrue(options.inject_peer_user_read, "allreduce inject peer-user-read set");
}

void TestParseInvalidOpReturnsUnsupported() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "broadcast",
        "--rank-size", "2",
        "--count", "16",
        "--datatype", "int32",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "invalid op parse status");
}

void TestParseRankSizeOverflowReturnsUnsupported() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "allgather",
        "--rank-size", "2147483648",
        "--count", "16",
        "--datatype", "int32",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "rank-size overflow parse status");
}

void TestRunCheckerCliPassExitCode() {
    tilexr::checker::CliOptions options = MakeValidAllGatherOptions();
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "run cli pass exit code");
    ExpectTrue(stderr_stream.str().empty(), "run cli pass stderr empty");
    ExpectTrue(stdout_stream.str().find("checker: PASS") != std::string::npos,
               "run cli pass stdout");
}

void TestNormalizeExecutorFailStatusPreservedForReporting() {
    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Fail("executor found mismatches");

    tilexr::checker::OutputMismatch mismatch;
    mismatch.rank = 0;
    mismatch.element_index = 1;
    mismatch.expected = 1;
    mismatch.actual = 2;
    mismatch.context = "synthetic executor mismatch";
    result.mismatches.push_back(mismatch);

    const tilexr::checker::CheckerStatus status =
        tilexr::checker::NormalizeExecutorCliStatus(result);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kFail),
                "normalize executor fail status");
}

void TestRunCheckerCliInjectedFindingExitCode() {
    tilexr::checker::CliOptions options = MakeValidAllGatherOptions();
    options.inject_read_before_copy = true;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 1, "run cli injected finding exit code");
    ExpectTrue(stdout_stream.str().find("checker: FAIL") != std::string::npos,
               "run cli injected finding stdout fail");
}

void TestRunCheckerCliInvalidOutputDirExitCode() {
    tilexr::checker::CliOptions options = MakeValidAllGatherOptions();
    options.output_dir.clear();
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 2, "run cli invalid output dir exit code");
}

void TestRunCheckerCliReportWriteFailureExitCode() {
    tilexr::checker::CliOptions options = MakeValidAllGatherOptions();
    options.output_dir = MakeTempFile() + "/child";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 3, "run cli report write failure exit code");
}

}  // namespace

int main() {
    TestParseAllGatherRoundRobin();
    TestParseAllReduceSumWithInjectionFlags();
    TestParseInvalidOpReturnsUnsupported();
    TestParseRankSizeOverflowReturnsUnsupported();
    TestRunCheckerCliPassExitCode();
    TestNormalizeExecutorFailStatusPreservedForReporting();
    TestRunCheckerCliInjectedFindingExitCode();
    TestRunCheckerCliInvalidOutputDirExitCode();
    TestRunCheckerCliReportWriteFailureExitCode();
    return g_failures == 0 ? 0 : 1;
}
