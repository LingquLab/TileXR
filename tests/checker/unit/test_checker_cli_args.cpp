#include <cstdint>
#include <iostream>
#include <string>

#include "tilexr/checker/cli.h"

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

}  // namespace

int main() {
    TestParseAllGatherRoundRobin();
    TestParseAllReduceSumWithInjectionFlags();
    TestParseInvalidOpReturnsUnsupported();
    return g_failures == 0 ? 0 : 1;
}
