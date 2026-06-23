#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

#include <cstdio>

#include <sys/stat.h>
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

std::string ReadFile(const std::string &path) {
    std::ifstream input(path.c_str());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

void WriteFile(const std::string &path, const std::string &content) {
    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    output << content;
    if (!output) {
        std::cerr << "write failed: " << path << "\n";
        ++g_failures;
    }
}

bool FileExists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
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
        "--server-count", "2",
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
    ExpectEqInt(options.test_case.server_count, 2, "allgather server count");
    ExpectEqInt(static_cast<int>(options.test_case.count), 16, "allgather count");
    ExpectEqInt(static_cast<int>(options.test_case.scheduler),
                static_cast<int>(tilexr::checker::SchedulerMode::kRoundRobin),
                "allgather scheduler");
    ExpectEqString(options.output_dir, "/tmp/checker-out", "allgather output dir");
    ExpectTrue(!options.inject_read_before_copy, "allgather inject read-before-copy default");
    ExpectTrue(!options.inject_peer_user_read, "allgather inject peer-user-read default");
    ExpectTrue(!options.inject_pipe_wait, "allgather inject pipe-wait default");
    ExpectTrue(!options.show_help, "allgather show help default");
}

void TestParseAllGatherHierarchyDoubleRingAlgorithm() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "allgather",
        "--rank-size", "10",
        "--count", "16",
        "--datatype", "int32",
        "--scheduler", "round_robin",
        "--algorithm", "allgather_hierarchy_double_ring",
        "--output-dir", "/tmp/checker-hdb",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "hdb allgather parse status");
    ExpectEqInt(static_cast<int>(options.test_case.op),
                static_cast<int>(tilexr::checker::CollectiveOp::kAllGather),
                "hdb allgather op");
    ExpectEqInt(options.test_case.rank_size, 10, "hdb allgather rank size");
    ExpectEqInt(static_cast<int>(options.test_case.algorithm),
                static_cast<int>(tilexr::checker::AlgorithmId::kAllGatherHierarchyDoubleRing),
                "hdb allgather algorithm");
    ExpectEqString(options.output_dir, "/tmp/checker-hdb", "hdb allgather output dir");
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
        "--inject-pipe-wait",
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
    ExpectTrue(options.inject_pipe_wait, "allreduce inject pipe-wait set");
}

void TestParseAllReduceBigDataAlgorithm() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "allreduce",
        "--reduce-op", "sum",
        "--rank-size", "4",
        "--count", "524288",
        "--datatype", "int32",
        "--scheduler", "round_robin",
        "--algorithm", "allreduce_big_data",
        "--output-dir", "/tmp/checker-allreduce-big-data",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allreduce big data parse status");
    ExpectEqInt(static_cast<int>(options.test_case.op),
                static_cast<int>(tilexr::checker::CollectiveOp::kAllReduce),
                "allreduce big data op");
    ExpectEqInt(options.test_case.rank_size, 4, "allreduce big data rank size");
    ExpectEqInt(static_cast<int>(options.test_case.algorithm),
                static_cast<int>(tilexr::checker::AlgorithmId::kAllReduceBigData),
                "allreduce big data algorithm");
    ExpectEqString(options.output_dir, "/tmp/checker-allreduce-big-data",
                   "allreduce big data output dir");
}

void TestParseUnsupportedDatatypeKeepsExecutableCaseForReporting() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "allgather",
        "--rank-size", "4",
        "--count", "16",
        "--datatype", "fp16",
        "--scheduler", "round_robin",
        "--output-dir", "/tmp/checker-unsupported-report",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "unsupported datatype parse status for reportable execution");
    ExpectEqInt(static_cast<int>(options.test_case.data_type),
                static_cast<int>(TileXR::TILEXR_DATA_TYPE_FP16),
                "unsupported datatype parse dtype preserved");
    ExpectEqString(options.output_dir, "/tmp/checker-unsupported-report",
                   "unsupported datatype parse output dir");
}

void TestParseEpDispatch() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "ep_dispatch",
        "--rank-size", "2",
        "--server-count", "2",
        "--bs", "3",
        "--h", "4",
        "--top-k", "2",
        "--moe-expert-num", "4",
        "--datatype", "fp16",
        "--inject-ep-window-read",
        "--output-dir", "/tmp/checker-ep-dispatch",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "ep dispatch parse status");
    ExpectEqInt(static_cast<int>(options.test_case.op),
                static_cast<int>(tilexr::checker::CollectiveOp::kEpDispatch),
                "ep dispatch op");
    ExpectEqInt(options.test_case.rank_size, 2, "ep dispatch rank size");
    ExpectEqInt(options.test_case.server_count, 2, "ep dispatch server count");
    ExpectEqInt(static_cast<int>(options.test_case.bs), 3, "ep dispatch bs");
    ExpectEqInt(static_cast<int>(options.test_case.h), 4, "ep dispatch hidden size");
    ExpectEqInt(static_cast<int>(options.test_case.top_k), 2, "ep dispatch top k");
    ExpectEqInt(static_cast<int>(options.test_case.moe_expert_num), 4,
                "ep dispatch expert count");
    ExpectEqInt(static_cast<int>(options.test_case.data_type),
                static_cast<int>(TileXR::TILEXR_DATA_TYPE_FP16),
                "ep dispatch dtype");
    ExpectEqString(options.output_dir, "/tmp/checker-ep-dispatch",
                   "ep dispatch output dir");
    ExpectTrue(options.inject_ep_window_read, "ep dispatch inject window read flag");
}

void TestParseEpCombineKeepsExecutableCaseForReporting() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "ep_combine",
        "--rank-size", "2",
        "--server-count", "2",
        "--bs", "3",
        "--h", "4",
        "--top-k", "2",
        "--moe-expert-num", "4",
        "--datatype", "fp16",
        "--output-dir", "/tmp/checker-ep-combine",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "ep combine parse status");
    ExpectEqInt(static_cast<int>(options.test_case.op),
                static_cast<int>(tilexr::checker::CollectiveOp::kEpCombine),
                "ep combine op");
    ExpectEqString(options.output_dir, "/tmp/checker-ep-combine",
                   "ep combine output dir");
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

void TestParseInvalidServerCountReturnsUnsupported() {
    const char *argv[] = {
        "tilexr_checker",
        "--op", "allgather",
        "--rank-size", "4",
        "--server-count", "3",
        "--count", "16",
        "--datatype", "int32",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "invalid server count parse status");
}

void TestParseGenerateTraceAdapterMode() {
    const char *argv[] = {
        "tilexr_checker",
        "--generate-trace-adapter",
        "--adapter-name", "alltoall_demo",
        "--source-file", "src/collectives/kernels/alltoall_demo.h",
        "--target-header", "alltoall_demo.h",
        "--output", "/tmp/alltoall_demo_trace_shim.h",
        "--manifest-output", "/tmp/alltoall_demo_trace_manifest.json",
        "--runner-output", "/tmp/alltoall_demo_trace_runner.cpp",
        "--probe-output", "/tmp/alltoall_demo_trace_probe.cpp",
        "--runner-function", "RunAllToAllDemoTrace",
        "--runner-materializer", "MaterializeAllToAllDemo",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "generate trace adapter parse status");
    ExpectTrue(options.generate_trace_adapter, "generate trace adapter mode set");
    ExpectEqString(options.trace_adapter_spec.adapter_name, "alltoall_demo",
                   "generate adapter name");
    ExpectEqString(options.trace_adapter_spec.source_file,
                   "src/collectives/kernels/alltoall_demo.h",
                   "generate source file");
    ExpectEqString(options.trace_adapter_spec.target_header, "alltoall_demo.h",
                   "generate target header");
    ExpectEqString(options.trace_adapter_output, "/tmp/alltoall_demo_trace_shim.h",
                   "generate output path");
    ExpectEqString(options.trace_adapter_manifest_output,
                   "/tmp/alltoall_demo_trace_manifest.json",
                   "generate manifest output path");
    ExpectEqString(options.trace_runner_output,
                   "/tmp/alltoall_demo_trace_runner.cpp",
                   "generate runner output path");
    ExpectEqString(options.trace_probe_output,
                   "/tmp/alltoall_demo_trace_probe.cpp",
                   "generate probe output path");
    ExpectEqString(options.trace_runner_spec.function_name,
                   "RunAllToAllDemoTrace",
                   "generate runner function");
    ExpectEqString(options.trace_runner_spec.materializer_name,
                   "MaterializeAllToAllDemo",
                   "generate runner materializer");
    ExpectEqString(options.trace_probe_spec.adapter_name, "alltoall_demo",
                   "generate probe adapter name");
    ExpectEqString(options.trace_probe_spec.shim_include, "alltoall_demo_trace_shim.h",
                   "generate probe local shim include");
    ExpectEqString(options.trace_probe_spec.expected_source_file,
                   "src/collectives/kernels/alltoall_demo.h",
                   "generate probe expected source file");
    ExpectEqString(options.trace_probe_spec.expected_target_header,
                   "alltoall_demo.h",
                   "generate probe expected target header");
}

void TestParseGenerateTraceAdapterStrictSourceObservation() {
    const char *argv[] = {
        "tilexr_checker",
        "--generate-trace-adapter",
        "--strict-source-observation",
        "--adapter-name", "raw_copy_demo",
        "--source-file", "src/collectives/kernels/raw_copy_demo.h",
        "--target-header", "raw_copy_demo.h",
        "--output", "/tmp/raw_copy_demo_trace_shim.h",
        "--manifest-output", "/tmp/raw_copy_demo_trace_manifest.json",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "generate strict trace adapter parse status");
    ExpectTrue(options.trace_adapter_spec.strict_source_observation,
               "generate strict source observation flag");
}

void TestParseVerificationReportOutput() {
    const char *argv[] = {
        "tilexr_checker",
        "--verify-installed-trace",
        "--adapter-name", "allreduce_big_data",
        "--repo-root", TILEXR_SOURCE_ROOT,
        "--verification-report", "/tmp/tilexr-checker-verification.json",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "verification report parse status");
    ExpectEqString(options.verification_report_output,
                   "/tmp/tilexr-checker-verification.json",
                   "verification report output path");
}

void TestParseInstalledTraceInventoryModes() {
    {
        const char *argv[] = {
            "tilexr_checker",
            "--list-installed-traces",
            "--repo-root", TILEXR_SOURCE_ROOT,
        };
        tilexr::checker::CliOptions options;
        tilexr::checker::CheckerStatus status =
            tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                          argv, &options);

        ExpectEqInt(static_cast<int>(status.code),
                    static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                    "parse list installed traces status");
        ExpectTrue(options.list_installed_traces, "parse list installed traces flag");
        ExpectEqString(options.repo_root, TILEXR_SOURCE_ROOT,
                       "parse list installed traces repo root");
    }

    {
        const char *argv[] = {
            "tilexr_checker",
            "--verify-all-installed-traces",
            "--repo-root", TILEXR_SOURCE_ROOT,
            "--verification-report", "/tmp/tilexr-checker-all-installed.json",
        };
        tilexr::checker::CliOptions options;
        tilexr::checker::CheckerStatus status =
            tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                          argv, &options);

        ExpectEqInt(static_cast<int>(status.code),
                    static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                    "parse verify all installed traces status");
        ExpectTrue(options.verify_all_installed_traces,
                   "parse verify all installed traces flag");
        ExpectEqString(options.verification_report_output,
                       "/tmp/tilexr-checker-all-installed.json",
                       "parse verify all installed traces report");
    }
}

void TestParseCapabilityInventoryMode() {
    const char *argv[] = {
        "tilexr_checker",
        "--list-capabilities",
        "--repo-root", TILEXR_SOURCE_ROOT,
        "--capability-report", "/tmp/tilexr-checker-capabilities.json",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                      argv, &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "parse capability inventory status");
    ExpectTrue(options.list_capabilities, "parse capability inventory flag");
    ExpectEqString(options.capability_report_output,
                   "/tmp/tilexr-checker-capabilities.json",
                   "parse capability inventory report");
}

void TestParseAnalyzeTraceSourceMode() {
    const char *argv[] = {
        "tilexr_checker",
        "--analyze-trace-source",
        "--source-file", "src/collectives/kernels/allreduce_big_data.h",
        "--adapter-name", "allreduce_big_data",
        "--repo-root", "/tmp/tilexr-repo",
        "--trace-analysis-output", "/tmp/tilexr-trace-analysis.json",
        "--event-trace-template-output", "/tmp/tilexr-trace-template.jsonl",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                      argv, &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "analyze trace source parse status");
    ExpectTrue(options.analyze_trace_source, "analyze trace source flag");
    ExpectEqString(options.trace_adapter_spec.source_file,
                   "src/collectives/kernels/allreduce_big_data.h",
                   "analyze trace source source file");
    ExpectEqString(options.trace_adapter_spec.source_scan_file,
                   "/tmp/tilexr-repo/src/collectives/kernels/allreduce_big_data.h",
                   "analyze trace source scan path uses repo root");
    ExpectEqString(options.trace_adapter_spec.adapter_name, "allreduce_big_data",
                   "analyze trace source adapter name");
    ExpectEqString(options.repo_root, "/tmp/tilexr-repo",
                   "analyze trace source repo root");
    ExpectEqString(options.trace_analysis_output, "/tmp/tilexr-trace-analysis.json",
                   "analyze trace source output");
    ExpectEqString(options.event_trace_template_output,
                   "/tmp/tilexr-trace-template.jsonl",
                   "analyze trace source event template output");
}

void TestParseValidateEventTraceMode() {
    const char *argv[] = {
        "tilexr_checker",
        "--validate-event-trace",
        "--event-trace", "/tmp/tilexr-events.jsonl",
        "--output-dir", "/tmp/tilexr-event-trace-report",
        "--repo-root", "/tmp/tilexr-repo",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                      argv, &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "validate event trace parse status");
    ExpectTrue(options.validate_event_trace, "validate event trace flag");
    ExpectEqString(options.event_trace_input, "/tmp/tilexr-events.jsonl",
                   "validate event trace input path");
    ExpectEqString(options.output_dir, "/tmp/tilexr-event-trace-report",
                   "validate event trace output dir");
    ExpectEqString(options.repo_root, "/tmp/tilexr-repo",
                   "validate event trace repo root");
}

void TestParseValidateEventTraceRequiresInput() {
    const char *argv[] = {
        "tilexr_checker",
        "--validate-event-trace",
        "--output-dir", "/tmp/tilexr-event-trace-report",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                      argv, &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "validate event trace missing input parse status");
}

void TestParseGenerateTraceAdapterInfersNameAndTargetHeader() {
    const char *argv[] = {
        "tilexr_checker",
        "--generate-trace-adapter",
        "--source-file", "src/collectives/kernels/91093/allgather_hierarchy_double_ring.h",
        "--output", "/tmp/allgather_hdb_trace_shim.h",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "generate trace adapter inferred parse status");
    ExpectEqString(options.trace_adapter_spec.adapter_name,
                   "allgather_hierarchy_double_ring",
                   "generate inferred adapter name");
    ExpectEqString(options.trace_adapter_spec.target_header,
                   "91093/allgather_hierarchy_double_ring.h",
                   "generate inferred target header");
}

void TestParseGenerateTraceAdapterOutputDirDefaultsBundle() {
    const char *argv[] = {
        "tilexr_checker",
        "--generate-trace-adapter",
        "--source-file", "src/collectives/kernels/91093/allgather_hierarchy_double_ring.h",
        "--output-dir", "/tmp/tilexr-checker-generated",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "generate trace adapter output-dir bundle parse status");
    ExpectEqString(options.trace_adapter_output,
                   "/tmp/tilexr-checker-generated/allgather_hierarchy_double_ring_trace_shim.h",
                   "generate bundle adapter output");
    ExpectEqString(options.trace_adapter_manifest_output,
                   "/tmp/tilexr-checker-generated/allgather_hierarchy_double_ring_trace_manifest.json",
                   "generate bundle manifest output");
    ExpectEqString(options.trace_runner_output,
                   "/tmp/tilexr-checker-generated/allgather_hierarchy_double_ring_trace_runner.cpp",
                   "generate bundle runner output");
    ExpectEqString(options.trace_probe_output,
                   "/tmp/tilexr-checker-generated/allgather_hierarchy_double_ring_trace_probe.cpp",
                   "generate bundle probe output");
    ExpectEqString(options.trace_onboarding_output,
                   "/tmp/tilexr-checker-generated/allgather_hierarchy_double_ring_trace_onboarding.md",
                   "generate bundle onboarding output");
    ExpectEqString(options.trace_runner_spec.function_name,
                   "RunAllGatherHierarchyDoubleRingTrace",
                   "generate bundle runner function");
    ExpectEqString(options.trace_runner_spec.materializer_name,
                   "MaterializeAllGatherHierarchyDoubleRing",
                   "generate bundle runner materializer");
}

void TestParseScaffoldTraceBundleDefaultsAndReport() {
    const char *argv[] = {
        "tilexr_checker",
        "--scaffold-trace-bundle",
        "--source-file", "src/collectives/kernels/allreduce_big_data.h",
        "--output-dir", "/tmp/tilexr-checker-scaffold-arbd",
        "--repo-root", TILEXR_SOURCE_ROOT,
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "scaffold trace bundle parse status");
    ExpectTrue(options.scaffold_trace_bundle, "scaffold trace bundle flag");
    ExpectEqString(options.trace_adapter_spec.adapter_name,
                   "allreduce_big_data",
                   "scaffold inferred adapter name");
    ExpectEqString(options.trace_adapter_spec.target_header,
                   "allreduce_big_data.h",
                   "scaffold inferred target header");
    ExpectEqString(options.trace_adapter_output,
                   "/tmp/tilexr-checker-scaffold-arbd/allreduce_big_data_trace_shim.h",
                   "scaffold adapter output");
    ExpectEqString(options.trace_adapter_manifest_output,
                   "/tmp/tilexr-checker-scaffold-arbd/allreduce_big_data_trace_manifest.json",
                   "scaffold manifest output");
    ExpectEqString(options.trace_runner_output,
                   "/tmp/tilexr-checker-scaffold-arbd/allreduce_big_data_trace_runner.cpp",
                   "scaffold runner output");
    ExpectEqString(options.trace_probe_output,
                   "/tmp/tilexr-checker-scaffold-arbd/allreduce_big_data_trace_probe.cpp",
                   "scaffold probe output");
    ExpectEqString(options.trace_onboarding_output,
                   "/tmp/tilexr-checker-scaffold-arbd/allreduce_big_data_trace_onboarding.md",
                   "scaffold onboarding output");
    ExpectEqString(options.verification_report_output,
                   "/tmp/tilexr-checker-scaffold-arbd/verification.json",
                   "scaffold verification report output");
}

void TestParseScaffoldTraceBundleUsesRepoRootForSourceScan() {
    const char *argv[] = {
        "tilexr_checker",
        "--scaffold-trace-bundle",
        "--source-file", "src/collectives/kernels/demo.h",
        "--output-dir", "/tmp/tilexr-checker-scaffold-demo",
        "--repo-root", "/tmp/tilexr-checker-demo-repo",
    };

    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv,
                                      &options);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "scaffold source scan repo root parse status");
    ExpectEqString(options.trace_adapter_spec.source_file,
                   "src/collectives/kernels/demo.h",
                   "scaffold source scan keeps production source path");
    ExpectEqString(options.trace_adapter_spec.source_scan_file,
                   "/tmp/tilexr-checker-demo-repo/src/collectives/kernels/demo.h",
                   "scaffold source scan path uses repo root");
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

void TestRunCheckerCliWritesServerTopologyEvents() {
    tilexr::checker::CliOptions options = MakeValidAllGatherOptions();
    options.test_case.server_count = 2;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "run cli server topology exit code");
    ExpectTrue(stderr_stream.str().empty(), "run cli server topology stderr empty");
    const std::string events = ReadFile(options.output_dir + "/events.jsonl");
    ExpectTrue(events.find("\"server\":0") != std::string::npos,
               "run cli server topology event server");
    ExpectTrue(events.find("\"peer_server\":1") != std::string::npos,
               "run cli server topology event peer server");
}

void TestRunCheckerCliHelpShowsRepoRootForTraceBundleVerification() {
    tilexr::checker::CliOptions options;
    options.show_help = true;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "run cli help exit code");
    ExpectTrue(stderr_stream.str().empty(), "run cli help stderr empty");
    ExpectTrue(stdout_stream.str().find("--verify-trace-bundle") != std::string::npos,
               "run cli help verify bundle mode");
    ExpectTrue(stdout_stream.str().find("--scaffold-trace-bundle") != std::string::npos,
               "run cli help scaffold bundle mode");
    ExpectTrue(stdout_stream.str().find("--analyze-trace-source") != std::string::npos,
               "run cli help analyze trace source mode");
    ExpectTrue(stdout_stream.str().find("--validate-event-trace") != std::string::npos,
               "run cli help validate event trace mode");
    ExpectTrue(stdout_stream.str().find(
                   "--verify-trace-bundle --adapter-name <name> "
                   "--output-dir <generated-checker-dir> [--repo-root <repo>]") !=
                   std::string::npos,
               "run cli help verify bundle repo root");
}

void TestRunCheckerCliGenerateTraceAdapter() {
    tilexr::checker::CliOptions options;
    options.generate_trace_adapter = true;
    options.trace_adapter_spec.adapter_name = "alltoall_demo";
    const std::string source_path = MakeTempFile();
    {
        std::ofstream source(source_path.c_str(), std::ios::out | std::ios::trunc);
        source << "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n"
               << "DataCopy(dst, src, 32);\n";
    }
    options.trace_adapter_spec.source_file = source_path;
    options.trace_adapter_spec.target_header = "alltoall_demo.h";
    const std::string output_dir = MakeTempDir();
    options.trace_adapter_output = output_dir + "/alltoall_demo_trace_shim.h";
    options.trace_adapter_manifest_output =
        output_dir + "/alltoall_demo_trace_manifest.json";
    options.trace_runner_output = output_dir + "/alltoall_demo_trace_runner.cpp";
    options.trace_probe_output = output_dir + "/alltoall_demo_trace_probe.cpp";
    options.trace_onboarding_output = output_dir + "/alltoall_demo_trace_onboarding.md";
    options.trace_runner_spec.function_name = "RunAllToAllDemoTrace";
    options.trace_runner_spec.materializer_name = "MaterializeAllToAllDemo";
    options.trace_runner_spec.shim_include = "tilexr/checker/alltoall_demo_trace_shim.h";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "run cli generate adapter exit code");
    ExpectTrue(stderr_stream.str().empty(), "run cli generate adapter stderr empty");
    ExpectTrue(stdout_stream.str().find("generated trace adapter:") != std::string::npos,
               "run cli generate adapter stdout");
    const std::string output = ReadFile(options.trace_adapter_output);
    ExpectTrue(output.find("TILEXR_CHECKER_ALLTOALL_DEMO_TRACE_SHIM_H") != std::string::npos,
               "run cli generated adapter content");
    const std::string manifest = ReadFile(options.trace_adapter_manifest_output);
    ExpectTrue(manifest.find("\"runner_integration\"") != std::string::npos,
               "run cli generated adapter manifest");
    ExpectTrue(manifest.find("\"trace_hooks\"") != std::string::npos,
               "run cli generated adapter manifest trace hooks");
    ExpectTrue(manifest.find("\"validation_commands\"") != std::string::npos,
               "run cli generated adapter manifest validation commands");
    ExpectTrue(manifest.find("\"installed_trace_seed\"") != std::string::npos,
               "run cli generated adapter manifest installed seed");
    ExpectTrue(manifest.find("\"required_event_coverage\"") != std::string::npos,
               "run cli generated adapter manifest coverage seed");
    ExpectTrue(manifest.find("\"kind\":\"COPY\"") != std::string::npos,
               "run cli generated adapter manifest copy coverage seed");
    ExpectTrue(manifest.find("\"occurrences\"") != std::string::npos,
               "run cli generated adapter manifest occurrences");
    ExpectTrue(manifest.find("\"snippet\":\"Collectives::CpGM2GMPingPong(bytes, input, output, op);\"") !=
                   std::string::npos,
               "run cli generated adapter manifest gm2gm snippet");
    ExpectTrue(manifest.find("\"snippet\":\"DataCopy(dst, src, 32);\"") !=
                   std::string::npos,
               "run cli generated adapter manifest raw copy snippet");
    const std::string runner = ReadFile(options.trace_runner_output);
    ExpectTrue(runner.find("RunAllToAllDemoTrace") != std::string::npos,
               "run cli generated runner skeleton");
    const std::string probe = ReadFile(options.trace_probe_output);
    ExpectTrue(probe.find("trace_adapter_alltoall_demo::Metadata()") != std::string::npos,
               "run cli generated probe metadata");
    ExpectTrue(probe.find("trace_adapter_alltoall_demo::Audit(&reason)") !=
                   std::string::npos,
               "run cli generated probe audit");
    ExpectTrue(probe.find("std::strcmp(metadata.adapter_name, \"alltoall_demo\") != 0") !=
                   std::string::npos,
               "run cli generated probe adapter metadata check");
    ExpectTrue(probe.find("std::strcmp(metadata.source_file, \"" + source_path + "\") != 0") !=
                   std::string::npos,
               "run cli generated probe source metadata check");
    ExpectTrue(probe.find("std::strcmp(metadata.target_header, \"alltoall_demo.h\") != 0") !=
                   std::string::npos,
               "run cli generated probe target metadata check");
    const std::string onboarding = ReadFile(options.trace_onboarding_output);
    ExpectTrue(onboarding.find("CollectiveExecutor") != std::string::npos,
               "run cli generated onboarding plan executor");
    ExpectTrue(onboarding.find("git diff -- src/collectives src/ep") != std::string::npos,
               "run cli generated onboarding plan validation");
    ExpectTrue(onboarding.find("## Manual Review Required") != std::string::npos,
               "run cli generated onboarding manual review section");
    ExpectTrue(onboarding.find("DataCopy") != std::string::npos,
               "run cli generated onboarding manual review symbol");
    ExpectTrue(onboarding.find("status: pending") != std::string::npos,
               "run cli generated onboarding manual review pending status");
    ExpectTrue(stdout_stream.str().find("generated trace runner skeleton:") != std::string::npos,
               "run cli generated runner skeleton stdout");
    ExpectTrue(stdout_stream.str().find("generated trace header probe:") != std::string::npos,
               "run cli generated probe stdout");
    ExpectTrue(stdout_stream.str().find("generated trace onboarding plan:") != std::string::npos,
               "run cli generated onboarding plan stdout");
}

void TestRunCheckerCliAnalyzeTraceSourceWritesJsonReport() {
    const std::string dir = MakeTempDir();
    const std::string repo_root = dir + "/repo";
    const std::string kernel_dir = repo_root + "/src/collectives/kernels";
    const std::string mkdir_cmd = "mkdir -p " + kernel_dir;
    ExpectEqInt(std::system(mkdir_cmd.c_str()), 0,
                "analyze trace source mkdir command");
    {
        std::ofstream output((kernel_dir + "/new_collective.h").c_str(),
                             std::ios::out | std::ios::trunc);
        output << "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n"
               << "sync.SetSyncFlag(magic, value, event, rank);\n"
               << "sync.WaitSyncFlag(magic, value, event, rank);\n"
               << "AscendC::PipeBarrier<PIPE_ALL>();\n"
               << "DataCopy(dst, src, 32);\n";
    }
    const std::string report_path = dir + "/trace_analysis.json";
    const std::string template_path = dir + "/trace_template.jsonl";
    const char *argv[] = {
        "tilexr_checker",
        "--analyze-trace-source",
        "--source-file", "src/collectives/kernels/new_collective.h",
        "--adapter-name", "new_collective",
        "--repo-root", repo_root.c_str(),
        "--trace-analysis-output", report_path.c_str(),
        "--event-trace-template-output", template_path.c_str(),
    };
    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus parse_status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                      argv, &options);
    ExpectTrue(parse_status.ok(), "run cli analyze trace source parse status");

    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;
    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "run cli analyze trace source exit code");
    ExpectTrue(stderr_stream.str().empty(),
               "run cli analyze trace source stderr empty");
    ExpectTrue(stdout_stream.str().find("auto hook symbols: 4") !=
                   std::string::npos,
               "run cli analyze trace source stdout auto hook symbol count");
    ExpectTrue(stdout_stream.str().find("auto hook occurrences: 4") !=
                   std::string::npos,
               "run cli analyze trace source stdout auto hook occurrence count");
    ExpectTrue(stdout_stream.str().find("manual review symbols: 1") !=
                   std::string::npos,
               "run cli analyze trace source stdout manual symbol count");
    ExpectTrue(stdout_stream.str().find("manual review actions: 1") !=
                   std::string::npos,
               "run cli analyze trace source stdout manual action count");
    ExpectTrue(stdout_stream.str().find("analysis report: " + report_path) !=
                   std::string::npos,
               "run cli analyze trace source stdout report path");
    ExpectTrue(stdout_stream.str().find("event trace template: " + template_path) !=
                   std::string::npos,
               "run cli analyze trace source stdout template path");
    const std::string report = ReadFile(report_path);
    ExpectTrue(report.find("\"mode\":\"trace_source_analysis\"") != std::string::npos,
               "run cli analyze trace source report mode");
    ExpectTrue(report.find("\"semi_automatic_trace\"") != std::string::npos,
               "run cli analyze trace source semi automatic section");
    ExpectTrue(report.find("\"symbol\":\"CpGM2GMPingPong\"") != std::string::npos,
               "run cli analyze trace source gm2gm candidate");
    ExpectTrue(report.find("\"symbol\":\"AscendC::PipeBarrier\"") != std::string::npos,
               "run cli analyze trace source pipe barrier candidate");
    ExpectTrue(report.find(
                   "\"action\":\"add a trace wrapper or explicit unsupported gate for DataCopy\"") !=
                   std::string::npos,
               "run cli analyze trace source raw data copy action");
    ExpectTrue(report.find(
                   "\"source_file\":\"src/collectives/kernels/new_collective.h\"") !=
                   std::string::npos,
               "run cli analyze trace source production path preserved");

    const std::string event_template = ReadFile(template_path);
    ExpectTrue(event_template.find("\"id\":1") != std::string::npos,
               "run cli analyze trace source template first event id");
    ExpectTrue(event_template.find("\"kind\":\"COPY\"") != std::string::npos,
               "run cli analyze trace source template copy event");
    ExpectTrue(event_template.find("\"source_line\":1") != std::string::npos,
               "run cli analyze trace source template copy source line");
    ExpectTrue(event_template.find("\"kind\":\"FLAG_STORE\"") != std::string::npos,
               "run cli analyze trace source template flag store event");
    ExpectTrue(event_template.find("\"source_line\":2") != std::string::npos,
               "run cli analyze trace source template flag store source line");
    ExpectTrue(event_template.find("\"kind\":\"FLAG_WAIT\"") != std::string::npos,
               "run cli analyze trace source template flag wait event");
    ExpectTrue(event_template.find("\"source_line\":3") != std::string::npos,
               "run cli analyze trace source template flag wait source line");
    ExpectTrue(event_template.find("\"kind\":\"PIPE_BARRIER\"") != std::string::npos,
               "run cli analyze trace source template pipe event");
    ExpectTrue(event_template.find("\"source_line\":4") != std::string::npos,
               "run cli analyze trace source template pipe source line");
    ExpectTrue(event_template.find("DataCopy") == std::string::npos,
               "run cli analyze trace source template skips manual review primitives");

    tilexr::checker::CliOptions validate_options;
    validate_options.validate_event_trace = true;
    validate_options.event_trace_input = template_path;
    validate_options.output_dir = dir + "/trace_template_report";
    validate_options.repo_root = repo_root;

    std::ostringstream validate_stdout;
    std::ostringstream validate_stderr;
    const int validate_exit =
        tilexr::checker::RunCheckerCli(validate_options, &validate_stdout,
                                       &validate_stderr);
    ExpectEqInt(validate_exit, 0,
                "run cli analyze trace source template validates");
    ExpectTrue(validate_stderr.str().empty(),
               "run cli analyze trace source template validate stderr empty");
    ExpectTrue(ReadFile(validate_options.output_dir + "/summary.txt").find(
                   "checker: PASS") != std::string::npos,
               "run cli analyze trace source template validate summary");
}

void TestRunCheckerCliScaffoldTraceBundleWritesVerificationReport() {
    const std::string output_dir = MakeTempDir();
    const std::string source_path = output_dir + "/demo_trace_target.h";
    {
        std::ofstream source(source_path.c_str(), std::ios::out | std::ios::trunc);
        source << "#pragma once\n"
               << "class DemoTraceTarget {\n"
               << "public:\n"
               << "    void Process() { AscendC::PipeBarrier<PIPE_ALL>(); }\n"
               << "};\n";
    }

    tilexr::checker::CliOptions options;
    options.scaffold_trace_bundle = true;
    options.trace_adapter_spec.adapter_name = "demo";
    options.trace_adapter_spec.source_file = source_path;
    options.trace_adapter_spec.target_header = "demo_trace_target.h";
    options.output_dir = output_dir;
    options.repo_root = TILEXR_SOURCE_ROOT;
    options.trace_adapter_output = output_dir + "/demo_trace_shim.h";
    options.trace_adapter_manifest_output = output_dir + "/demo_trace_manifest.json";
    options.trace_runner_output = output_dir + "/demo_trace_runner.cpp";
    options.trace_probe_output = output_dir + "/demo_trace_probe.cpp";
    options.trace_onboarding_output = output_dir + "/demo_trace_onboarding.md";
    options.trace_runner_spec.function_name = "RunDemoTrace";
    options.trace_runner_spec.materializer_name = "MaterializeDemo";
    options.trace_runner_spec.shim_include = "tilexr/checker/demo_trace_shim.h";
    options.verification_report_output = output_dir + "/verification.json";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "scaffold trace bundle exit code");
    ExpectTrue(stderr_stream.str().empty(), "scaffold trace bundle stderr empty");
    ExpectTrue(FileExists(options.trace_adapter_output),
               "scaffold trace bundle writes shim");
    ExpectTrue(FileExists(options.trace_adapter_manifest_output),
               "scaffold trace bundle writes manifest");
    ExpectTrue(FileExists(options.trace_runner_output),
               "scaffold trace bundle writes runner");
    ExpectTrue(FileExists(options.trace_probe_output),
               "scaffold trace bundle writes probe");
    ExpectTrue(FileExists(options.trace_onboarding_output),
               "scaffold trace bundle writes onboarding");
    ExpectTrue(FileExists(options.verification_report_output),
               "scaffold trace bundle writes verification report");
    ExpectTrue(stdout_stream.str().find("scaffold verification: incomplete") !=
                   std::string::npos,
               "scaffold trace bundle stdout verification state");
    ExpectTrue(stdout_stream.str().find("probe_compile: PASS") != std::string::npos,
               "scaffold trace bundle stdout probe compile");
    ExpectTrue(stdout_stream.str().find("generated trace onboarding plan:") !=
                   std::string::npos,
               "scaffold trace bundle stdout onboarding");
    const std::string report = ReadFile(options.verification_report_output);
    ExpectTrue(report.find("\"mode\":\"trace_bundle_scaffold\"") != std::string::npos,
               "scaffold trace bundle report mode");
    ExpectTrue(report.find("\"complete\":false") != std::string::npos,
               "scaffold trace bundle report incomplete");
    ExpectTrue(report.find("\"probe_compile\":\"PASS\"") != std::string::npos,
               "scaffold trace bundle report probe");
    ExpectTrue(report.find("\"unchecked_items\"") != std::string::npos,
               "scaffold trace bundle report unchecked items");
}

void TestRunCheckerCliScaffoldTraceBundleScansRelativeSourceFromRepoRoot() {
    const std::string repo_root = MakeTempDir();
    mkdir((repo_root + "/src").c_str(), 0700);
    mkdir((repo_root + "/src/collectives").c_str(), 0700);
    mkdir((repo_root + "/src/collectives/kernels").c_str(), 0700);
    {
        std::ofstream source((repo_root + "/src/collectives/kernels/demo.h").c_str(),
                             std::ios::out | std::ios::trunc);
        source << "#pragma once\n"
               << "inline void DemoTraceHook() {\n"
               << "    CpGM2GMPingPong(bytes, input, output, op);\n"
               << "}\n";
    }
    const std::string output_dir = MakeTempDir();

    const char *argv[] = {
        "tilexr_checker",
        "--scaffold-trace-bundle",
        "--adapter-name", "demo",
        "--source-file", "src/collectives/kernels/demo.h",
        "--target-header", "demo.h",
        "--output-dir", output_dir.c_str(),
        "--repo-root", repo_root.c_str(),
    };
    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus parse_status =
        tilexr::checker::ParseCliArgs(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                                      argv, &options);
    ExpectTrue(parse_status.ok(), "scaffold relative source parse status");
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "scaffold relative source exit code");
    ExpectTrue(stderr_stream.str().empty(), "scaffold relative source stderr empty");
    const std::string manifest = ReadFile(options.trace_adapter_manifest_output);
    ExpectTrue(manifest.find("\"source_file\":\"src/collectives/kernels/demo.h\"") !=
                   std::string::npos,
               "scaffold relative source manifest keeps production source");
    ExpectTrue(manifest.find("\"source_observation\":{\"available\":true") !=
                   std::string::npos,
               "scaffold relative source manifest source available");
    ExpectTrue(manifest.find("\"symbol\":\"CpGM2GMPingPong\"") != std::string::npos,
               "scaffold relative source manifest hook observed");
    ExpectTrue(manifest.find("\"symbol\":\"source_unavailable\"") == std::string::npos,
               "scaffold relative source manifest no unavailable action");
}

void TestRunCheckerCliGenerateTraceAdapterRejectsProductionOutput() {
    tilexr::checker::CliOptions options;
    options.generate_trace_adapter = true;
    options.trace_adapter_spec.adapter_name = "bad_demo";
    options.trace_adapter_spec.source_file = "src/collectives/kernels/bad_demo.h";
    options.trace_adapter_spec.target_header = "bad_demo.h";
    options.trace_adapter_output = "src/collectives/kernels/bad_demo_trace_shim.h";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 2, "run cli reject production adapter output exit code");
    ExpectTrue(stderr_stream.str().find("checker output path") != std::string::npos,
               "run cli reject production adapter output stderr");
}

void TestRunCheckerCliStrictGenerateTraceAdapterRejectsRawDataCopy() {
    tilexr::checker::CliOptions options;
    options.generate_trace_adapter = true;
    options.trace_adapter_spec.adapter_name = "raw_copy_demo";
    const std::string source_path = MakeTempFile();
    {
        std::ofstream source(source_path.c_str(), std::ios::out | std::ios::trunc);
        source << "DataCopy(dst, src, 32);\n";
    }
    options.trace_adapter_spec.source_file = source_path;
    options.trace_adapter_spec.target_header = "raw_copy_demo.h";
    options.trace_adapter_spec.strict_source_observation = true;
    const std::string output_dir = MakeTempDir();
    options.trace_adapter_output = output_dir + "/raw_copy_demo_trace_shim.h";
    options.trace_adapter_manifest_output =
        output_dir + "/raw_copy_demo_trace_manifest.json";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 2, "run cli strict raw copy exit code");
    ExpectTrue(stderr_stream.str().find("manual review candidates") != std::string::npos,
               "run cli strict raw copy stderr");
    ExpectTrue(stderr_stream.str().find("DataCopy") != std::string::npos,
               "run cli strict raw copy stderr symbol");
    ExpectTrue(!FileExists(options.trace_adapter_output),
               "run cli strict raw copy should not leave adapter output");
    ExpectTrue(!FileExists(options.trace_adapter_manifest_output),
               "run cli strict raw copy should not leave manifest output");
}

void TestRunCheckerCliVerifyTraceBundleReportsUncheckedItems() {
    const std::string output_dir = MakeTempDir();
    std::ofstream(output_dir + "/demo_trace_shim.h") << "shim\n";
    std::ofstream(output_dir + "/demo_trace_manifest.json")
        << "{\"source_preservation\":{},\"runner_integration\":{\"checklist\":[]}}\n";
    std::ofstream(output_dir + "/demo_trace_runner.cpp")
        << "bool block_schedule_defined = false;\n"
        << "bool operator_init_process_wired = false;\n"
        << "bool oracle_materializer_reviewed = false;\n";
    std::ofstream(output_dir + "/demo_trace_onboarding.md")
        << "Do not edit production sources.\nCollectiveExecutor\n";

    tilexr::checker::CliOptions options;
    options.verify_trace_bundle = true;
    options.trace_adapter_spec.adapter_name = "demo";
    options.output_dir = output_dir;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 2, "verify trace bundle unchecked exit code");
    ExpectTrue(stdout_stream.str().find("unchecked_items") != std::string::npos,
               "verify trace bundle unchecked stdout");
    ExpectTrue(stdout_stream.str().find("block_schedule_defined") != std::string::npos,
               "verify trace bundle block schedule stdout");
    ExpectTrue(stderr_stream.str().empty(), "verify trace bundle stderr empty");
}

void TestRunCheckerCliVerifyTraceBundleWritesFailureJsonReport() {
    const std::string output_dir = MakeTempDir();
    std::ofstream(output_dir + "/demo_trace_shim.h")
        << "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
        << "namespace tilexr { namespace checker { namespace trace_adapter_demo {\n"
        << "inline const TraceAdapterMetadata &Metadata();\n"
        << "inline bool Audit(const char **reason);\n"
        << "}}}\n";
    std::ofstream(output_dir + "/demo_trace_manifest.json")
        << "{\"metadata_api\":{},\"source_preservation\":{},"
        << "\"runner_integration\":{\"checklist\":[]},\"required_actions\":[]}\n";
    std::ofstream(output_dir + "/demo_trace_runner.cpp")
        << "bool block_schedule_defined = false;\n"
        << "bool operator_init_process_wired = false;\n"
        << "bool oracle_materializer_reviewed = false;\n";
    std::ofstream(output_dir + "/demo_trace_probe.cpp")
        << "trace_adapter_demo::Metadata();\n"
        << "trace_adapter_demo::Audit(&reason);\n";
    std::ofstream(output_dir + "/demo_trace_onboarding.md")
        << "Do not edit production sources.\nCollectiveExecutor\n";

    tilexr::checker::CliOptions options;
    options.verify_trace_bundle = true;
    options.trace_adapter_spec.adapter_name = "demo";
    options.output_dir = output_dir;
    options.verification_report_output = output_dir + "/verification.json";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 2, "verify trace bundle failure json exit code");
    ExpectTrue(stderr_stream.str().empty(), "verify trace bundle failure json stderr");
    const std::string report = ReadFile(options.verification_report_output);
    ExpectTrue(report.find("\"mode\":\"trace_bundle\"") != std::string::npos,
               "verify bundle failure json mode");
    ExpectTrue(report.find("\"complete\":false") != std::string::npos,
               "verify bundle failure json incomplete");
    ExpectTrue(report.find("\"status_code\":\"UNSUPPORTED\"") != std::string::npos,
               "verify bundle failure json status code");
    ExpectTrue(report.find("\"status_message\":\"trace probe compile/run failed\"") !=
                   std::string::npos,
               "verify bundle failure json status message");
    ExpectTrue(report.find("\"probe_compile\":\"FAIL\"") != std::string::npos,
               "verify bundle failure json probe status");
    ExpectTrue(report.find("probe_compile:") != std::string::npos,
               "verify bundle failure json probe log");
    ExpectTrue(report.find("\"unchecked_items\":[\"block_schedule_defined\"") !=
                   std::string::npos,
               "verify bundle failure json unchecked item");
}

void TestRunCheckerCliVerifyTraceBundleCompilesGeneratedProbe() {
    const std::string output_dir = MakeTempDir();
    const std::string source_path = output_dir + "/demo_trace_target.h";
    {
        std::ofstream source(source_path.c_str(), std::ios::out | std::ios::trunc);
        source << "#pragma once\n"
               << "class DemoTraceTarget {\n"
               << "public:\n"
               << "    void Process() { AscendC::PipeBarrier<PIPE_ALL>(); }\n"
               << "};\n";
    }

    tilexr::checker::CliOptions generate;
    generate.generate_trace_adapter = true;
    generate.trace_adapter_spec.adapter_name = "demo";
    generate.trace_adapter_spec.source_file = source_path;
    generate.trace_adapter_spec.target_header = "demo_trace_target.h";
    generate.output_dir = output_dir;
    generate.trace_adapter_output = output_dir + "/demo_trace_shim.h";
    generate.trace_adapter_manifest_output = output_dir + "/demo_trace_manifest.json";
    generate.trace_runner_output = output_dir + "/demo_trace_runner.cpp";
    generate.trace_probe_output = output_dir + "/demo_trace_probe.cpp";
    generate.trace_onboarding_output = output_dir + "/demo_trace_onboarding.md";
    generate.trace_runner_spec.function_name = "RunDemoTrace";
    generate.trace_runner_spec.materializer_name = "MaterializeDemo";
    generate.trace_runner_spec.shim_include = "tilexr/checker/demo_trace_shim.h";
    std::ostringstream generate_stdout;
    std::ostringstream generate_stderr;
    const int generate_exit =
        tilexr::checker::RunCheckerCli(generate, &generate_stdout, &generate_stderr);
    ExpectEqInt(generate_exit, 0, "generate bundle for verify probe exit code");

    {
        std::ofstream runner(output_dir + "/demo_trace_runner.cpp",
                             std::ios::out | std::ios::trunc);
        runner << "struct RunnerIntegrationChecklist {\n"
               << "  bool block_schedule_defined = true;\n"
               << "  bool operator_init_process_wired = true;\n"
               << "  bool oracle_materializer_reviewed = true;\n"
               << "};\n";
    }

    tilexr::checker::CliOptions verify;
    verify.verify_trace_bundle = true;
    verify.trace_adapter_spec.adapter_name = "demo";
    verify.output_dir = output_dir;
    verify.repo_root = TILEXR_SOURCE_ROOT;
    std::ostringstream verify_stdout;
    std::ostringstream verify_stderr;

    const int verify_exit =
        tilexr::checker::RunCheckerCli(verify, &verify_stdout, &verify_stderr);

    ExpectEqInt(verify_exit, 0, "verify trace bundle compiled probe exit code");
    ExpectTrue(verify_stdout.str().find("probe_compile: PASS") != std::string::npos,
               "verify trace bundle compiled probe stdout");
    ExpectTrue(verify_stderr.str().empty(), "verify trace bundle compiled probe stderr");
}

void TestRunCheckerCliVerifyInstalledTraceAlgorithm() {
    tilexr::checker::CliOptions options;
    options.verify_installed_trace = true;
    options.trace_adapter_spec.adapter_name = "allreduce_big_data";
    options.repo_root = TILEXR_SOURCE_ROOT;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "verify installed trace exit code");
    ExpectTrue(stdout_stream.str().find("installed algorithm verification: complete") !=
                   std::string::npos,
               "verify installed trace complete stdout");
    ExpectTrue(stdout_stream.str().find("RunAllReduceBigDataTrace") != std::string::npos,
               "verify installed trace runner stdout");
    ExpectTrue(stdout_stream.str().find("manifest: tools/checker/installed_traces/"
                                        "allreduce_big_data_trace_manifest.json") !=
                   std::string::npos,
               "verify installed trace manifest stdout");
    ExpectTrue(stdout_stream.str().find("runtime_smoke: PASS") != std::string::npos,
               "verify installed trace smoke stdout");
    ExpectTrue(stdout_stream.str().find("runtime_smoke_cases: 2") != std::string::npos,
               "verify installed trace smoke cases stdout");
    ExpectTrue(stderr_stream.str().empty(), "verify installed trace stderr empty");
}

void TestRunCheckerCliVerifyInstalledTraceWritesJsonReport() {
    const std::string output_dir = MakeTempDir();
    tilexr::checker::CliOptions options;
    options.verify_installed_trace = true;
    options.trace_adapter_spec.adapter_name = "allreduce_big_data";
    options.repo_root = TILEXR_SOURCE_ROOT;
    options.verification_report_output = output_dir + "/installed_report.json";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "verify installed trace json report exit code");
    ExpectTrue(stderr_stream.str().empty(),
               "verify installed trace json report stderr empty");
    const std::string report = ReadFile(options.verification_report_output);
    ExpectTrue(report.find("\"mode\":\"installed_trace\"") != std::string::npos,
               "verify installed json report mode");
    ExpectTrue(report.find("\"adapter_name\":\"allreduce_big_data\"") != std::string::npos,
               "verify installed json report adapter");
    ExpectTrue(report.find("\"complete\":true") != std::string::npos,
               "verify installed json report complete");
    ExpectTrue(report.find("\"status_code\":\"OK\"") != std::string::npos,
               "verify installed json report status code");
    ExpectTrue(report.find("\"status_message\":\"OK\"") != std::string::npos,
               "verify installed json report status message");
    ExpectTrue(report.find("\"probe_compile\":\"PASS\"") != std::string::npos,
               "verify installed json report probe");
    ExpectTrue(report.find("\"runtime_smoke\":\"PASS\"") != std::string::npos,
               "verify installed json report smoke");
    ExpectTrue(report.find("\"runtime_events\":2710") != std::string::npos,
               "verify installed json report events");
}

void TestRunCheckerCliVerifyInstalledHdbTraceAlgorithm() {
    tilexr::checker::CliOptions options;
    options.verify_installed_trace = true;
    options.trace_adapter_spec.adapter_name = "allgather_hierarchy_double_ring";
    options.repo_root = TILEXR_SOURCE_ROOT;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "verify installed hdb trace exit code");
    ExpectTrue(stdout_stream.str().find("installed algorithm verification: complete") !=
                   std::string::npos,
               "verify installed hdb complete stdout");
    ExpectTrue(stdout_stream.str().find("RunAllGatherHierarchyDoubleRingTrace") !=
                   std::string::npos,
               "verify installed hdb runner stdout");
    ExpectTrue(stdout_stream.str().find("manifest: tools/checker/installed_traces/"
                                        "allgather_hierarchy_double_ring_trace_manifest.json") !=
                   std::string::npos,
               "verify installed hdb manifest stdout");
    ExpectTrue(stdout_stream.str().find("runtime_smoke: PASS") != std::string::npos,
               "verify installed hdb smoke stdout");
    ExpectTrue(stdout_stream.str().find("runtime_smoke_cases: 2") != std::string::npos,
               "verify installed hdb smoke cases stdout");
    ExpectTrue(stderr_stream.str().empty(), "verify installed hdb stderr empty");
}

void TestRunCheckerCliListInstalledTraces() {
    tilexr::checker::CliOptions options;
    options.list_installed_traces = true;
    options.repo_root = TILEXR_SOURCE_ROOT;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "list installed traces exit code");
    ExpectTrue(stderr_stream.str().empty(), "list installed traces stderr");
    ExpectTrue(stdout_stream.str().find("allreduce_big_data") != std::string::npos,
               "list installed traces allreduce");
    ExpectTrue(stdout_stream.str().find("allgather_hierarchy_double_ring") !=
                   std::string::npos,
               "list installed traces hdb");
    ExpectTrue(stdout_stream.str().find("ep_dispatch") != std::string::npos,
               "list installed traces ep dispatch");
}

void TestRunCheckerCliVerifyAllInstalledTracesWritesJsonReport() {
    const std::string output_dir = MakeTempDir();
    tilexr::checker::CliOptions options;
    options.verify_all_installed_traces = true;
    options.repo_root = TILEXR_SOURCE_ROOT;
    options.verification_report_output = output_dir + "/all_installed_report.json";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "verify all installed traces exit code");
    ExpectTrue(stderr_stream.str().empty(), "verify all installed traces stderr");
    ExpectTrue(stdout_stream.str().find("allreduce_big_data: complete") !=
                   std::string::npos,
               "verify all installed traces stdout allreduce");
    ExpectTrue(stdout_stream.str().find("allgather_hierarchy_double_ring: complete") !=
                   std::string::npos,
               "verify all installed traces stdout hdb");
    ExpectTrue(stdout_stream.str().find("ep_dispatch: complete") != std::string::npos,
               "verify all installed traces stdout ep dispatch");
    const std::string report = ReadFile(options.verification_report_output);
    ExpectTrue(report.find("\"mode\":\"installed_trace_inventory\"") !=
                   std::string::npos,
               "verify all installed traces json mode");
    ExpectTrue(report.find("\"overall_status\":\"OK\"") != std::string::npos,
               "verify all installed traces json overall");
    ExpectTrue(report.find("\"adapter_name\":\"allreduce_big_data\"") !=
                   std::string::npos,
               "verify all installed traces json allreduce");
    ExpectTrue(report.find("\"adapter_name\":\"allgather_hierarchy_double_ring\"") !=
                   std::string::npos,
               "verify all installed traces json hdb");
    ExpectTrue(report.find("\"adapter_name\":\"ep_dispatch\"") !=
                   std::string::npos,
               "verify all installed traces json ep dispatch");
    ExpectTrue(report.find("\"runtime_smoke_cases\":2") !=
                   std::string::npos,
               "verify all installed traces json ep dispatch smoke cases");
}

void TestRunCheckerCliListCapabilitiesWritesJsonReport() {
    const std::string output_dir = MakeTempDir();
    tilexr::checker::CliOptions options;
    options.list_capabilities = true;
    options.repo_root = TILEXR_SOURCE_ROOT;
    options.capability_report_output = output_dir + "/capabilities.json";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "list capabilities exit code");
    ExpectTrue(stderr_stream.str().empty(), "list capabilities stderr");
    ExpectTrue(stdout_stream.str().find("allgather/default: supported") !=
                   std::string::npos,
               "list capabilities stdout generic allgather");
    ExpectTrue(stdout_stream.str().find("allreduce/allreduce_big_data: installed") !=
                   std::string::npos,
               "list capabilities stdout allreduce big data");
    ExpectTrue(stdout_stream.str().find("allgather/allgather_hierarchy_double_ring: installed") !=
                   std::string::npos,
               "list capabilities stdout hdb");
    ExpectTrue(stdout_stream.str().find("ep_dispatch/default: installed") !=
                   std::string::npos,
               "list capabilities stdout ep dispatch");
    ExpectTrue(stdout_stream.str().find("ep_combine/default: supported") !=
                   std::string::npos,
               "list capabilities stdout ep combine");
    ExpectTrue(FileExists(options.capability_report_output),
               "list capabilities report exists");
    const std::string report = ReadFile(options.capability_report_output);
    ExpectTrue(report.find("\"mode\":\"capability_inventory\"") != std::string::npos,
               "capability report mode");
    ExpectTrue(report.find("\"instruction_level_execution\":false") !=
                   std::string::npos,
               "capability report boundary");
    ExpectTrue(report.find("\"name\":\"allreduce_big_data\"") != std::string::npos,
               "capability report allreduce installed");
    ExpectTrue(report.find("\"name\":\"allgather_hierarchy_double_ring\"") !=
                   std::string::npos,
               "capability report hdb installed");
    ExpectTrue(report.find("\"name\":\"ep_dispatch\"") != std::string::npos,
               "capability report ep dispatch installed");
    ExpectTrue(report.find("\"name\":\"ep_combine\"") != std::string::npos,
               "capability report ep combine");
    ExpectTrue(report.find("\"status\":\"supported\"") != std::string::npos,
               "capability report ep combine supported");
    ExpectTrue(report.find("\"mode\":\"source_aligned_cpu_oracle\"") != std::string::npos,
               "capability report ep combine oracle");
    ExpectTrue(report.find("src/ep/kernels/tilexr_ep_combine_kernel.cpp") !=
                   std::string::npos,
               "capability report ep combine source");
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

void TestRunCheckerCliInjectedPipeFindingExitCode() {
    tilexr::checker::CliOptions options = MakeValidAllGatherOptions();
    options.inject_pipe_wait = true;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 1, "run cli injected pipe finding exit code");
    ExpectTrue(stdout_stream.str().find("PIPE_WAIT_NO_PRODUCER") != std::string::npos,
               "run cli injected pipe finding stdout");
}

void TestRunCheckerCliInjectedEpWindowReadWritesSourceLocatedReport() {
    tilexr::checker::CliOptions options;
    options.test_case.op = tilexr::checker::CollectiveOp::kEpDispatch;
    options.test_case.rank_size = 2;
    options.test_case.server_count = 2;
    options.test_case.bs = 3;
    options.test_case.h = 4;
    options.test_case.top_k = 2;
    options.test_case.moe_expert_num = 4;
    options.test_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
    options.test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    options.test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    options.test_case.magic = 0x1234;
    options.output_dir = MakeTempDir();
    options.inject_ep_window_read = true;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 1, "run cli injected ep window read exit code");
    ExpectTrue(stderr_stream.str().empty(), "run cli injected ep window read stderr empty");
    ExpectTrue(stdout_stream.str().find("READ_BEFORE_WRITE") != std::string::npos,
               "run cli injected ep window read stdout finding");
    const std::string summary = ReadFile(options.output_dir + "/summary.txt");
    const std::string report = ReadFile(options.output_dir + "/checker_report.json");
    const std::string events = ReadFile(options.output_dir + "/events.jsonl");
    ExpectTrue(summary.find("source: src/ep/kernels/tilexr_ep_combine_kernel.cpp:211") !=
                   std::string::npos,
               "run cli injected ep window read summary source");
    ExpectTrue(summary.find("buffer: RegisteredCommBuffer") != std::string::npos,
               "run cli injected ep window read summary buffer");
    ExpectTrue(summary.find("slot: 99") != std::string::npos,
               "run cli injected ep window read summary slot");
    ExpectTrue(summary.find("offset: 999999") != std::string::npos,
               "run cli injected ep window read summary offset");
    ExpectTrue(report.find("\"kind\":\"READ_BEFORE_WRITE\"") != std::string::npos,
               "run cli injected ep window read report kind");
    ExpectTrue(report.find("\"buffer_role\":\"RegisteredCommBuffer\"") != std::string::npos,
               "run cli injected ep window read report role");
    ExpectTrue(report.find("\"source_excerpt\":[]") != std::string::npos,
               "run cli injected ep window read missing source excerpt empty");
    ExpectTrue(events.find("cli injected ep window read without producer") != std::string::npos,
               "run cli injected ep window read event detail");
}

void TestRunCheckerCliResolvesSourceExcerptFromRepoRoot() {
    const std::string repo_root = MakeTempDir();
    const std::string source_path = repo_root + "/src/ep/kernels";
    mkdir((repo_root + "/src").c_str(), 0700);
    mkdir((repo_root + "/src/ep").c_str(), 0700);
    mkdir(source_path.c_str(), 0700);
    {
        std::ofstream source((source_path + "/tilexr_ep_combine_kernel.cpp").c_str(),
                             std::ios::out | std::ios::trunc);
        for (int line = 1; line <= 210; ++line) {
            source << "combine filler " << line << "\n";
        }
        source << "combine source line 211 target\n"
               << "combine source line 212 context\n";
    }

    tilexr::checker::CliOptions options;
    options.test_case.op = tilexr::checker::CollectiveOp::kEpDispatch;
    options.test_case.rank_size = 2;
    options.test_case.server_count = 2;
    options.test_case.bs = 3;
    options.test_case.h = 4;
    options.test_case.top_k = 2;
    options.test_case.moe_expert_num = 4;
    options.test_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
    options.test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    options.test_case.magic = 0x1234;
    options.output_dir = MakeTempDir();
    options.repo_root = repo_root;
    options.inject_ep_window_read = true;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 1, "run cli repo-root source excerpt exit code");
    ExpectTrue(stderr_stream.str().empty(),
               "run cli repo-root source excerpt stderr empty");
    const std::string report = ReadFile(options.output_dir + "/checker_report.json");
    ExpectTrue(report.find("\"source_excerpt\":[") != std::string::npos,
               "run cli repo-root source excerpt field");
    ExpectTrue(report.find("combine source line 211 target") != std::string::npos,
               "run cli repo-root source excerpt target text");
}

void TestRunCheckerCliUnsupportedExecutorStillWritesReports() {
    tilexr::checker::CliOptions options = MakeValidAllGatherOptions();
    options.test_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 2, "run cli unsupported executor exit code");
    ExpectTrue(stdout_stream.str().find("checker: UNSUPPORTED") != std::string::npos,
               "run cli unsupported stdout status");
    ExpectTrue(stderr_stream.str().empty(), "run cli unsupported stderr empty");
    const std::string summary = ReadFile(options.output_dir + "/summary.txt");
    const std::string findings = ReadFile(options.output_dir + "/findings.json");
    const std::string report = ReadFile(options.output_dir + "/checker_report.json");
    ExpectTrue(summary.find("checker: UNSUPPORTED") != std::string::npos,
               "run cli unsupported summary status");
    ExpectTrue(findings.find("UNSUPPORTED_API") != std::string::npos,
               "run cli unsupported findings kind");
    ExpectTrue(report.find("\"status\":\"UNSUPPORTED\"") != std::string::npos,
               "run cli unsupported checker report status");
}

void TestRunCheckerCliEpCombineWritesSourceLocatedReports() {
    tilexr::checker::CliOptions options;
    options.test_case.op = tilexr::checker::CollectiveOp::kEpCombine;
    options.test_case.rank_size = 2;
    options.test_case.server_count = 2;
    options.test_case.bs = 3;
    options.test_case.h = 4;
    options.test_case.top_k = 2;
    options.test_case.moe_expert_num = 4;
    options.test_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
    options.test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    options.test_case.magic = 0x1234;
    options.output_dir = MakeTempDir();
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 0, "run cli ep combine exit code");
    ExpectTrue(stderr_stream.str().empty(), "run cli ep combine stderr");
    ExpectTrue(stdout_stream.str().find("checker: PASS") != std::string::npos,
               "run cli ep combine stdout");
    const std::string summary = ReadFile(options.output_dir + "/summary.txt");
    const std::string events = ReadFile(options.output_dir + "/events.jsonl");
    const std::string report = ReadFile(options.output_dir + "/checker_report.json");
    ExpectTrue(summary.find("checker: PASS") !=
                   std::string::npos,
               "run cli ep combine summary pass");
    ExpectTrue(summary.find("source: src/ep/kernels/tilexr_ep_combine_kernel.cpp") !=
                   std::string::npos,
               "run cli ep combine summary source");
    ExpectTrue(events.find("\"source_file\":\"src/ep/kernels/tilexr_ep_combine_kernel.cpp\"") !=
                   std::string::npos,
               "run cli ep combine events source");
    ExpectTrue(events.find("ep combine read src slot header") !=
                   std::string::npos,
               "run cli ep combine events window read");
    ExpectTrue(events.find("ep combine write restored payload") !=
                   std::string::npos,
               "run cli ep combine events output write");
    ExpectTrue(report.find("\"status\":\"PASS\"") !=
                   std::string::npos,
               "run cli ep combine report status");
    ExpectTrue(report.find("\"source_file\":\"src/ep/kernels/tilexr_ep_combine_kernel.cpp\"") !=
                   std::string::npos,
               "run cli ep combine report source");
}

void TestRunCheckerCliValidateEventTraceWritesSourceLocatedReport() {
    const std::string trace_path = MakeTempFile();
    WriteFile(trace_path,
              "{\"id\":42,\"kind\":\"READ\",\"rank\":0,\"peer_rank\":1,"
              "\"server\":0,\"peer_server\":1,\"core\":2,\"pipe\":-1,"
              "\"event_id\":-1,\"buffer_role\":\"RegisteredCommBuffer\","
              "\"slot\":3,\"magic\":0,\"offset\":128,\"bytes\":64,"
              "\"allow_future_producer\":false,"
              "\"source_file\":\"src/ep/kernels/tilexr_ep_combine_kernel.cpp\","
              "\"source_line\":211,\"detail\":\"external combine read without producer\"}\n");

    tilexr::checker::CliOptions options;
    options.validate_event_trace = true;
    options.event_trace_input = trace_path;
    options.output_dir = MakeTempDir();
    options.repo_root = ".";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 1, "run cli validate event trace exit code");
    ExpectTrue(stderr_stream.str().empty(), "run cli validate event trace stderr");
    ExpectTrue(stdout_stream.str().find("READ_BEFORE_WRITE") != std::string::npos,
               "run cli validate event trace stdout finding");
    const std::string summary = ReadFile(options.output_dir + "/summary.txt");
    const std::string findings = ReadFile(options.output_dir + "/findings.json");
    const std::string events = ReadFile(options.output_dir + "/events.jsonl");
    const std::string report = ReadFile(options.output_dir + "/checker_report.json");
    ExpectTrue(summary.find("checker: FAIL") != std::string::npos,
               "run cli validate event trace summary fail");
    ExpectTrue(summary.find("source: src/ep/kernels/tilexr_ep_combine_kernel.cpp:211") !=
                   std::string::npos,
               "run cli validate event trace summary source");
    ExpectTrue(summary.find("slot: 3") != std::string::npos,
               "run cli validate event trace summary slot");
    ExpectTrue(summary.find("offset: 128") != std::string::npos,
               "run cli validate event trace summary offset");
    ExpectTrue(summary.find("event id: 42") != std::string::npos,
               "run cli validate event trace preserves input event id");
    ExpectTrue(findings.find("\"kind\":\"READ_BEFORE_WRITE\"") != std::string::npos,
               "run cli validate event trace findings kind");
    ExpectTrue(findings.find("\"event_log_id\":42") != std::string::npos,
               "run cli validate event trace findings event id");
    ExpectTrue(events.find("external combine read without producer") != std::string::npos,
               "run cli validate event trace preserves event detail");
    ExpectTrue(report.find("\"mode\":\"event_trace_validation\"") != std::string::npos,
               "run cli validate event trace report mode");
    ExpectTrue(report.find("\"event_trace_input\":\"" + trace_path + "\"") !=
                   std::string::npos,
               "run cli validate event trace report input path");
}

void TestRunCheckerCliValidateEventTraceChecksInstalledManifestCoverage() {
    const std::string dir = MakeTempDir();
    const std::string repo_root = dir + "/repo";
    const std::string manifest_dir = repo_root + "/tools/checker/installed_traces";
    const std::string mkdir_cmd = "mkdir -p " + manifest_dir;
    ExpectEqInt(std::system(mkdir_cmd.c_str()), 0,
                "validate event trace manifest mkdir command");
    WriteFile(manifest_dir + "/demo_trace_manifest.json",
              "{"
              "\"installed_trace_seed\":{"
              "\"required_event_coverage\":["
              "{\"kind\":\"COPY\","
              "\"source_file\":\"src/collectives/kernels/demo.h\","
              "\"source_line\":7}"
              "]}}");

    const std::string trace_path = dir + "/events.jsonl";
    WriteFile(trace_path,
              "{\"id\":1,\"kind\":\"PIPE_BARRIER\",\"rank\":0,\"peer_rank\":-1,"
              "\"server\":0,\"peer_server\":-1,\"core\":0,\"pipe\":0,"
              "\"event_id\":-1,\"buffer_role\":\"Metadata\","
              "\"slot\":-1,\"magic\":0,\"offset\":0,\"bytes\":0,"
              "\"allow_future_producer\":false,"
              "\"source_file\":\"src/collectives/kernels/demo.h\","
              "\"source_line\":9,\"detail\":\"observed different hook\"}\n");

    tilexr::checker::CliOptions options;
    options.validate_event_trace = true;
    options.event_trace_input = trace_path;
    options.output_dir = dir + "/report";
    options.repo_root = repo_root;
    options.trace_adapter_spec.adapter_name = "demo";
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;

    const int exit_code =
        tilexr::checker::RunCheckerCli(options, &stdout_stream, &stderr_stream);

    ExpectEqInt(exit_code, 1,
                "run cli validate event trace manifest coverage exit code");
    ExpectTrue(stderr_stream.str().empty(),
               "run cli validate event trace manifest coverage stderr");
    ExpectTrue(stdout_stream.str().find("EVENT_COVERAGE_GAP") != std::string::npos,
               "run cli validate event trace manifest coverage stdout finding");
    const std::string summary = ReadFile(options.output_dir + "/summary.txt");
    const std::string findings = ReadFile(options.output_dir + "/findings.json");
    const std::string report = ReadFile(options.output_dir + "/checker_report.json");
    ExpectTrue(summary.find("checker: FAIL") != std::string::npos,
               "run cli validate event trace manifest coverage summary fail");
    ExpectTrue(summary.find("top finding: EVENT_COVERAGE_GAP") != std::string::npos,
               "run cli validate event trace manifest coverage summary kind");
    ExpectTrue(summary.find("source: src/collectives/kernels/demo.h:7") !=
                   std::string::npos,
               "run cli validate event trace manifest coverage summary source");
    ExpectTrue(findings.find("\"kind\":\"EVENT_COVERAGE_GAP\"") != std::string::npos,
               "run cli validate event trace manifest coverage findings kind");
    ExpectTrue(findings.find("event_coverage:COPY:src/collectives/kernels/demo.h:7") !=
                   std::string::npos,
               "run cli validate event trace manifest coverage findings context");
    ExpectTrue(report.find("\"mode\":\"event_trace_validation\"") != std::string::npos,
               "run cli validate event trace manifest coverage report mode");
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
    TestParseAllGatherHierarchyDoubleRingAlgorithm();
    TestParseAllReduceSumWithInjectionFlags();
    TestParseAllReduceBigDataAlgorithm();
    TestParseUnsupportedDatatypeKeepsExecutableCaseForReporting();
    TestParseEpDispatch();
    TestParseEpCombineKeepsExecutableCaseForReporting();
    TestParseInvalidOpReturnsUnsupported();
    TestParseRankSizeOverflowReturnsUnsupported();
    TestParseInvalidServerCountReturnsUnsupported();
    TestParseGenerateTraceAdapterMode();
    TestParseGenerateTraceAdapterStrictSourceObservation();
    TestParseVerificationReportOutput();
    TestParseInstalledTraceInventoryModes();
    TestParseCapabilityInventoryMode();
    TestParseAnalyzeTraceSourceMode();
    TestParseValidateEventTraceMode();
    TestParseValidateEventTraceRequiresInput();
    TestParseGenerateTraceAdapterInfersNameAndTargetHeader();
    TestParseGenerateTraceAdapterOutputDirDefaultsBundle();
    TestParseScaffoldTraceBundleDefaultsAndReport();
    TestParseScaffoldTraceBundleUsesRepoRootForSourceScan();
    TestRunCheckerCliPassExitCode();
    TestRunCheckerCliWritesServerTopologyEvents();
    TestRunCheckerCliHelpShowsRepoRootForTraceBundleVerification();
    TestRunCheckerCliGenerateTraceAdapter();
    TestRunCheckerCliAnalyzeTraceSourceWritesJsonReport();
    TestRunCheckerCliScaffoldTraceBundleWritesVerificationReport();
    TestRunCheckerCliScaffoldTraceBundleScansRelativeSourceFromRepoRoot();
    TestRunCheckerCliGenerateTraceAdapterRejectsProductionOutput();
    TestRunCheckerCliStrictGenerateTraceAdapterRejectsRawDataCopy();
    TestRunCheckerCliVerifyTraceBundleReportsUncheckedItems();
    TestRunCheckerCliVerifyTraceBundleWritesFailureJsonReport();
    TestRunCheckerCliVerifyTraceBundleCompilesGeneratedProbe();
    TestRunCheckerCliVerifyInstalledTraceAlgorithm();
    TestRunCheckerCliVerifyInstalledTraceWritesJsonReport();
    TestRunCheckerCliVerifyInstalledHdbTraceAlgorithm();
    TestRunCheckerCliListInstalledTraces();
    TestRunCheckerCliVerifyAllInstalledTracesWritesJsonReport();
    TestRunCheckerCliListCapabilitiesWritesJsonReport();
    TestNormalizeExecutorFailStatusPreservedForReporting();
    TestRunCheckerCliInjectedFindingExitCode();
    TestRunCheckerCliInjectedPipeFindingExitCode();
    TestRunCheckerCliInjectedEpWindowReadWritesSourceLocatedReport();
    TestRunCheckerCliResolvesSourceExcerptFromRepoRoot();
    TestRunCheckerCliUnsupportedExecutorStillWritesReports();
    TestRunCheckerCliEpCombineWritesSourceLocatedReports();
    TestRunCheckerCliValidateEventTraceWritesSourceLocatedReport();
    TestRunCheckerCliValidateEventTraceChecksInstalledManifestCoverage();
    TestRunCheckerCliInvalidOutputDirExitCode();
    TestRunCheckerCliReportWriteFailureExitCode();
    return g_failures == 0 ? 0 : 1;
}
