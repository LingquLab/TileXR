#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "tilexr/checker/cli.h"
#include "tilexr/checker/diagnostics.h"
#include "tilexr/checker/event.h"
#include "tilexr/checker/executor.h"
#include "tilexr/checker/oracle.h"
#include "tilexr/checker/world.h"

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

void ExpectContains(const std::string &text, const std::string &needle,
                    const char *message) {
    if (text.find(needle) == std::string::npos) {
        std::cerr << message << ": missing " << needle << "\n";
        ++g_failures;
    }
}

std::string ReadFile(const std::string &path) {
    std::ifstream input(path.c_str());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

std::string MakeTempDir() {
    char dir_template[] = "/tmp/tilexr-checker-test-XXXXXX";
    char *created = mkdtemp(dir_template);
    if (created == nullptr) {
        std::cerr << "mkdtemp failed: " << std::strerror(errno) << "\n";
        ++g_failures;
        return std::string();
    }
    return std::string(created);
}

tilexr::checker::CheckerCase MakeAllGatherCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllGather;
    test_case.rank_size = 2;
    test_case.count = 16;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    test_case.magic = 0x44;
    return test_case;
}

tilexr::checker::RankWorld MakeWorld(const tilexr::checker::CheckerCase &test_case) {
    const size_t input_bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    const size_t output_bytes =
        static_cast<size_t>(test_case.rank_size * test_case.count) * sizeof(int32_t);
    return tilexr::checker::RankWorld::Create(test_case.rank_size, input_bytes, output_bytes,
                                              input_bytes);
}

void InjectReadBeforeCopy(tilexr::checker::EventLog *events) {
    tilexr::checker::Event event;
    event.kind = tilexr::checker::EventKind::kRead;
    event.rank = 1;
    event.peer_rank = 0;
    event.core = 3;
    event.buffer_role = tilexr::checker::BufferRole::kCommData;
    event.slot = 0;
    event.offset = 64;
    event.bytes = 16;
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = "injected read-before-copy";
    events->Add(event);
}

void TestWriteReportFilesCreatesArtifacts() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;
    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    const std::string output_dir = MakeTempDir();

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, world.events(), &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "report writer status");
    ExpectTrue(!paths.summary_txt.empty(), "summary path set");
    ExpectTrue(!paths.findings_json.empty(), "findings path set");
    ExpectTrue(!paths.events_jsonl.empty(), "events path set");
    ExpectTrue(!paths.checker_report_json.empty(), "checker report path set");
    ExpectContains(ReadFile(paths.summary_txt), "checker: PASS", "summary pass");
    ExpectContains(ReadFile(paths.summary_txt), "case: allgather", "summary case");
    ExpectContains(ReadFile(paths.summary_txt), "mismatches: 0", "summary mismatch count");
    ExpectContains(ReadFile(paths.findings_json), "[]", "findings empty json");
    ExpectContains(ReadFile(paths.events_jsonl), "\"kind\":\"COPY\"", "events include copy");
    ExpectContains(ReadFile(paths.checker_report_json), "\"status\":\"PASS\"",
                   "checker report pass");
    ExpectContains(ReadFile(paths.checker_report_json), "\"summary_txt\":\"summary.txt\"",
                   "checker report summary artifact name");
}

void TestInjectedBadTraceWritesTopFindingAndNextAction() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;
    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    InjectReadBeforeCopy(&world.events());
    result.findings =
        tilexr::checker::MergeFindings(result.findings,
                                       tilexr::checker::CheckOrdering(world.events()));
    result.status = tilexr::checker::CheckerStatus::Fail("injected read-before-copy");
    result.event_count = world.events().events().size();

    const std::string output_dir = MakeTempDir();
    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, world.events(), &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "injected report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    const std::string report_json = ReadFile(paths.checker_report_json);
    ExpectContains(summary, "checker: FAIL", "summary fail");
    ExpectContains(summary, "top finding: READ_BEFORE_COPY", "summary top finding");
    ExpectContains(summary, "next action:", "summary next action");
    ExpectContains(report_json, "\"core\":3", "checker report top finding core");
    ExpectContains(report_json, "\"source_file\":\"", "checker report top finding source file");
    ExpectContains(report_json, "\"source_line\":", "checker report top finding source line");
}

void TestPureMismatchSummaryUsesFailStatus() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::EventLog events;
    const std::string output_dir = MakeTempDir();

    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Fail("synthetic output mismatch");
    result.event_count = 0;

    tilexr::checker::OutputMismatch mismatch;
    mismatch.rank = 0;
    mismatch.element_index = 3;
    mismatch.expected = 100003;
    mismatch.actual = 7;
    mismatch.context = "synthetic mismatch";
    result.mismatches.push_back(mismatch);

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, events, &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "pure mismatch report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    ExpectContains(summary, "checker: FAIL", "pure mismatch summary fail");
    ExpectContains(summary, "mismatches: 1", "pure mismatch summary mismatch count");
    ExpectContains(ReadFile(paths.checker_report_json), "\"status\":\"FAIL\"",
                   "pure mismatch checker report fail");
}

}  // namespace

int main() {
    TestWriteReportFilesCreatesArtifacts();
    TestInjectedBadTraceWritesTopFindingAndNextAction();
    TestPureMismatchSummaryUsesFailStatus();
    return g_failures == 0 ? 0 : 1;
}
