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
    test_case.algorithm = tilexr::checker::AlgorithmId::kDefault;
    test_case.magic = 0x44;
    return test_case;
}

tilexr::checker::CheckerCase MakeAllGatherHdbCase() {
    tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    test_case.rank_size = 10;
    test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    test_case.algorithm = tilexr::checker::AlgorithmId::kAllGatherHierarchyDoubleRing;
    return test_case;
}

tilexr::checker::CheckerCase MakeAllReduceBigDataCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 4;
    test_case.count = 524288;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    test_case.algorithm = tilexr::checker::AlgorithmId::kAllReduceBigData;
    test_case.magic = 0x55;
    return test_case;
}

tilexr::checker::CheckerCase MakeEpDispatchCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kEpDispatch;
    test_case.rank_size = 2;
    test_case.server_count = 2;
    test_case.bs = 3;
    test_case.h = 4;
    test_case.top_k = 2;
    test_case.moe_expert_num = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    test_case.magic = 0x66;
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
    event.server = 1;
    event.peer_server = 0;
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

void AddFlagEvent(tilexr::checker::EventLog *events, tilexr::checker::EventKind kind,
                  int rank, int peer_rank, int slot, uint64_t magic,
                  const std::string &detail, int line) {
    tilexr::checker::Event event;
    event.kind = kind;
    event.rank = rank;
    event.peer_rank = peer_rank;
    event.server = rank;
    event.peer_server = peer_rank;
    event.buffer_role = tilexr::checker::BufferRole::kCommFlag;
    event.slot = slot;
    event.magic = magic;
    event.source_file = "src/collectives/kernels/test_flag_timeline.h";
    event.source_line = line;
    event.detail = detail;
    events->Add(event);
}

void AddPipeEvent(tilexr::checker::EventLog *events, tilexr::checker::EventKind kind,
                  int pipe, int event_id, const std::string &detail, int line) {
    tilexr::checker::Event event;
    event.kind = kind;
    event.rank = 0;
    event.server = 0;
    event.core = 2;
    event.pipe = pipe;
    event.event_id = event_id;
    event.buffer_role = tilexr::checker::BufferRole::kMetadata;
    event.source_file = "src/collectives/kernels/test_pipe_timeline.h";
    event.source_line = line;
    event.detail = detail;
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
    ExpectContains(ReadFile(paths.checker_report_json),
                   "\"topology\":{\"rank_size\":2,\"server_count\":1,\"local_rank_size\":2}",
                   "checker report topology");
    ExpectContains(ReadFile(paths.checker_report_json), "\"summary_txt\":\"summary.txt\"",
                   "checker report summary artifact name");
    ExpectContains(ReadFile(paths.checker_report_json),
                   "\"artifact_paths\":{",
                   "checker report artifact paths");
    ExpectContains(ReadFile(paths.checker_report_json),
                   "\"summary_txt\":\"" + paths.summary_txt + "\"",
                   "checker report summary artifact path");
    ExpectContains(ReadFile(paths.checker_report_json),
                   "\"events_jsonl\":\"" + paths.events_jsonl + "\"",
                   "checker report events artifact path");
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
    ExpectContains(summary, "server: 1", "summary top finding server");
    ExpectContains(summary, "peer server: 0", "summary top finding peer server");
    ExpectContains(summary, "event id:", "summary top finding event id");
    ExpectContains(summary, "source:", "summary top finding source");
    ExpectContains(summary, "next action:", "summary next action");
    ExpectContains(report_json, "\"server\":1", "checker report top finding server");
    ExpectContains(report_json, "\"peer_server\":0", "checker report top finding peer server");
    ExpectContains(report_json, "\"event_log_id\":", "checker report top finding event log id");
    ExpectContains(report_json, "\"event_context\":[", "checker report event context");
    ExpectContains(report_json, "\"relative\":0", "checker report triggering event context");
    ExpectContains(report_json, "\"core\":3", "checker report top finding core");
    ExpectContains(report_json, "\"source_file\":\"", "checker report top finding source file");
    ExpectContains(report_json, "\"source_line\":", "checker report top finding source line");
}

void TestFlagFindingReportIncludesRelevantTimeline() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::EventLog events;
    AddFlagEvent(&events, tilexr::checker::EventKind::kFlagStore, 0, 1, 7,
                 (1ULL << 32) | 1ULL, "old store for peer", 40);
    AddFlagEvent(&events, tilexr::checker::EventKind::kFlagWait, 1, 0, 7,
                 (2ULL << 32) | 1ULL, "wait missing producer", 50);
    AddFlagEvent(&events, tilexr::checker::EventKind::kFlagStore, 0, 1, 7,
                 (3ULL << 32) | 1ULL, "newer store same slot", 60);

    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Fail("synthetic stale flag");
    result.findings = tilexr::checker::CheckOrdering(events);
    result.event_count = events.events().size();

    const std::string output_dir = MakeTempDir();
    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, events, &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "flag timeline report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    const std::string report_json = ReadFile(paths.checker_report_json);
    ExpectContains(summary, "timeline events: 3",
                   "flag summary timeline event count");
    ExpectContains(summary, "timeline first relation: same_flag_slot",
                   "flag summary timeline first relation");
    ExpectContains(summary, "timeline: checker_report.json top_finding.timeline",
                   "flag summary timeline report pointer");
    ExpectContains(report_json, "\"timeline\":[", "flag report timeline field");
    ExpectContains(report_json, "\"relation\":\"same_flag_slot\"",
                   "flag report same slot relation");
    ExpectContains(report_json, "\"kind\":\"FLAG_STORE\"",
                   "flag report store timeline entry");
    ExpectContains(report_json, "\"kind\":\"FLAG_WAIT\"",
                   "flag report wait timeline entry");
    ExpectContains(report_json, "\"magic_epoch\":1",
                   "flag report old magic epoch");
    ExpectContains(report_json, "\"magic_epoch\":2",
                   "flag report wait magic epoch");
    ExpectContains(report_json, "\"magic_epoch\":3",
                   "flag report newer magic epoch");
    ExpectContains(report_json, "\"source_line\":50",
                   "flag report triggering wait source");
}

void TestPipeFindingReportIncludesRelevantTimeline() {
    const tilexr::checker::CheckerCase test_case = MakeAllReduceBigDataCase();
    tilexr::checker::EventLog events;
    AddPipeEvent(&events, tilexr::checker::EventKind::kPipeSet, 1, 2,
                 "different event id", 70);
    AddPipeEvent(&events, tilexr::checker::EventKind::kPipeBarrier, 1, -1,
                 "barrier near wait", 71);
    AddPipeEvent(&events, tilexr::checker::EventKind::kPipeWait, 1, 3,
                 "wait missing producer", 72);

    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Fail("synthetic pipe wait");
    result.findings = tilexr::checker::CheckOrdering(events);
    result.event_count = events.events().size();

    const std::string output_dir = MakeTempDir();
    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, events, &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "pipe timeline report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    const std::string report_json = ReadFile(paths.checker_report_json);
    ExpectContains(summary, "timeline events: 3",
                   "pipe summary timeline event count");
    ExpectContains(summary, "timeline first relation: same_pipe_event",
                   "pipe summary timeline first relation");
    ExpectContains(report_json, "\"timeline\":[", "pipe report timeline field");
    ExpectContains(report_json, "\"relation\":\"same_pipe_event\"",
                   "pipe report same pipe relation");
    ExpectContains(report_json, "\"kind\":\"PIPE_BARRIER\"",
                   "pipe report barrier timeline entry");
    ExpectContains(report_json, "\"kind\":\"PIPE_WAIT\"",
                   "pipe report wait timeline entry");
    ExpectContains(report_json, "\"pipe\":1",
                   "pipe report pipe id");
    ExpectContains(report_json, "\"event_id\":3",
                   "pipe report triggering event id");
    ExpectContains(report_json, "\"source_line\":72",
                   "pipe report triggering wait source");
}

void TestUnsupportedSummaryKeepsUnsupportedStatus() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::EventLog events;
    const std::string output_dir = MakeTempDir();

    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Unsupported(
        "only int32 checker cases are supported");

    tilexr::checker::Finding finding;
    finding.id = "UNSUPPORTED_API:test";
    finding.kind = tilexr::checker::FindingKind::kUnsupportedApi;
    finding.severity = tilexr::checker::Severity::kWarning;
    finding.message = result.status.message;
    finding.next_action = "Use a supported checker datatype.";
    finding.confidence = 0.9;
    result.findings.Add(finding);

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, events, &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "unsupported report writer status");
    ExpectContains(ReadFile(paths.summary_txt), "checker: UNSUPPORTED",
                   "unsupported summary status");
    ExpectContains(ReadFile(paths.checker_report_json), "\"status\":\"UNSUPPORTED\"",
                   "unsupported checker report status");
}

void TestCheckerReportIncludesSourceExcerptForTopFinding() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::EventLog events;
    const std::string output_dir = MakeTempDir();
    const std::string source_path = output_dir + "/synthetic_source.h";
    {
        std::ofstream source(source_path.c_str(), std::ios::out | std::ios::trunc);
        source << "line one setup\n"
               << "line two producer\n"
               << "line three failing read\n"
               << "line four cleanup\n";
    }

    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Fail("synthetic finding");
    tilexr::checker::Finding finding;
    finding.id = "READ_BEFORE_WRITE:test";
    finding.kind = tilexr::checker::FindingKind::kReadBeforeWrite;
    finding.severity = tilexr::checker::Severity::kError;
    finding.message = "synthetic source excerpt finding";
    finding.next_action = "inspect source excerpt";
    finding.source_file = source_path;
    finding.source_line = 3;
    finding.confidence = 0.9;
    result.findings.Add(finding);

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, events, &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "source excerpt report writer status");
    const std::string report_json = ReadFile(paths.checker_report_json);
    ExpectContains(report_json, "\"source_excerpt\":[",
                   "checker report source excerpt field");
    ExpectContains(report_json, "\"line\":2",
                   "checker report source excerpt previous line");
    ExpectContains(report_json, "\"text\":\"line two producer\"",
                   "checker report source excerpt previous text");
    ExpectContains(report_json, "\"line\":3",
                   "checker report source excerpt target line");
    ExpectContains(report_json, "\"text\":\"line three failing read\"",
                   "checker report source excerpt target text");
    ExpectContains(report_json, "\"is_target\":true",
                   "checker report source excerpt target marker");
}

void TestCheckerReportResolvesRelativeSourceExcerptFromSourceRoot() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::EventLog events;
    const std::string output_dir = MakeTempDir();
    const std::string source_root = MakeTempDir();
    const std::string relative_source = "relative_source_for_excerpt.h";
    {
        std::ofstream source((source_root + "/" + relative_source).c_str(),
                             std::ios::out | std::ios::trunc);
        source << "root line one\n"
               << "root line two target\n"
               << "root line three\n";
    }

    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Fail("relative source finding");
    tilexr::checker::Finding finding;
    finding.id = "READ_BEFORE_WRITE:relative";
    finding.kind = tilexr::checker::FindingKind::kReadBeforeWrite;
    finding.severity = tilexr::checker::Severity::kError;
    finding.message = "relative source excerpt finding";
    finding.next_action = "inspect source root excerpt";
    finding.source_file = relative_source;
    finding.source_line = 2;
    finding.confidence = 0.9;
    result.findings.Add(finding);

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, events,
                                          source_root, &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "relative source excerpt report writer status");
    const std::string report_json = ReadFile(paths.checker_report_json);
    ExpectContains(report_json, "\"source_excerpt\":[",
                   "checker report relative source excerpt field");
    ExpectContains(report_json, "\"line\":2",
                   "checker report relative source excerpt target line");
    ExpectContains(report_json, "\"text\":\"root line two target\"",
                   "checker report relative source excerpt target text");
}

void TestSummaryAndReportClassifyTopFindingServerScope() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::EventLog events;
    const std::string output_dir = MakeTempDir();

    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Fail("cross-server finding");
    tilexr::checker::Finding finding;
    finding.id = "READ_BEFORE_WRITE:cross-server";
    finding.kind = tilexr::checker::FindingKind::kReadBeforeWrite;
    finding.severity = tilexr::checker::Severity::kError;
    finding.message = "cross-server source not ready";
    finding.next_action = "inspect cross-server ordering";
    finding.rank = 0;
    finding.peer_rank = 3;
    finding.server = 0;
    finding.peer_server = 1;
    finding.buffer_role = tilexr::checker::BufferRole::kCommData;
    finding.source_file = "src/collectives/kernels/cross_server.h";
    finding.source_line = 42;
    finding.confidence = 0.9;
    result.findings.Add(finding);

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, events, &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "server scope report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    const std::string report_json = ReadFile(paths.checker_report_json);
    ExpectContains(summary, "server scope: cross_server",
                   "summary cross-server scope");
    ExpectContains(report_json, "\"server_scope\":\"cross_server\"",
                   "checker report cross-server scope");
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
    const std::string report_json = ReadFile(paths.checker_report_json);
    ExpectContains(summary, "checker: FAIL", "pure mismatch summary fail");
    ExpectContains(summary, "top finding: OUTPUT_MISMATCH",
                   "pure mismatch summary top finding");
    ExpectContains(summary, "expected: 100003",
                   "pure mismatch summary expected");
    ExpectContains(summary, "actual: 7",
                   "pure mismatch summary actual");
    ExpectContains(summary, "context: synthetic mismatch",
                   "pure mismatch summary context");
    ExpectContains(summary, "mismatches: 1", "pure mismatch summary mismatch count");
    ExpectContains(report_json, "\"status\":\"FAIL\"",
                   "pure mismatch checker report fail");
    ExpectContains(report_json, "\"expected\":100003",
                   "pure mismatch checker report expected");
    ExpectContains(report_json, "\"actual\":7",
                   "pure mismatch checker report actual");
    ExpectContains(report_json, "\"context\":\"synthetic mismatch\"",
                   "pure mismatch checker report context");
}

void TestMismatchReportLinksProducingOutputEvent() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherCase();
    tilexr::checker::EventLog events;
    tilexr::checker::Event write;
    write.kind = tilexr::checker::EventKind::kWrite;
    write.rank = 0;
    write.peer_rank = 0;
    write.server = 0;
    write.peer_server = 0;
    write.buffer_role = tilexr::checker::BufferRole::kUserOutput;
    write.slot = 0;
    write.offset = 3 * sizeof(int32_t);
    write.bytes = sizeof(int32_t);
    write.source_file = "src/collectives/kernels/debug_writer.h";
    write.source_line = 88;
    write.detail = "write corrupted output";
    events.Add(write);
    const std::string output_dir = MakeTempDir();

    tilexr::checker::RunResult result;
    result.status = tilexr::checker::CheckerStatus::Fail("synthetic output mismatch");
    result.event_count = events.events().size();

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
                "producer-linked mismatch report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    const std::string report_json = ReadFile(paths.checker_report_json);
    ExpectContains(summary, "source: src/collectives/kernels/debug_writer.h:88",
                   "producer-linked mismatch summary source");
    ExpectContains(summary, "event id: 1",
                   "producer-linked mismatch summary event id");
    ExpectContains(report_json, "\"event_log_id\":1",
                   "producer-linked mismatch report event id");
    ExpectContains(report_json,
                   "\"source_file\":\"src/collectives/kernels/debug_writer.h\"",
                   "producer-linked mismatch report source");
    ExpectContains(report_json, "\"relative\":0",
                   "producer-linked mismatch report event context");
}

void TestHierarchyDoubleRingReportIdentifiesProductionSource() {
    const tilexr::checker::CheckerCase test_case = MakeAllGatherHdbCase();
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;
    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    const std::string output_dir = MakeTempDir();

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, world.events(), &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "hdb report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    const std::string report_json = ReadFile(paths.checker_report_json);
    const std::string events_jsonl = ReadFile(paths.events_jsonl);
    ExpectContains(summary, "checker: PASS", "hdb summary pass");
    ExpectContains(summary, "algorithm: allgather_hierarchy_double_ring",
                   "hdb summary algorithm");
    ExpectContains(summary,
                   "source: src/collectives/kernels/91093/allgather_hierarchy_double_ring.h",
                   "hdb summary source");
    ExpectContains(report_json, "\"algorithm\":\"allgather_hierarchy_double_ring\"",
                   "hdb checker report algorithm");
    ExpectContains(report_json, "\"algorithm_selection\":",
                   "hdb checker report selection explanation");
    ExpectContains(report_json, "\"eligible\":true",
                   "hdb checker report selection eligible");
    ExpectContains(report_json, "rank_size > 8",
                   "hdb checker report selection reason");
    ExpectContains(report_json,
                   "\"source_file\":\"src/collectives/kernels/91093/allgather_hierarchy_double_ring.h\"",
                   "hdb checker report source");
    ExpectContains(events_jsonl,
                   "\"source_file\":\"src/collectives/kernels/91093/allgather_hierarchy_double_ring.h\"",
                   "hdb event source");
    ExpectContains(events_jsonl, "\"kind\":\"COPY\"",
                   "hdb trace records comm copy events");
    ExpectContains(events_jsonl, "\"source_line\":139",
                   "hdb trace records production hccs ring call-site line");
    ExpectContains(events_jsonl, "\"source_line\":234",
                   "hdb trace records production sio output call-site line");
}

void TestAllReduceBigDataReportIdentifiesProductionTraceSource() {
    const tilexr::checker::CheckerCase test_case = MakeAllReduceBigDataCase();
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;
    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    const std::string output_dir = MakeTempDir();

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, world.events(), &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allreduce big data report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    const std::string report_json = ReadFile(paths.checker_report_json);
    const std::string events_jsonl = ReadFile(paths.events_jsonl);
    ExpectContains(summary, "checker: PASS", "allreduce big data summary pass");
    ExpectContains(summary, "algorithm: allreduce_big_data",
                   "allreduce big data summary algorithm");
    ExpectContains(summary, "source: src/collectives/kernels/allreduce_big_data.h",
                   "allreduce big data summary source");
    ExpectContains(report_json, "\"algorithm\":\"allreduce_big_data\"",
                   "allreduce big data checker report algorithm");
    ExpectContains(report_json, "\"algorithm_selection\":",
                   "allreduce big data checker report selection explanation");
    ExpectContains(report_json, "\"eligible\":true",
                   "allreduce big data checker report selection eligible");
    ExpectContains(report_json, "data size >= 2MiB",
                   "allreduce big data checker report selection reason");
    ExpectContains(report_json, "\"source_file\":\"src/collectives/kernels/allreduce_big_data.h\"",
                   "allreduce big data checker report source");
    ExpectContains(events_jsonl, "\"source_file\":\"src/collectives/kernels/allreduce_big_data.h\"",
                   "allreduce big data event source");
    ExpectContains(events_jsonl, "\"kind\":\"COPY\"",
                   "allreduce big data trace records comm copy events");
    ExpectContains(events_jsonl, "\"buffer_role\":\"CommData\"",
                   "allreduce big data trace records comm buffer events");
    ExpectContains(events_jsonl, "\"source_line\":156",
                   "allreduce big data trace records production call-site line");
    ExpectContains(events_jsonl, "\"source_line\":191",
                   "allreduce big data trace records consumer reduction call-site line");
    ExpectContains(events_jsonl, "\"source_line\":207",
                   "allreduce big data trace records consumer output call-site line");
    ExpectContains(events_jsonl, "\"kind\":\"PIPE_SET\"",
                   "allreduce big data trace records aicore pipe set events");
    ExpectContains(events_jsonl, "\"kind\":\"PIPE_WAIT\"",
                   "allreduce big data trace records aicore pipe wait events");
    ExpectContains(events_jsonl, "\"kind\":\"PIPE_BARRIER\"",
                   "allreduce big data trace records aicore pipe barrier events");
    ExpectContains(events_jsonl, "\"source_file\":\"src/collectives/kernels/collectives.h\"",
                   "allreduce big data pipe events point to included production helper");
    ExpectContains(events_jsonl, "\"source_line\":442",
                   "allreduce big data pipe barrier records SetAtomic call-site");
    ExpectContains(events_jsonl, "\"source_line\":273",
                   "allreduce big data pipe wait records ping-pong call-site");
}

void TestEpDispatchReportIdentifiesEpSourceAndTopology() {
    const tilexr::checker::CheckerCase test_case = MakeEpDispatchCase();
    const size_t input_bytes = static_cast<size_t>(test_case.bs * test_case.h) * sizeof(uint16_t);
    const size_t output_bytes = tilexr::checker::EpDispatchOutputBytes(test_case);
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(test_case.rank_size, input_bytes, output_bytes,
                                           output_bytes);
    tilexr::checker::CollectiveExecutor executor;
    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    const std::string output_dir = MakeTempDir();

    tilexr::checker::ReportPaths paths;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteReportFiles(output_dir, test_case, result, world.events(), &paths);

    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "ep dispatch report writer status");
    const std::string summary = ReadFile(paths.summary_txt);
    const std::string report_json = ReadFile(paths.checker_report_json);
    const std::string events_jsonl = ReadFile(paths.events_jsonl);
    ExpectContains(summary, "checker: PASS", "ep dispatch summary pass");
    ExpectContains(summary, "source: src/ep/kernels/tilexr_ep_dispatch_kernel.cpp",
                   "ep dispatch summary source");
    ExpectContains(report_json, "\"topology\":{\"rank_size\":2,\"server_count\":2",
                   "ep dispatch topology");
    ExpectContains(report_json,
                   "\"source_file\":\"src/ep/kernels/tilexr_ep_dispatch_kernel.cpp\"",
                   "ep dispatch checker report source");
    ExpectContains(report_json, "\"ep_dispatch_layout\":",
                   "ep dispatch checker report layout");
    ExpectContains(report_json, "\"payload_bytes\":96",
                   "ep dispatch checker report payload bytes");
    ExpectContains(report_json, "\"assist\":{\"offset\":96,\"bytes\":96}",
                   "ep dispatch checker report assist layout");
    ExpectContains(report_json, "\"recv_counts\":{\"offset\":192,\"bytes\":8}",
                   "ep dispatch checker report recv count layout");
    ExpectContains(report_json, "\"expert_token_nums\":{\"offset\":200,\"bytes\":16}",
                   "ep dispatch checker report expert count layout");
    ExpectContains(report_json,
                   "\"window\":{\"rank_size\":2,\"local_expert_num\":2,\"dtype_bytes\":2",
                   "ep dispatch checker report production window layout");
    ExpectContains(report_json, "\"max_routes_per_src\":6",
                   "ep dispatch checker report max routes per src");
    ExpectContains(report_json, "\"row_bytes\":8",
                   "ep dispatch checker report row bytes");
    ExpectContains(report_json, "\"payload_bytes_per_slot\":64",
                   "ep dispatch checker report window payload bytes");
    ExpectContains(report_json, "\"assist_bytes_per_slot\":96",
                   "ep dispatch checker report window assist bytes");
    ExpectContains(report_json, "\"slot_bytes\":224",
                   "ep dispatch checker report window slot bytes");
    ExpectContains(report_json, "\"total_bytes\":512",
                   "ep dispatch checker report window total bytes");
    ExpectContains(events_jsonl, "ep dispatch route payload",
                   "ep dispatch event route payload");
    ExpectContains(events_jsonl, "ep dispatch assist",
                   "ep dispatch event assist");
    ExpectContains(events_jsonl, "ep dispatch window header",
                   "ep dispatch event window header");
    ExpectContains(events_jsonl, "ep dispatch src slot header",
                   "ep dispatch event slot header");
    ExpectContains(events_jsonl, "\"buffer_role\":\"RegisteredCommBuffer\"",
                   "ep dispatch event registered comm buffer");
    ExpectContains(events_jsonl, "\"source_line\":143",
                   "ep dispatch window header source line");
    ExpectContains(events_jsonl, "\"source_line\":159",
                   "ep dispatch slot header source line");
}

}  // namespace

int main() {
    TestWriteReportFilesCreatesArtifacts();
    TestInjectedBadTraceWritesTopFindingAndNextAction();
    TestFlagFindingReportIncludesRelevantTimeline();
    TestPipeFindingReportIncludesRelevantTimeline();
    TestUnsupportedSummaryKeepsUnsupportedStatus();
    TestCheckerReportIncludesSourceExcerptForTopFinding();
    TestCheckerReportResolvesRelativeSourceExcerptFromSourceRoot();
    TestSummaryAndReportClassifyTopFindingServerScope();
    TestPureMismatchSummaryUsesFailStatus();
    TestMismatchReportLinksProducingOutputEvent();
    TestHierarchyDoubleRingReportIdentifiesProductionSource();
    TestAllReduceBigDataReportIdentifiesProductionTraceSource();
    TestEpDispatchReportIdentifiesEpSourceAndTopology();
    return g_failures == 0 ? 0 : 1;
}
