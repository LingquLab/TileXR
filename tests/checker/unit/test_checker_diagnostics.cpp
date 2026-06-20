#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tilexr/checker/case.h"
#include "tilexr/checker/diagnostics.h"
#include "tilexr/checker/event.h"
#include "tilexr/checker/oracle.h"
#include "tilexr/checker/report.h"
#include "tilexr/checker/types.h"

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

void ExpectEqSize(size_t actual, size_t expected, const char *message) {
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

void ExpectContains(const std::string &text, const std::string &needle,
                    const char *message) {
    if (text.find(needle) == std::string::npos) {
        std::cerr << message << ": missing " << needle << "\n";
        ++g_failures;
    }
}

std::string SourcePath(const std::string &relative_path) {
    return std::string(TILEXR_SOURCE_ROOT) + "/" + relative_path;
}

std::string ReadFile(const std::string &path) {
    std::ifstream input(path.c_str());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

tilexr::checker::CheckerCase MakeCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllGather;
    test_case.rank_size = 2;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    test_case.magic = 0x22;
    return test_case;
}

tilexr::checker::Event MakeEvent(tilexr::checker::EventKind kind, int rank,
                                 int peer_rank, tilexr::checker::BufferRole role,
                                 int slot, uint64_t magic, size_t offset,
                                 size_t bytes, const std::string &detail) {
    tilexr::checker::Event event;
    event.kind = kind;
    event.rank = rank;
    event.peer_rank = peer_rank;
    event.buffer_role = role;
    event.slot = slot;
    event.magic = magic;
    event.offset = offset;
    event.bytes = bytes;
    event.detail = detail;
    event.source_file = "synthetic.cpp";
    event.source_line = 42;
    return event;
}

void TestOrderingFindingsAndPriority() {
    tilexr::checker::EventLog events;
    events.Add(MakeEvent(tilexr::checker::EventKind::kRead, 1, 0,
                         tilexr::checker::BufferRole::kCommData, 0, 0, 0, 16,
                         "consumer read before producer copy"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagWait, 1, 0,
                         tilexr::checker::BufferRole::kCommFlag, 3, 0x20, 0,
                         sizeof(uint64_t), "wait without store"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kRead, 1, 0,
                         tilexr::checker::BufferRole::kUserInput, 0, 0, 4, 8,
                         "direct peer user read"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 3, "ordering finding count");
    ExpectTrue(findings.HasErrors(), "ordering findings should include errors");

    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "top finding should exist");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kFlagNoProducer),
                    "top finding priority should favor missing producer flag");
    }
}

void TestReadBeforeCopySummaryMatchesGolden() {
    tilexr::checker::EventLog events;
    events.Add(MakeEvent(tilexr::checker::EventKind::kRead, 1, 0,
                         tilexr::checker::BufferRole::kCommData, 0, 0, 64, 32,
                         "comm-data read"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "read-before-copy top finding exists");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kReadBeforeCopy),
                    "read before copy finding kind");
    }

    const std::string summary =
        tilexr::checker::RenderSummary(MakeCase(), findings, events.events().size());
    const std::string expected = ReadFile(
        SourcePath("tests/checker/golden/read_before_copy_summary.txt"));
    ExpectEqString(summary, expected, "read before copy summary matches golden");
}

void TestDirectPeerUserBufferFinding() {
    tilexr::checker::EventLog events;
    events.Add(MakeEvent(tilexr::checker::EventKind::kRead, 0, 1,
                         tilexr::checker::BufferRole::kUserOutput, 0, 0, 12, 4,
                         "peer output read"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 1, "direct peer user finding count");
    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "direct peer user top finding exists");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kDirectPeerUserBuffer),
                    "direct peer user finding kind");
    }
}

void TestOutputMismatchKindsAndJson() {
    std::vector<tilexr::checker::OutputMismatch> mismatches;

    tilexr::checker::OutputMismatch mismatch;
    mismatch.rank = 0;
    mismatch.element_index = 7;
    mismatch.expected = 4;
    mismatch.actual = 9;
    mismatch.context = "validation output mismatch";
    mismatches.push_back(mismatch);

    tilexr::checker::OutputMismatch unsupported;
    unsupported.rank = -1;
    unsupported.element_index = -1;
    unsupported.expected = 0;
    unsupported.actual = 0;
    unsupported.context = "validation failed: unsupported rank size";
    mismatches.push_back(unsupported);

    tilexr::checker::FindingSet findings =
        tilexr::checker::CheckOutputMismatches(mismatches);
    ExpectEqSize(findings.findings().size(), 2, "output mismatch finding count");

    const std::string json = tilexr::checker::RenderFindingsJson(findings);
    ExpectContains(json, "\"kind\":\"OUTPUT_MISMATCH\"",
                   "json includes output mismatch kind");
    ExpectContains(json, "\"kind\":\"UNSUPPORTED_API\"",
                   "json includes unsupported api kind");
}

void TestEventsJsonlIncludesStableKinds() {
    tilexr::checker::EventLog events;
    events.Add(MakeEvent(tilexr::checker::EventKind::kCopy, 0, 1,
                         tilexr::checker::BufferRole::kCommData, 1, 0, 0, 16,
                         "copy event"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagStore, 0, 1,
                         tilexr::checker::BufferRole::kCommFlag, 1, 0x33, 0,
                         sizeof(uint64_t), "store event"));

    const std::string jsonl = tilexr::checker::RenderEventsJsonl(events);
    ExpectContains(jsonl, "\"kind\":\"COPY\"", "events jsonl includes copy kind");
    ExpectContains(jsonl, "\"kind\":\"FLAG_STORE\"",
                   "events jsonl includes flag store kind");
}

}  // namespace

int main() {
    TestOrderingFindingsAndPriority();
    TestReadBeforeCopySummaryMatchesGolden();
    TestDirectPeerUserBufferFinding();
    TestOutputMismatchKindsAndJson();
    TestEventsJsonlIncludesStableKinds();
    return g_failures == 0 ? 0 : 1;
}
