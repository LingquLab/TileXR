#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kernel_operator.h"
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

size_t FindingCount(const tilexr::checker::FindingSet &findings,
                    tilexr::checker::FindingKind kind) {
    size_t count = 0;
    const std::vector<tilexr::checker::Finding> &items = findings.findings();
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].kind == kind) {
            ++count;
        }
    }
    return count;
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
    tilexr::checker::Event &read_event = events.Add(
        MakeEvent(tilexr::checker::EventKind::kRead, 1, 0,
                  tilexr::checker::BufferRole::kCommData, 0, 0, 64, 32,
                  "comm-data read"));
    read_event.core = 7;
    read_event.server = 1;
    read_event.peer_server = 0;

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "read-before-copy top finding exists");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kReadBeforeCopy),
                    "read before copy finding kind");
        ExpectEqInt(top->server, 1, "read before copy finding server");
        ExpectEqInt(top->peer_server, 0, "read before copy finding peer server");
        ExpectEqInt(static_cast<int>(top->event_log_id), 1,
                    "read before copy finding event log id");
    }

    const std::string summary =
        tilexr::checker::RenderSummary(MakeCase(), findings, events.events().size());
    const std::string expected = ReadFile(
        SourcePath("tests/checker/golden/read_before_copy_summary.txt"));
    ExpectEqString(summary, expected, "read before copy summary matches golden");
    ExpectContains(summary, "core: 7", "summary includes core");
    ExpectContains(summary, "event id: 1", "summary includes event id");
    ExpectContains(summary, "source: synthetic.cpp:42", "summary includes source location");
    ExpectContains(summary, "offset: 64", "summary includes offset");
    ExpectContains(summary, "bytes: 32", "summary includes bytes");
}

void TestReadBeforeCopySatisfiedByShimProducerModel() {
    tilexr::checker::EventLog events;
    events.Add(MakeEvent(tilexr::checker::EventKind::kCopy, 1, 0,
                         tilexr::checker::BufferRole::kCommData, 1, 0, 32, 64,
                         "producer copy into peer-owned comm data"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kRead, 0, 0,
                         tilexr::checker::BufferRole::kCommData, 1, 0, 48, 16,
                         "consumer read from same physical comm data window"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 0,
                 "shim producer copy should satisfy later comm-data read");
}

void TestBlockingTraceReadCanBeSatisfiedByFutureProducer() {
    tilexr::checker::EventLog events;
    tilexr::checker::Event read =
        MakeEvent(tilexr::checker::EventKind::kRead, 1, 0,
                  tilexr::checker::BufferRole::kCommData, 0, 0, 32, 16,
                  "blocking trace read");
    read.allow_future_producer = true;
    events.Add(read);
    events.Add(MakeEvent(tilexr::checker::EventKind::kCopy, 0, 0,
                         tilexr::checker::BufferRole::kCommData, 0, 0, 32, 16,
                         "producer copy reached after wait"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 0,
                 "blocking trace read should accept future producer copy");
}

void TestRegisteredCommBufferReadSatisfiedByWindowProducer() {
    tilexr::checker::EventLog events;
    events.Add(MakeEvent(tilexr::checker::EventKind::kWrite, 1, 0,
                         tilexr::checker::BufferRole::kRegisteredCommBuffer,
                         0, 0, 128, 64,
                         "ep dispatch src slot header"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kRead, 0, 1,
                         tilexr::checker::BufferRole::kRegisteredCommBuffer,
                         0, 0, 160, 16,
                         "ep combine read src slot header"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 0,
                 "registered comm buffer read should be satisfied by covering write");
}

void TestRegisteredCommBufferReadBeforeWriteReportsWindowSource() {
    tilexr::checker::EventLog events;
    tilexr::checker::Event read =
        MakeEvent(tilexr::checker::EventKind::kRead, 0, 1,
                  tilexr::checker::BufferRole::kRegisteredCommBuffer,
                  1, 0, 288, 64,
                  "ep combine read missing src slot header");
    read.source_file = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    read.source_line = 159;
    read.server = 0;
    read.peer_server = 1;
    events.Add(read);

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 1,
                 "registered comm buffer missing producer finding count");
    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "registered comm buffer top finding exists");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kReadBeforeWrite),
                    "registered comm buffer finding kind");
        ExpectEqInt(static_cast<int>(top->buffer_role),
                    static_cast<int>(tilexr::checker::BufferRole::kRegisteredCommBuffer),
                    "registered comm buffer finding role");
        ExpectEqInt(top->slot, 1, "registered comm buffer finding slot");
        ExpectEqInt(static_cast<int>(top->offset), 288,
                    "registered comm buffer finding offset");
        ExpectEqString(top->source_file,
                       "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp",
                       "registered comm buffer finding source");
        ExpectContains(top->message, "Registered communication window",
                       "registered comm buffer finding message");
    }
}

void TestStaleMagicDetectedEvenWhenExactStoreExists() {
    tilexr::checker::EventLog events;
    const uint64_t old_value = (0x20ULL << 32) | 0ULL;
    const uint64_t new_value = (0x21ULL << 32) | 0ULL;
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagStore, 0, 1,
                         tilexr::checker::BufferRole::kCommFlag, 3, old_value, 0,
                         sizeof(uint64_t), "older exact store"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagStore, 0, 1,
                         tilexr::checker::BufferRole::kCommFlag, 3, new_value, 0,
                         sizeof(uint64_t), "newer producer store"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagWait, 1, 0,
                         tilexr::checker::BufferRole::kCommFlag, 3, old_value, 0,
                         sizeof(uint64_t), "wait on stale magic"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 1, "stale magic finding count");
    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "stale magic top finding exists");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kFlagStaleMagic),
                    "stale magic should be reported despite exact matching store");
    }
}

void TestBroadcastFlagStoreSatisfiesConsumerWait() {
    tilexr::checker::EventLog events;
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagStore, 0, -1,
                         tilexr::checker::BufferRole::kCommFlag, 3, 0x20, 0,
                         sizeof(uint64_t), "broadcast store"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagWait, 1, 0,
                         tilexr::checker::BufferRole::kCommFlag, 3, 0x20, 0,
                         sizeof(uint64_t), "consumer wait"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 0,
                 "broadcast flag store should satisfy consumer wait");
}

void TestFutureFlagStoreSatisfiesBlockingWait() {
    tilexr::checker::EventLog events;
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagWait, 1, 0,
                         tilexr::checker::BufferRole::kCommFlag, 3, 0x20, 0,
                         sizeof(uint64_t), "blocking wait"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagStore, 0, -1,
                         tilexr::checker::BufferRole::kCommFlag, 3, 0x20, 0,
                         sizeof(uint64_t), "producer reaches flag later in trace"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 0,
                 "future flag store should satisfy blocking wait dependency");
}

void TestHigherFlagValueSameMagicSatisfiesWait() {
    tilexr::checker::EventLog events;
    const uint64_t wait_value = (0x20ULL << 32) | 1ULL;
    const uint64_t later_value = (0x20ULL << 32) | 3ULL;
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagWait, 1, 0,
                         tilexr::checker::BufferRole::kCommFlag, 3, wait_value, 0,
                         sizeof(uint64_t), "wait lower value"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagStore, 0, -1,
                         tilexr::checker::BufferRole::kCommFlag, 3, later_value, 0,
                         sizeof(uint64_t), "store higher value with same magic"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 0,
                 "higher flag value in same magic epoch should satisfy wait");
}

void TestMutualFlagWaitsReportDeadlock() {
    tilexr::checker::EventLog events;
    const uint64_t wait_value = (0x31ULL << 32) | 1ULL;
    tilexr::checker::Event wait0 = MakeEvent(tilexr::checker::EventKind::kFlagWait, 0, 1,
                                             tilexr::checker::BufferRole::kCommFlag, 7,
                                             wait_value, 0, sizeof(uint64_t),
                                             "rank0 waits rank1");
    wait0.server = 0;
    wait0.peer_server = 1;
    tilexr::checker::Event wait1 = MakeEvent(tilexr::checker::EventKind::kFlagWait, 1, 0,
                                             tilexr::checker::BufferRole::kCommFlag, 7,
                                             wait_value, 0, sizeof(uint64_t),
                                             "rank1 waits rank0");
    wait1.server = 1;
    wait1.peer_server = 0;
    events.Add(wait0);
    events.Add(wait1);

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(FindingCount(findings, tilexr::checker::FindingKind::kDeadlock),
                 1, "mutual flag waits deadlock finding count");
    ExpectEqSize(FindingCount(findings, tilexr::checker::FindingKind::kFlagNoProducer),
                 2, "mutual flag waits missing producer detail count");
    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "mutual flag waits top finding exists");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kDeadlock),
                    "mutual flag waits top finding kind");
        ExpectEqInt(top->rank, 0, "mutual flag waits deadlock rank");
        ExpectEqInt(top->peer_rank, 1, "mutual flag waits deadlock peer");
        ExpectContains(top->message, "rank 0 waits rank 1",
                       "mutual flag waits deadlock message first edge");
        ExpectContains(top->message, "rank 1 waits rank 0",
                       "mutual flag waits deadlock message second edge");
        ExpectContains(top->next_action, "Break the cycle",
                       "mutual flag waits deadlock next action");
    }
}

void TestPipeWaitWithoutProducerIsReported() {
    tilexr::checker::EventLog events;
    tilexr::checker::Event wait = MakeEvent(tilexr::checker::EventKind::kPipeWait, 0, -1,
                                            tilexr::checker::BufferRole::kMetadata, -1, 0, 0,
                                            0, "pipe wait without producer");
    wait.core = 3;
    wait.pipe = AscendC::PIPE_V;
    wait.event_id = AscendC::EVENT_ID1;
    events.Add(wait);

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(FindingCount(findings, tilexr::checker::FindingKind::kPipeWaitNoProducer),
                 1, "pipe wait without producer finding count");
    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "pipe wait missing producer top finding exists");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kPipeWaitNoProducer),
                    "pipe wait missing producer finding kind");
    }
}

void TestPipeWaitIsSatisfiedByEarlierSetOnSameCorePipeAndEvent() {
    tilexr::checker::EventLog events;
    tilexr::checker::Event set = MakeEvent(tilexr::checker::EventKind::kPipeSet, 0, -1,
                                           tilexr::checker::BufferRole::kMetadata, -1, 0, 0, 0,
                                           "pipe set");
    set.core = 3;
    set.pipe = AscendC::PIPE_V;
    set.event_id = AscendC::EVENT_ID1;
    events.Add(set);

    tilexr::checker::Event wait = set;
    wait.kind = tilexr::checker::EventKind::kPipeWait;
    wait.detail = "pipe wait";
    events.Add(wait);

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(FindingCount(findings, tilexr::checker::FindingKind::kPipeWaitNoProducer),
                 0, "matching pipe set should satisfy pipe wait");
}

void TestFindingPriorityFavorsStaleMagicOverNoProducer() {
    tilexr::checker::EventLog events;
    const uint64_t old_value = (0x20ULL << 32) | 0ULL;
    const uint64_t new_value = (0x21ULL << 32) | 0ULL;
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagStore, 0, 1,
                         tilexr::checker::BufferRole::kCommFlag, 3, new_value, 0,
                         sizeof(uint64_t), "newer producer store"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagWait, 1, 0,
                         tilexr::checker::BufferRole::kCommFlag, 3, old_value, 0,
                         sizeof(uint64_t), "stale wait without exact producer"));
    events.Add(MakeEvent(tilexr::checker::EventKind::kFlagWait, 0, 1,
                         tilexr::checker::BufferRole::kCommFlag, 5, 0x44, 0,
                         sizeof(uint64_t), "missing producer wait"));

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(events);
    ExpectEqSize(findings.findings().size(), 2, "priority comparison finding count");
    const tilexr::checker::Finding *top = findings.TopFinding();
    ExpectTrue(top != nullptr, "priority comparison top finding exists");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kFlagStaleMagic),
                    "stale magic should outrank no producer");
    }
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

void TestEventLogPreservesImportedEventIds() {
    tilexr::checker::EventLog events;
    tilexr::checker::Event imported =
        MakeEvent(tilexr::checker::EventKind::kRead, 0, 1,
                  tilexr::checker::BufferRole::kRegisteredCommBuffer, 3, 0,
                  128, 64, "imported event");
    imported.id = 42;
    events.Add(imported);
    tilexr::checker::Event generated =
        MakeEvent(tilexr::checker::EventKind::kWrite, 1, 0,
                  tilexr::checker::BufferRole::kRegisteredCommBuffer, 3, 0,
                  128, 64, "generated event after import");
    events.Add(generated);

    const std::vector<tilexr::checker::Event> &items = events.events();
    ExpectEqSize(items.size(), 2, "event log preserve ids count");
    if (items.size() == 2) {
        ExpectEqInt(static_cast<int>(items[0].id), 42,
                    "event log preserves imported id");
        ExpectEqInt(static_cast<int>(items[1].id), 43,
                    "event log next generated id follows imported id");
    }
}

}  // namespace

int main() {
    TestOrderingFindingsAndPriority();
    TestReadBeforeCopySummaryMatchesGolden();
    TestReadBeforeCopySatisfiedByShimProducerModel();
    TestBlockingTraceReadCanBeSatisfiedByFutureProducer();
    TestRegisteredCommBufferReadSatisfiedByWindowProducer();
    TestRegisteredCommBufferReadBeforeWriteReportsWindowSource();
    TestStaleMagicDetectedEvenWhenExactStoreExists();
    TestBroadcastFlagStoreSatisfiesConsumerWait();
    TestFutureFlagStoreSatisfiesBlockingWait();
    TestHigherFlagValueSameMagicSatisfiesWait();
    TestMutualFlagWaitsReportDeadlock();
    TestPipeWaitWithoutProducerIsReported();
    TestPipeWaitIsSatisfiedByEarlierSetOnSameCorePipeAndEvent();
    TestFindingPriorityFavorsStaleMagicOverNoProducer();
    TestDirectPeerUserBufferFinding();
    TestOutputMismatchKindsAndJson();
    TestEventsJsonlIncludesStableKinds();
    TestEventLogPreservesImportedEventIds();
    return g_failures == 0 ? 0 : 1;
}
