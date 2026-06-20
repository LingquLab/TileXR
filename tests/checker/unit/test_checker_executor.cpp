#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

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

void ExpectEqSize(size_t actual, size_t expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

void ExpectContains(const std::string &text, const std::string &needle,
                    const char *message) {
    if (text.find(needle) == std::string::npos) {
        std::cerr << message << ": missing " << needle << " in " << text << "\n";
        ++g_failures;
    }
}

tilexr::checker::CheckerCase MakeAllGatherCase(int rank_size, int64_t count,
                                               tilexr::checker::SchedulerMode scheduler) {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllGather;
    test_case.rank_size = rank_size;
    test_case.count = count;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = scheduler;
    test_case.magic = 0x41;
    return test_case;
}

tilexr::checker::CheckerCase MakeAllReduceCase(tilexr::checker::SchedulerMode scheduler) {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 2;
    test_case.count = 16;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = scheduler;
    test_case.magic = 0x52;
    return test_case;
}

tilexr::checker::RankWorld MakeWorld(const tilexr::checker::CheckerCase &test_case) {
    const size_t input_bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    const size_t output_bytes =
        static_cast<size_t>(test_case.op == tilexr::checker::CollectiveOp::kAllGather
                                ? test_case.rank_size * test_case.count
                                : test_case.count) *
        sizeof(int32_t);
    return tilexr::checker::RankWorld::Create(test_case.rank_size, input_bytes, output_bytes,
                                              input_bytes);
}

bool HasFindingKind(const tilexr::checker::FindingSet &findings,
                    tilexr::checker::FindingKind kind) {
    const std::vector<tilexr::checker::Finding> &items = findings.findings();
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].kind == kind) {
            return true;
        }
    }
    return false;
}

void ExpectEvent(const tilexr::checker::Event &event, tilexr::checker::EventKind kind, int rank,
                 const char *message) {
    if (event.kind != kind || event.rank != rank) {
        std::cerr << message << ": actual_kind=" << static_cast<int>(event.kind)
                  << " actual_rank=" << event.rank
                  << " expected_kind=" << static_cast<int>(kind)
                  << " expected_rank=" << rank << "\n";
        ++g_failures;
    }
}

void InjectPeerUserRead(tilexr::checker::RankWorld *world, int reader_rank, int owner_rank) {
    tilexr::checker::Event event;
    event.kind = tilexr::checker::EventKind::kRead;
    event.rank = reader_rank;
    event.peer_rank = owner_rank;
    event.buffer_role = tilexr::checker::BufferRole::kUserInput;
    event.slot = reader_rank;
    event.offset = 0;
    event.bytes = sizeof(int32_t);
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = "injected peer user read";
    world->events().Add(event);
}

void InjectStaleMagicWait(tilexr::checker::RankWorld *world, int rank, int peer_rank, int slot) {
    tilexr::checker::Event store;
    store.kind = tilexr::checker::EventKind::kFlagStore;
    store.rank = peer_rank;
    store.peer_rank = rank;
    store.buffer_role = tilexr::checker::BufferRole::kCommFlag;
    store.slot = slot;
    store.magic = 0x99;
    store.offset = 0;
    store.bytes = sizeof(uint64_t);
    store.source_file = __FILE__;
    store.source_line = __LINE__;
    store.detail = "injected newer store";
    world->events().Add(store);

    tilexr::checker::Event wait;
    wait.kind = tilexr::checker::EventKind::kFlagWait;
    wait.rank = rank;
    wait.peer_rank = peer_rank;
    wait.buffer_role = tilexr::checker::BufferRole::kCommFlag;
    wait.slot = slot;
    wait.magic = 0x98;
    wait.offset = 0;
    wait.bytes = sizeof(uint64_t);
    wait.source_file = __FILE__;
    wait.source_line = __LINE__;
    wait.detail = "injected stale wait";
    world->events().Add(wait);
}

void InjectCommDataReadWithoutProducer(tilexr::checker::RankWorld *world, int rank, int owner_rank) {
    tilexr::checker::Event event;
    event.kind = tilexr::checker::EventKind::kRead;
    event.rank = rank;
    event.peer_rank = owner_rank;
    event.buffer_role = tilexr::checker::BufferRole::kCommData;
    event.slot = rank;
    event.offset = 0;
    event.bytes = sizeof(int32_t);
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = "injected comm-data read without producer";
    world->events().Add(event);
}

void TestAllGatherSerialPasses() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allgather serial status");
    ExpectEqSize(result.mismatches.size(), 0, "allgather serial mismatches");
    ExpectEqSize(result.findings.findings().size(), 0, "allgather serial findings");
    ExpectEqSize(result.event_count, static_cast<size_t>(14), "allgather serial event count");
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), static_cast<size_t>(14), "allgather serial events size");
    if (events.size() >= 8) {
        ExpectEvent(events[0], tilexr::checker::EventKind::kCopy, 0, "serial event 0");
        ExpectEvent(events[1], tilexr::checker::EventKind::kCopy, 1, "serial event 1");
        ExpectEvent(events[2], tilexr::checker::EventKind::kFlagStore, 0, "serial event 2");
        ExpectEvent(events[3], tilexr::checker::EventKind::kFlagStore, 1, "serial event 3");
        ExpectEvent(events[4], tilexr::checker::EventKind::kFlagWait, 0, "serial event 4");
        ExpectEvent(events[5], tilexr::checker::EventKind::kFlagWait, 1, "serial event 5");
        ExpectEvent(events[6], tilexr::checker::EventKind::kRead, 0, "serial event 6");
        ExpectEvent(events[7], tilexr::checker::EventKind::kWrite, 0, "serial event 7");
    }
}

void TestAllGatherRoundRobinPasses() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(4, 16, tilexr::checker::SchedulerMode::kRoundRobin);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allgather round robin status");
    ExpectEqSize(result.mismatches.size(), 0, "allgather round robin mismatches");
    ExpectEqSize(result.findings.findings().size(), 0, "allgather round robin findings");
    ExpectEqSize(result.event_count, static_cast<size_t>(60), "allgather round robin event count");
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), static_cast<size_t>(60), "allgather round robin events size");
    if (events.size() >= 8) {
        ExpectEvent(events[0], tilexr::checker::EventKind::kCopy, 0, "round robin event 0");
        ExpectEvent(events[1], tilexr::checker::EventKind::kCopy, 1, "round robin event 1");
        ExpectEvent(events[2], tilexr::checker::EventKind::kCopy, 2, "round robin event 2");
        ExpectEvent(events[3], tilexr::checker::EventKind::kCopy, 3, "round robin event 3");
        ExpectEvent(events[4], tilexr::checker::EventKind::kFlagStore, 0, "round robin event 4");
        ExpectEvent(events[5], tilexr::checker::EventKind::kFlagStore, 0, "round robin event 5");
        ExpectEvent(events[6], tilexr::checker::EventKind::kFlagStore, 0, "round robin event 6");
        ExpectEvent(events[7], tilexr::checker::EventKind::kFlagStore, 1, "round robin event 7");
    }
}

void TestAllGatherSchedulersProduceDifferentOrders() {
    tilexr::checker::CheckerCase serial_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::CheckerCase round_robin_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kRoundRobin);
    tilexr::checker::RankWorld serial_world = MakeWorld(serial_case);
    tilexr::checker::RankWorld round_robin_world = MakeWorld(round_robin_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult serial_result = executor.Run(&serial_world, serial_case);
    tilexr::checker::RunResult round_robin_result = executor.Run(&round_robin_world, round_robin_case);

    ExpectEqInt(static_cast<int>(serial_result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "serial compare status");
    ExpectEqInt(static_cast<int>(round_robin_result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "round robin compare status");

    const std::vector<tilexr::checker::Event> &serial_events = serial_world.events().events();
    const std::vector<tilexr::checker::Event> &round_robin_events = round_robin_world.events().events();
    ExpectEqSize(serial_events.size(), static_cast<size_t>(14), "serial compare event count");
    ExpectEqSize(round_robin_events.size(), static_cast<size_t>(14), "round robin compare event count");
    if (serial_events.size() >= 8 && round_robin_events.size() >= 5) {
        ExpectEvent(serial_events[10], tilexr::checker::EventKind::kRead, 1,
                    "serial rank 1 starts reads after rank 0 reads");
        ExpectEvent(round_robin_events[0], tilexr::checker::EventKind::kCopy, 0,
                    "round robin first publish rank 0");
        ExpectEvent(round_robin_events[1], tilexr::checker::EventKind::kCopy, 1,
                    "round robin second publish rank 1");
        ExpectEvent(round_robin_events[2], tilexr::checker::EventKind::kFlagStore, 0,
                    "round robin stores begin after all publishes");
        ExpectEvent(round_robin_events[8], tilexr::checker::EventKind::kRead, 1,
                    "round robin interleaves rank 1 on peer 0 before rank 0 peer 1");
        ExpectTrue(serial_events[8].rank != round_robin_events[8].rank,
                   "scheduler orders differ in read phase rank order");
    }
}

void TestAllReducePassesForBothSchedulers() {
    const tilexr::checker::SchedulerMode schedulers[] = {
        tilexr::checker::SchedulerMode::kSerial,
        tilexr::checker::SchedulerMode::kRoundRobin,
    };

    for (size_t i = 0; i < sizeof(schedulers) / sizeof(schedulers[0]); ++i) {
        tilexr::checker::CheckerCase test_case = MakeAllReduceCase(schedulers[i]);
        tilexr::checker::RankWorld world = MakeWorld(test_case);
        tilexr::checker::CollectiveExecutor executor;

        tilexr::checker::RunResult result = executor.Run(&world, test_case);
        ExpectEqInt(static_cast<int>(result.status.code),
                    static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                    "allreduce status");
        ExpectEqSize(result.mismatches.size(), 0, "allreduce mismatches");
        ExpectEqSize(result.findings.findings().size(), 0, "allreduce findings");
        ExpectEqSize(result.event_count, static_cast<size_t>(12), "allreduce event count");
    }
}

void TestUnsupportedDatatypeReturnsFinding() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kSerial);
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "unsupported datatype status");
    ExpectEqSize(result.mismatches.size(), 0, "unsupported datatype mismatches");
    ExpectEqSize(result.event_count, static_cast<size_t>(0), "unsupported datatype event count");
    ExpectTrue(HasFindingKind(result.findings, tilexr::checker::FindingKind::kUnsupportedApi),
               "unsupported datatype finding kind");
}

void TestRunSerialReportsInjectedPeerUserReadFinding() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;
    executor.SetPostTraceHookForTest(
        [](tilexr::checker::RankWorld *hook_world) { InjectPeerUserRead(hook_world, 1, 0); });

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kFail),
                "serial injected peer run fails");
    ExpectTrue(HasFindingKind(result.findings,
                              tilexr::checker::FindingKind::kDirectPeerUserBuffer),
               "serial run reports peer user read finding");
}

void TestRunSerialReportsInjectedStaleMagicFinding() {
    tilexr::checker::CheckerCase test_case =
        MakeAllReduceCase(tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;
    executor.SetPostTraceHookForTest(
        [](tilexr::checker::RankWorld *hook_world) { InjectStaleMagicWait(hook_world, 1, 0, 1); });

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kFail),
                "serial injected stale run fails");
    ExpectTrue(HasFindingKind(result.findings, tilexr::checker::FindingKind::kFlagStaleMagic),
               "serial run reports stale magic finding");
}

void TestRunSerialReportsInjectedReadBeforeCopyFinding() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;
    executor.SetPostTraceHookForTest(
        [](tilexr::checker::RankWorld *hook_world) {
            InjectCommDataReadWithoutProducer(hook_world, 1, 0);
        });

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kFail),
                "serial injected read-before-copy run fails");
    ExpectTrue(HasFindingKind(result.findings, tilexr::checker::FindingKind::kReadBeforeCopy),
               "serial run reports read-before-copy finding");
}

}  // namespace

int main() {
    TestAllGatherSerialPasses();
    TestAllGatherRoundRobinPasses();
    TestAllGatherSchedulersProduceDifferentOrders();
    TestAllReducePassesForBothSchedulers();
    TestUnsupportedDatatypeReturnsFinding();
    TestRunSerialReportsInjectedPeerUserReadFinding();
    TestRunSerialReportsInjectedStaleMagicFinding();
    TestRunSerialReportsInjectedReadBeforeCopyFinding();
    return g_failures == 0 ? 0 : 1;
}
