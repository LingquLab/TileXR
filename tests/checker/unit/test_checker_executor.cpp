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
    ExpectTrue(HasFindingKind(result.findings, tilexr::checker::FindingKind::kUnsupportedApi),
               "unsupported datatype finding kind");
}

void TestInjectedPeerUserReadProducesFinding() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    InjectPeerUserRead(&world, 1, 0);
    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(world.events());
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "baseline injected peer run ok");
    ExpectTrue(HasFindingKind(findings, tilexr::checker::FindingKind::kDirectPeerUserBuffer),
               "peer user read finding");
}

void TestInjectedStaleMagicProducesFinding() {
    tilexr::checker::CheckerCase test_case =
        MakeAllReduceCase(tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    InjectStaleMagicWait(&world, 1, 0, 1);
    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(world.events());
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "baseline injected stale run ok");
    ExpectTrue(HasFindingKind(findings, tilexr::checker::FindingKind::kFlagStaleMagic),
               "stale magic finding");
}

}  // namespace

int main() {
    TestAllGatherSerialPasses();
    TestAllGatherRoundRobinPasses();
    TestAllReducePassesForBothSchedulers();
    TestUnsupportedDatatypeReturnsFinding();
    TestInjectedPeerUserReadProducesFinding();
    TestInjectedStaleMagicProducesFinding();
    return g_failures == 0 ? 0 : 1;
}
