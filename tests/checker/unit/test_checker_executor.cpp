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
    test_case.magic = 0x65;
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
    store.magic = (0x99ULL << 32);
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
    wait.magic = (0x98ULL << 32);
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

void TestAllGatherRoundRobinRecordsServerTopology() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(4, 16, tilexr::checker::SchedulerMode::kRoundRobin);
    test_case.server_count = 2;
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allgather server topology status");

    bool saw_cross_server_read = false;
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == tilexr::checker::EventKind::kRead &&
            events[i].server == 0 && events[i].peer_server == 1) {
            saw_cross_server_read = true;
        }
    }
    ExpectTrue(saw_cross_server_read, "allgather records cross-server read topology");
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

void TestEpDispatchPassesAndRecordsRouteEvents() {
    tilexr::checker::CheckerCase test_case = MakeEpDispatchCase();
    const size_t input_bytes = static_cast<size_t>(test_case.bs * test_case.h) * sizeof(uint16_t);
    const size_t output_bytes = tilexr::checker::EpDispatchOutputBytes(test_case);
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(test_case.rank_size, input_bytes, output_bytes,
                                           output_bytes);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "ep dispatch status");
    ExpectEqSize(result.mismatches.size(), 0, "ep dispatch mismatches");
    ExpectEqSize(result.findings.findings().size(), 0, "ep dispatch findings");

    bool saw_cross_server_route = false;
    bool saw_assist_event = false;
    bool saw_window_header = false;
    bool saw_slot_header = false;
    bool saw_slot1_header_offset = false;
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == tilexr::checker::EventKind::kCopy &&
            events[i].server == 0 && events[i].peer_server == 1 &&
            events[i].detail.find("ep dispatch route payload") != std::string::npos) {
            saw_cross_server_route = true;
        }
        if (events[i].kind == tilexr::checker::EventKind::kWrite &&
            events[i].detail.find("ep dispatch assist") != std::string::npos) {
            saw_assist_event = true;
        }
        if (events[i].kind == tilexr::checker::EventKind::kWrite &&
            events[i].buffer_role == tilexr::checker::BufferRole::kRegisteredCommBuffer &&
            events[i].detail.find("ep dispatch window header") != std::string::npos &&
            events[i].source_file == "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp" &&
            events[i].source_line == 143) {
            saw_window_header = true;
        }
        if (events[i].kind == tilexr::checker::EventKind::kWrite &&
            events[i].buffer_role == tilexr::checker::BufferRole::kRegisteredCommBuffer &&
            events[i].detail.find("ep dispatch src slot header") != std::string::npos &&
            events[i].source_file == "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp" &&
            events[i].source_line == 159) {
            saw_slot_header = true;
            if (events[i].peer_rank == 1 && events[i].slot == 1 &&
                events[i].offset == 288 && events[i].bytes == 64) {
                saw_slot1_header_offset = true;
            }
        }
    }
    ExpectTrue(saw_cross_server_route, "ep dispatch records cross-server route event");
    ExpectTrue(saw_assist_event, "ep dispatch records assist write event");
    ExpectTrue(saw_window_header, "ep dispatch records production window header event");
    ExpectTrue(saw_slot_header, "ep dispatch records production slot header event");
    ExpectTrue(saw_slot1_header_offset, "ep dispatch records slot header production offset");

    const size_t assist_offset = tilexr::checker::EpDispatchMetadataOffset(
        test_case, tilexr::checker::EpDispatchMetadataRole::kAssist);
    const size_t recv_offset = tilexr::checker::EpDispatchMetadataOffset(
        test_case, tilexr::checker::EpDispatchMetadataRole::kRecvCounts);
    const size_t expert_offset = tilexr::checker::EpDispatchMetadataOffset(
        test_case, tilexr::checker::EpDispatchMetadataRole::kExpertTokenNums);
    int32_t assist_tuple[4] = {};
    int32_t recv_count = 0;
    int64_t expert_count = 0;
    ExpectTrue(world.UserOutput(1).ReadBytes(assist_offset, assist_tuple,
                                            sizeof(assist_tuple)).ok(),
               "ep dispatch reads assist metadata");
    ExpectEqInt(assist_tuple[0], 0, "ep dispatch assist src rank");
    ExpectEqInt(assist_tuple[1], 0, "ep dispatch assist token");
    ExpectEqInt(assist_tuple[2], 1, "ep dispatch assist topk");
    ExpectEqInt(assist_tuple[3], 3, "ep dispatch assist expert");
    ExpectTrue(world.UserOutput(1).ReadBytes(recv_offset, &recv_count,
                                            sizeof(recv_count)).ok(),
               "ep dispatch reads recv count metadata");
    ExpectEqInt(recv_count, 3, "ep dispatch recv count src rank 0");
    ExpectTrue(world.UserOutput(1).ReadBytes(expert_offset, &expert_count,
                                            sizeof(expert_count)).ok(),
               "ep dispatch reads expert count metadata");
    ExpectEqInt(static_cast<int>(expert_count), 2,
                "ep dispatch local expert 0 count");
}

void TestEpDispatchMetadataMismatchIsReported() {
    tilexr::checker::CheckerCase test_case = MakeEpDispatchCase();
    const size_t input_bytes = static_cast<size_t>(test_case.bs * test_case.h) * sizeof(uint16_t);
    const size_t output_bytes = tilexr::checker::EpDispatchOutputBytes(test_case);
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(test_case.rank_size, input_bytes, output_bytes,
                                           output_bytes);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "ep dispatch baseline status before metadata corruption");

    const size_t assist_offset = tilexr::checker::EpDispatchMetadataOffset(
        test_case, tilexr::checker::EpDispatchMetadataRole::kAssist);
    const int32_t bad_expert = -7;
    ExpectTrue(world.UserOutput(1).WriteBytes(assist_offset + 3 * sizeof(int32_t),
                                             &bad_expert, sizeof(bad_expert)).ok(),
               "ep dispatch corrupt assist expert");

    const std::vector<int32_t> expert_ids =
        tilexr::checker::DefaultEpDispatchExpertIds(test_case);
    const std::vector<tilexr::checker::OutputMismatch> mismatches =
        tilexr::checker::CompareEpDispatchOutput(world, test_case, expert_ids, 8);
    ExpectEqSize(mismatches.size(), 1, "ep dispatch metadata mismatch count");
    if (!mismatches.empty()) {
        ExpectEqInt(mismatches[0].rank, 1, "ep dispatch metadata mismatch rank");
        ExpectEqInt(mismatches[0].expected, 3, "ep dispatch metadata mismatch expected");
        ExpectEqInt(mismatches[0].actual, -7, "ep dispatch metadata mismatch actual");
        ExpectTrue(mismatches[0].context.find("assist") != std::string::npos,
                   "ep dispatch metadata mismatch context");
    }
}

void TestEpDispatchWindowEventsCanGuardFutureCombineReads() {
    tilexr::checker::CheckerCase test_case = MakeEpDispatchCase();
    const size_t input_bytes = static_cast<size_t>(test_case.bs * test_case.h) * sizeof(uint16_t);
    const size_t output_bytes = tilexr::checker::EpDispatchOutputBytes(test_case);
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(test_case.rank_size, input_bytes, output_bytes,
                                           output_bytes);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "ep dispatch window combine baseline status");

    tilexr::checker::Event combine_read;
    combine_read.kind = tilexr::checker::EventKind::kRead;
    combine_read.rank = 0;
    combine_read.peer_rank = 1;
    combine_read.server = world.ServerOfRank(0);
    combine_read.peer_server = world.ServerOfRank(1);
    combine_read.buffer_role = tilexr::checker::BufferRole::kRegisteredCommBuffer;
    combine_read.slot = 0;
    combine_read.offset = 64;
    combine_read.bytes = 64;
    combine_read.source_file = "src/ep/kernels/tilexr_ep_combine_kernel.cpp";
    combine_read.source_line = 211;
    combine_read.detail = "ep combine read src slot header";
    world.events().Add(combine_read);

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(world.events());
    ExpectEqSize(findings.findings().size(), 0,
                 "ep dispatch slot header should satisfy future combine read");

    tilexr::checker::Event missing_read = combine_read;
    missing_read.slot = 1;
    missing_read.offset = 288;
    missing_read.detail = "ep combine read missing future slot";
    tilexr::checker::EventLog missing_events;
    missing_events.Add(missing_read);
    tilexr::checker::FindingSet missing_findings =
        tilexr::checker::CheckOrdering(missing_events);
    ExpectEqSize(missing_findings.findings().size(), 1,
                 "ep combine read without dispatch window producer finding count");
    const tilexr::checker::Finding *top = missing_findings.TopFinding();
    ExpectTrue(top != nullptr, "ep combine missing producer top finding");
    if (top != nullptr) {
        ExpectEqInt(static_cast<int>(top->kind),
                    static_cast<int>(tilexr::checker::FindingKind::kReadBeforeWrite),
                    "ep combine missing producer finding kind");
        ExpectEqInt(top->slot, 1, "ep combine missing producer slot");
        ExpectEqInt(static_cast<int>(top->offset), 288,
                    "ep combine missing producer offset");
    }
}

void TestEpCombinePassesAndRecordsSourceLocatedReads() {
    tilexr::checker::CheckerCase test_case = MakeEpDispatchCase();
    test_case.op = tilexr::checker::CollectiveOp::kEpCombine;
    const size_t input_bytes = tilexr::checker::EpDispatchOutputBytes(test_case);
    const size_t output_bytes = tilexr::checker::EpCombineOutputBytes(test_case);
    const size_t comm_data_bytes = tilexr::checker::EpDispatchOutputBytes(test_case);
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(test_case.rank_size, input_bytes, output_bytes,
                                           comm_data_bytes);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "ep combine status");
    ExpectEqSize(result.mismatches.size(), 0, "ep combine mismatches");
    ExpectEqSize(result.findings.findings().size(), 0, "ep combine findings");

    bool saw_window_read = false;
    bool saw_output_write = false;
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == tilexr::checker::EventKind::kRead &&
            events[i].buffer_role == tilexr::checker::BufferRole::kRegisteredCommBuffer &&
            events[i].source_file == "src/ep/kernels/tilexr_ep_combine_kernel.cpp" &&
            events[i].source_line == 211) {
            saw_window_read = true;
        }
        if (events[i].kind == tilexr::checker::EventKind::kWrite &&
            events[i].buffer_role == tilexr::checker::BufferRole::kUserOutput &&
            events[i].source_file == "src/ep/kernels/tilexr_ep_combine_kernel.cpp") {
            saw_output_write = true;
        }
    }
    ExpectTrue(saw_window_read, "ep combine records source-located window reads");
    ExpectTrue(saw_output_write, "ep combine records source-located output writes");

    uint16_t token0_top1_h0 = 0;
    const size_t top1_offset =
        static_cast<size_t>((0 * test_case.top_k + 1) * test_case.h) * sizeof(uint16_t);
    ExpectTrue(world.UserOutput(0).ReadBytes(top1_offset, &token0_top1_h0,
                                            sizeof(token0_top1_h0)).ok(),
               "ep combine reads rank0 token0 top1 output");
    ExpectEqInt(token0_top1_h0, tilexr::checker::EpDispatchInputValue(0, 0, 0),
                "ep combine restores rank0 token0 top1 payload");
}

void TestAllReduceSchedulersProduceDifferentOrders() {
    tilexr::checker::CheckerCase serial_case =
        MakeAllReduceCase(tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::CheckerCase round_robin_case =
        MakeAllReduceCase(tilexr::checker::SchedulerMode::kRoundRobin);
    tilexr::checker::RankWorld serial_world = MakeWorld(serial_case);
    tilexr::checker::RankWorld round_robin_world = MakeWorld(round_robin_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult serial_result = executor.Run(&serial_world, serial_case);
    tilexr::checker::RunResult round_robin_result = executor.Run(&round_robin_world, round_robin_case);

    ExpectEqInt(static_cast<int>(serial_result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allreduce serial compare status");
    ExpectEqInt(static_cast<int>(round_robin_result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "allreduce round robin compare status");
    ExpectEqSize(serial_result.mismatches.size(), 0, "allreduce serial compare mismatches");
    ExpectEqSize(round_robin_result.mismatches.size(), 0, "allreduce round robin compare mismatches");
    ExpectEqSize(serial_result.findings.findings().size(), 0, "allreduce serial compare findings");
    ExpectEqSize(round_robin_result.findings.findings().size(), 0, "allreduce round robin compare findings");

    const std::vector<tilexr::checker::Event> &serial_events = serial_world.events().events();
    const std::vector<tilexr::checker::Event> &round_robin_events = round_robin_world.events().events();
    ExpectEqSize(serial_events.size(), static_cast<size_t>(12), "allreduce serial compare event count");
    ExpectEqSize(round_robin_events.size(), static_cast<size_t>(12), "allreduce round robin compare event count");
    if (serial_events.size() >= 10 && round_robin_events.size() >= 10) {
        ExpectEvent(serial_events[6], tilexr::checker::EventKind::kRead, 0,
                    "allreduce serial first read rank 0");
        ExpectEvent(serial_events[8], tilexr::checker::EventKind::kWrite, 0,
                    "allreduce serial first write rank 0");
        ExpectEvent(serial_events[9], tilexr::checker::EventKind::kRead, 1,
                    "allreduce serial rank 1 starts after rank 0 write");
        ExpectEvent(round_robin_events[6], tilexr::checker::EventKind::kRead, 0,
                    "allreduce round robin first read rank 0");
        ExpectEvent(round_robin_events[7], tilexr::checker::EventKind::kRead, 1,
                    "allreduce round robin interleaves rank 1 read");
        ExpectEvent(round_robin_events[10], tilexr::checker::EventKind::kWrite, 0,
                    "allreduce round robin rank 0 write after all reads");
        ExpectTrue(serial_events[7].rank != round_robin_events[7].rank,
                   "allreduce scheduler orders differ in read phase rank order");
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

void TestRunSerialMatchesDirectOrderingDiagnostics() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    tilexr::checker::FindingSet direct_findings = tilexr::checker::CheckOrdering(world.events());

    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "serial run status");
    ExpectEqSize(result.findings.findings().size(), direct_findings.findings().size(),
                 "serial run/direct ordering finding count match");
    ExpectEqSize(FindingCount(result.findings, tilexr::checker::FindingKind::kReadBeforeCopy),
                 FindingCount(direct_findings, tilexr::checker::FindingKind::kReadBeforeCopy),
                 "serial run/direct read-before-copy finding count match");
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

void TestInjectedReadBeforeCopyProducesFinding() {
    tilexr::checker::CheckerCase test_case =
        MakeAllGatherCase(2, 16, tilexr::checker::SchedulerMode::kSerial);
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    tilexr::checker::CollectiveExecutor executor;

    tilexr::checker::RunResult result = executor.Run(&world, test_case);
    InjectCommDataReadWithoutProducer(&world, 1, 0);
    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(world.events());
    ExpectEqInt(static_cast<int>(result.status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                "baseline injected read-before-copy run ok");
    ExpectTrue(HasFindingKind(findings, tilexr::checker::FindingKind::kReadBeforeCopy),
               "read-before-copy finding");
}

}  // namespace

int main() {
    TestAllGatherSerialPasses();
    TestAllGatherRoundRobinPasses();
    TestAllGatherRoundRobinRecordsServerTopology();
    TestAllGatherSchedulersProduceDifferentOrders();
    TestAllReducePassesForBothSchedulers();
    TestEpDispatchPassesAndRecordsRouteEvents();
    TestEpDispatchMetadataMismatchIsReported();
    TestEpDispatchWindowEventsCanGuardFutureCombineReads();
    TestEpCombinePassesAndRecordsSourceLocatedReads();
    TestAllReduceSchedulersProduceDifferentOrders();
    TestUnsupportedDatatypeReturnsFinding();
    TestRunSerialMatchesDirectOrderingDiagnostics();
    TestInjectedPeerUserReadProducesFinding();
    TestInjectedStaleMagicProducesFinding();
    TestInjectedReadBeforeCopyProducesFinding();
    return g_failures == 0 ? 0 : 1;
}
