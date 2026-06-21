#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define TILEXR_CHECKER_ENABLE_PIPE_TRACE_HOOK
#include "kernel_operator.h"
#include "tilexr/checker/diagnostics.h"
#include "tilexr/checker/report.h"
#include "tilexr/checker/trace_runtime.h"
#include "tilexr/checker/shim_runtime.h"
#include "tilexr/checker/world.h"

namespace {

int g_failures = 0;

void UseCheckerShimTypes() {
    AscendC::GlobalTensorBase global;
    AscendC::LocalTensorBase local;
    (void)global;
    (void)local;
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

void ExpectEqU64(uint64_t actual, uint64_t expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

void ExpectEqU8(uint8_t actual, uint8_t expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << static_cast<int>(actual)
                  << " expected=" << static_cast<int>(expected) << "\n";
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

void ExpectNotContains(const std::string &text, const std::string &needle,
                       const char *message) {
    if (text.find(needle) != std::string::npos) {
        std::cerr << message << ": unexpectedly found " << needle << "\n";
        ++g_failures;
    }
}

int32_t ReadInt32(const tilexr::checker::ByteBuffer &buffer, size_t index) {
    int32_t value = 0;
    tilexr::checker::CheckerStatus status = buffer.ReadInt32(index, &value);
    if (!status.ok()) {
        std::cerr << "read int32 failed: " << status.message << "\n";
        ++g_failures;
    }
    return value;
}

void TestCopyMovesBytesAndLogsEvent() {
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(2, 16, 16, 16);
    tilexr::checker::ShimRuntime runtime(&world);

    ExpectTrue(world.UserInput(1).WriteInt32(0, 7).ok(), "seed source input");
    ExpectTrue(world.UserInput(1).WriteInt32(1, 11).ok(), "seed source input second");

    tilexr::checker::CheckerStatus status =
        runtime.Copy(1, 0, tilexr::checker::BufferRole::kCommData, 0,
                     tilexr::checker::BufferRole::kUserInput, 0,
                     2 * sizeof(int32_t), TILEXR_CHECKER_HERE, "copy user input to peer comm");
    ExpectTrue(status.ok(), "copy status ok");

    ExpectEqInt(ReadInt32(world.CommData(0, 1), 0), 7, "copied first int32");
    ExpectEqInt(ReadInt32(world.CommData(0, 1), 1), 11, "copied second int32");

    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 1, "copy event count");
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kCopy),
                "copy event kind");
    ExpectEqInt(events[0].rank, 1, "copy event rank");
    ExpectEqInt(events[0].peer_rank, 0, "copy event peer rank");
    ExpectEqInt(static_cast<int>(events[0].buffer_role),
                static_cast<int>(tilexr::checker::BufferRole::kCommData),
                "copy event buffer role");
    ExpectEqInt(events[0].slot, 1, "copy event slot");
    ExpectEqSize(events[0].bytes, 2 * sizeof(int32_t), "copy event bytes");
    ExpectEqString(events[0].detail, "copy user input to peer comm", "copy event detail");
    ExpectTrue(!events[0].source_file.empty(), "copy event source file");
    ExpectTrue(events[0].source_line > 0, "copy event source line");
}

void TestFlagEventsCaptureMagic() {
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(2, 16, 16, 16);
    tilexr::checker::ShimRuntime runtime(&world);
    UseCheckerShimTypes();

    const int rank = 1;
    const int peer_rank = 0;
    const int slot = 1;
    const uint64_t magic = 0x1234ULL;

    tilexr::checker::CheckerStatus store =
        runtime.StoreFlag(rank, peer_rank, slot, magic, TILEXR_CHECKER_HERE, "store flag");
    tilexr::checker::CheckerStatus wait =
        runtime.WaitFlag(peer_rank, rank, slot, magic, TILEXR_CHECKER_HERE, "wait flag");
    ExpectTrue(store.ok(), "store flag status ok");
    ExpectTrue(wait.ok(), "wait flag status ok");

    uint64_t stored_magic_at_slot = 0;
    ExpectTrue(
        world.CommFlag(rank, slot).ReadBytes(0, &stored_magic_at_slot, sizeof(stored_magic_at_slot))
            .ok(),
        "read stored flag bytes at slot");
    ExpectEqU64(stored_magic_at_slot, magic, "stored flag value at slot");

    if (slot != peer_rank) {
        uint64_t stored_magic_at_peer_slot = 0;
        ExpectTrue(world.CommFlag(rank, peer_rank)
                       .ReadBytes(0, &stored_magic_at_peer_slot,
                                  sizeof(stored_magic_at_peer_slot))
                       .ok(),
                   "read stored flag bytes at peer slot");
        ExpectEqU64(stored_magic_at_peer_slot, 0, "peer slot remains untouched");
    }

    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 2, "flag event count");
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kFlagStore),
                "store event kind");
    ExpectEqInt(events[0].rank, rank, "store event rank");
    ExpectEqInt(events[0].peer_rank, peer_rank, "store event peer rank");
    ExpectEqU64(events[0].magic, magic, "store event magic");
    ExpectEqInt(events[0].slot, slot, "store event slot");
    ExpectEqInt(static_cast<int>(events[0].buffer_role),
                static_cast<int>(tilexr::checker::BufferRole::kCommFlag),
                "store event role");
    ExpectEqSize(events[0].offset, 0, "store event offset");
    ExpectEqSize(events[0].bytes, sizeof(magic), "store event bytes");
    ExpectEqInt(static_cast<int>(events[1].kind),
                static_cast<int>(tilexr::checker::EventKind::kFlagWait),
                "wait event kind");
    ExpectEqInt(events[1].rank, peer_rank, "wait event rank");
    ExpectEqInt(events[1].peer_rank, rank, "wait event peer rank");
    ExpectEqU64(events[1].magic, magic, "wait event magic");
    ExpectEqInt(events[1].slot, slot, "wait event slot");
    ExpectEqInt(static_cast<int>(events[1].buffer_role),
                static_cast<int>(tilexr::checker::BufferRole::kCommFlag),
                "wait event role");
    ExpectEqSize(events[1].offset, 0, "wait event offset");
    ExpectEqSize(events[1].bytes, sizeof(magic), "wait event bytes");
}

void TestNullWorldAndUnsupportedRoleFail() {
    tilexr::checker::ShimRuntime runtime(nullptr);

    tilexr::checker::CheckerStatus null_status =
        runtime.Barrier(0, 0, TILEXR_CHECKER_HERE, "null barrier");
    ExpectTrue(!null_status.ok(), "null runtime should fail");

    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(2, 16, 16, 16);
    tilexr::checker::ShimRuntime world_runtime(&world);
    tilexr::checker::CheckerStatus unsupported =
        world_runtime.RecordRead(0, 1, tilexr::checker::BufferRole::kLocalUb, 0, 4,
                                 TILEXR_CHECKER_HERE, "unsupported role");
    ExpectTrue(!unsupported.ok(), "unsupported role should fail");
}

void TestRecordWriteAndBarrierEvents() {
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(2, 16, 16, 16);
    tilexr::checker::ShimRuntime runtime(&world);

    tilexr::checker::CheckerStatus write =
        runtime.RecordWrite(0, 1, tilexr::checker::BufferRole::kUserOutput, 4, 8,
                            TILEXR_CHECKER_HERE, "record write");
    tilexr::checker::CheckerStatus barrier =
        runtime.Barrier(0, 3, TILEXR_CHECKER_HERE, "barrier event");
    ExpectTrue(write.ok(), "record write status ok");
    ExpectTrue(barrier.ok(), "barrier status ok");

    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 2, "write and barrier event count");
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kWrite),
                "write event kind");
    ExpectEqInt(events[0].slot, 0, "write event slot");
    ExpectEqInt(static_cast<int>(events[1].kind),
                static_cast<int>(tilexr::checker::EventKind::kBarrier),
                "barrier event kind");
    ExpectEqInt(events[1].core, 3, "barrier event core");
}

void TestEventsCaptureServerTopology() {
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(4, 16, 16, 16);
    world.ConfigureServers(2);
    tilexr::checker::ShimRuntime runtime(&world);

    ExpectTrue(world.UserInput(0).WriteInt32(0, 7).ok(), "seed cross-server copy input");
    tilexr::checker::CheckerStatus status =
        runtime.Copy(0, 3, tilexr::checker::BufferRole::kCommData, 0,
                     tilexr::checker::BufferRole::kUserInput, 0,
                     sizeof(int32_t), TILEXR_CHECKER_HERE,
                     "copy to cross-server peer comm");
    ExpectTrue(status.ok(), "cross-server copy status ok");

    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 1, "cross-server event count");
    if (events.empty()) {
        return;
    }
    ExpectEqInt(events[0].server, 0, "cross-server event rank server");
    ExpectEqInt(events[0].peer_server, 1, "cross-server event peer server");
    const std::string jsonl = tilexr::checker::RenderEventsJsonl(world.events());
    ExpectContains(jsonl, "\"server\":0", "events json includes server");
    ExpectContains(jsonl, "\"peer_server\":1", "events json includes peer server");
}

void TestCheckerLocalShimIncludePathOnly() {
    const std::string checker_cmake = ReadFile(SourcePath("tests/checker/CMakeLists.txt"));
    const std::string tools_cmake = ReadFile(SourcePath("tools/checker/CMakeLists.txt"));

    ExpectContains(checker_cmake, "test_tilexr_checker_shim_events",
                   "checker test target wired");
    ExpectContains(checker_cmake, "tools/checker/shim", "checker shim include path wired");
    ExpectNotContains(tools_cmake, "tools/checker/shim", "checker core should not export shim");
}

void TestTraceAdapterKeepsAlgorithmShimsThin() {
    const std::string common_shim =
        ReadFile(SourcePath("tools/checker/shim/tilexr/checker/collective_trace_shim.h"));
    const std::string allreduce_adapter =
        ReadFile(SourcePath("tools/checker/shim/tilexr/checker/allreduce_big_data_trace_shim.h"));
    const std::string hdb_adapter =
        ReadFile(SourcePath("tools/checker/shim/tilexr/checker/allgather_hdb_trace_shim.h"));

    ExpectNotContains(common_shim, "#include \"allreduce_big_data.h\"",
                      "common trace shim should not include a production algorithm");
    ExpectNotContains(common_shim, "#define CpGM2GMPingPong",
                      "common trace shim should not own production call macros");

    ExpectContains(allreduce_adapter, "TILEXR_CHECKER_TRACE_TARGET_HEADER",
                   "allreduce adapter declares target header");
    ExpectContains(allreduce_adapter, "collective_trace_adapter.h",
                   "allreduce adapter uses generic trace adapter");
    ExpectNotContains(allreduce_adapter, "#define CpGM2GMPingPong",
                      "allreduce adapter should not duplicate trace macros");

    ExpectContains(hdb_adapter, "TILEXR_CHECKER_TRACE_TARGET_HEADER",
                   "hdb adapter declares target header");
    ExpectContains(hdb_adapter, "collective_trace_adapter.h",
                   "hdb adapter uses generic trace adapter");
    ExpectNotContains(hdb_adapter, "#define CpGM2GMPingPong",
                      "hdb adapter should not duplicate trace macros");
}

void TestAscendCPipePrimitivesRecordTraceEvents() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllGather;
    test_case.rank_size = 2;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(2, 16, 16, 16);
    tilexr::checker::TraceRuntime runtime(&world, test_case);
    runtime.SetKernelContext(1, 3, 8);
    tilexr::checker::TraceRuntime::SetCurrent(&runtime);

    const int set_line = __LINE__ + 1;
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(AscendC::EVENT_ID1);
    const int wait_line = __LINE__ + 1;
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(AscendC::EVENT_ID1);
    const int barrier_line = __LINE__ + 1;
    AscendC::PipeBarrier<AscendC::PIPE_V>();

    tilexr::checker::TraceRuntime::SetCurrent(nullptr);
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 3, "pipe primitive event count");
    if (events.size() < 3) {
        return;
    }
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kPipeSet),
                "pipe set event kind");
    ExpectEqInt(static_cast<int>(events[1].kind),
                static_cast<int>(tilexr::checker::EventKind::kPipeWait),
                "pipe wait event kind");
    ExpectEqInt(static_cast<int>(events[2].kind),
                static_cast<int>(tilexr::checker::EventKind::kPipeBarrier),
                "pipe barrier event kind");
    ExpectEqInt(events[0].rank, 1, "pipe set rank");
    ExpectEqInt(events[0].core, 3, "pipe set core");
    ExpectEqInt(events[0].event_id, AscendC::EVENT_ID1, "pipe set event id");
    ExpectEqInt(events[0].pipe, AscendC::PIPE_V, "pipe set pipe");
    ExpectEqInt(events[1].pipe, AscendC::PIPE_V, "pipe wait pipe");
    ExpectEqInt(events[2].pipe, AscendC::PIPE_V, "pipe barrier pipe");
    ExpectContains(events[0].source_file, "test_checker_shim_events.cpp",
                   "pipe set source file");
    ExpectEqInt(events[0].source_line, set_line, "pipe set source line");
    ExpectEqInt(events[1].source_line, wait_line, "pipe wait source line");
    ExpectEqInt(events[2].source_line, barrier_line, "pipe barrier source line");
    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(world.events());
    ExpectEqSize(findings.findings().size(), 0, "matched pipe wait findings");
}

void TestTraceRuntimeReportsUnresolvedGmCopyAddress() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 1;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(1, 16, 16, 16);
    tilexr::checker::TraceRuntime runtime(&world, test_case);
    runtime.SetKernelContext(0, 0, 1);

    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {0, 0, 0, 0};
    const int copy_line = __LINE__ + 3;
    tilexr::checker::CheckerStatus status =
        runtime.RecordCopy(dst, src, sizeof(src), TileXR::Op::ADD,
                           TILEXR_CHECKER_HERE, "unregistered gm copy");

    ExpectTrue(status.ok(), "unresolved gm copy record status");
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 1, "unresolved gm copy diagnostic event count");
    if (events.empty()) {
        return;
    }
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kDiagnostic),
                "unresolved gm copy diagnostic event kind");
    ExpectContains(events[0].detail, "unresolved trace address",
                   "unresolved gm copy diagnostic detail");
    ExpectContains(events[0].source_file, "test_checker_shim_events.cpp",
                   "unresolved gm copy diagnostic source file");
    ExpectEqInt(events[0].source_line, copy_line,
                "unresolved gm copy diagnostic source line");

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(world.events());
    ExpectEqSize(findings.findings().size(), 1,
                 "unresolved gm copy unsupported finding count");
    if (findings.findings().empty()) {
        return;
    }
    ExpectEqInt(static_cast<int>(findings.findings()[0].kind),
                static_cast<int>(tilexr::checker::FindingKind::kUnsupportedApi),
                "unresolved gm copy unsupported finding kind");
    ExpectContains(findings.findings()[0].next_action, "RegisterCollectiveTraceRanges",
                   "unresolved gm copy next action");
}

void TestRawAscendCDataCopyRecordsTraceRuntimeCopy() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 1;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(1, 16, 16, 16);
    tilexr::checker::TraceRuntime runtime(&world, test_case);
    runtime.SetKernelContext(0, 2, 4);

    uint8_t src[16] = {0};
    uint8_t dst[16] = {0};
    runtime.AddRange(src, sizeof(src), tilexr::checker::BufferRole::kUserInput,
                     0, 0);
    runtime.AddRange(dst, sizeof(dst), tilexr::checker::BufferRole::kCommData,
                     0, 0);
    tilexr::checker::TraceRuntime::SetCurrent(&runtime);
    const int copy_line = __LINE__ + 1;
    AscendC::DataCopy(dst, src, sizeof(src));
    tilexr::checker::TraceRuntime::SetCurrent(nullptr);

    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 2, "raw DataCopy trace event count");
    if (events.size() < 2) {
        return;
    }
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kRead),
                "raw DataCopy read event kind");
    ExpectEqInt(static_cast<int>(events[1].kind),
                static_cast<int>(tilexr::checker::EventKind::kCopy),
                "raw DataCopy copy event kind");
    ExpectEqInt(events[0].core, 2, "raw DataCopy read core");
    ExpectEqInt(events[1].core, 2, "raw DataCopy copy core");
    ExpectEqSize(events[1].bytes, sizeof(src), "raw DataCopy copy bytes");
    ExpectContains(events[1].source_file, "test_checker_shim_events.cpp",
                   "raw DataCopy copy source file");
    ExpectEqInt(events[1].source_line, copy_line,
                "raw DataCopy copy source line");
    ExpectContains(events[1].detail, "AscendC::DataCopy",
                   "raw DataCopy copy detail");
}

void TestRawAscendCDataCopyReportsUnregisteredAddress() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 1;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(1, 16, 16, 16);
    tilexr::checker::TraceRuntime runtime(&world, test_case);
    runtime.SetKernelContext(0, 0, 1);

    uint8_t src[16] = {0};
    uint8_t dst[16] = {0};
    tilexr::checker::TraceRuntime::SetCurrent(&runtime);
    const int copy_line = __LINE__ + 1;
    AscendC::DataCopy(dst, src, sizeof(src));
    tilexr::checker::TraceRuntime::SetCurrent(nullptr);

    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 1, "raw unregistered DataCopy event count");
    if (events.empty()) {
        return;
    }
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kDiagnostic),
                "raw unregistered DataCopy diagnostic kind");
    ExpectContains(events[0].detail, "unresolved trace address",
                   "raw unregistered DataCopy diagnostic detail");
    ExpectEqInt(events[0].source_line, copy_line,
                "raw unregistered DataCopy diagnostic source line");

    tilexr::checker::FindingSet findings =
        tilexr::checker::CheckOrdering(world.events());
    ExpectEqSize(findings.findings().size(), 1,
                 "raw unregistered DataCopy unsupported finding count");
    if (!findings.findings().empty()) {
        ExpectEqInt(static_cast<int>(findings.findings()[0].kind),
                    static_cast<int>(tilexr::checker::FindingKind::kUnsupportedApi),
                    "raw unregistered DataCopy unsupported kind");
    }
}

void TestRawAscendCDataCopyPadRecordsContiguousTraceRuntimeCopy() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 1;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(1, 32, 32, 32);
    tilexr::checker::TraceRuntime runtime(&world, test_case);
    runtime.SetKernelContext(0, 1, 2);

    uint8_t src[32] = {0};
    uint8_t dst[32] = {0};
    runtime.AddRange(src, sizeof(src), tilexr::checker::BufferRole::kUserInput,
                     0, 0);
    runtime.AddRange(dst, sizeof(dst), tilexr::checker::BufferRole::kCommData,
                     0, 0);
    AscendC::DataCopyExtParams params(2, 8, 0, 0, 0);
    tilexr::checker::TraceRuntime::SetCurrent(&runtime);
    const int copy_line = __LINE__ + 1;
    AscendC::DataCopyPad(dst, src, params);
    tilexr::checker::TraceRuntime::SetCurrent(nullptr);

    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 2, "raw DataCopyPad trace event count");
    if (events.size() < 2) {
        return;
    }
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kRead),
                "raw DataCopyPad read event kind");
    ExpectEqInt(static_cast<int>(events[1].kind),
                static_cast<int>(tilexr::checker::EventKind::kCopy),
                "raw DataCopyPad copy event kind");
    ExpectEqInt(events[1].core, 1, "raw DataCopyPad copy core");
    ExpectEqSize(events[1].bytes, 16, "raw DataCopyPad copy bytes");
    ExpectEqInt(events[1].source_line, copy_line,
                "raw DataCopyPad copy source line");
    ExpectContains(events[1].detail, "AscendC::DataCopyPad",
                   "raw DataCopyPad copy detail");
}

void TestRawAscendCDataCopyPadReportsUnsupportedStride() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 1;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(1, 32, 32, 32);
    tilexr::checker::TraceRuntime runtime(&world, test_case);
    runtime.SetKernelContext(0, 0, 1);

    uint8_t src[32] = {0};
    uint8_t dst[32] = {0};
    runtime.AddRange(src, sizeof(src), tilexr::checker::BufferRole::kUserInput,
                     0, 0);
    runtime.AddRange(dst, sizeof(dst), tilexr::checker::BufferRole::kCommData,
                     0, 0);
    AscendC::DataCopyExtParams params(2, 8, 4, 0, 0);
    tilexr::checker::TraceRuntime::SetCurrent(&runtime);
    const int copy_line = __LINE__ + 1;
    AscendC::DataCopyPad(dst, src, params);
    tilexr::checker::TraceRuntime::SetCurrent(nullptr);

    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 1, "raw strided DataCopyPad event count");
    if (events.empty()) {
        return;
    }
    ExpectEqInt(static_cast<int>(events[0].kind),
                static_cast<int>(tilexr::checker::EventKind::kDiagnostic),
                "raw strided DataCopyPad diagnostic kind");
    ExpectContains(events[0].detail, "unsupported trace data copy",
                   "raw strided DataCopyPad diagnostic detail");
    ExpectContains(events[0].detail, "AscendC::DataCopyPad",
                   "raw strided DataCopyPad diagnostic symbol");
    ExpectEqInt(events[0].source_line, copy_line,
                "raw strided DataCopyPad diagnostic source line");

    tilexr::checker::FindingSet findings =
        tilexr::checker::CheckOrdering(world.events());
    ExpectEqSize(findings.findings().size(), 1,
                 "raw strided DataCopyPad unsupported finding count");
    if (!findings.findings().empty()) {
        ExpectEqInt(static_cast<int>(findings.findings()[0].kind),
                    static_cast<int>(tilexr::checker::FindingKind::kUnsupportedApi),
                    "raw strided DataCopyPad unsupported kind");
    }
}

void TestTraceRuntimeReportsCopyBeyondRegisteredStorage() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 1;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(1, 16, 16, 16);
    tilexr::checker::TraceRuntime runtime(&world, test_case);
    runtime.SetKernelContext(0, 0, 1);

    uint8_t src_storage[8] = {0};
    uint8_t dst_storage[8] = {0};
    runtime.AddRange(src_storage, 1024, sizeof(src_storage),
                     tilexr::checker::BufferRole::kUserInput, 0, 0);
    runtime.AddRange(dst_storage, 1024, sizeof(dst_storage),
                     tilexr::checker::BufferRole::kCommData, 0, 0);
    tilexr::checker::CheckerStatus status =
        runtime.RecordCopy(dst_storage, src_storage, 16, TileXR::Op::ADD,
                           TILEXR_CHECKER_HERE, "copy beyond backing storage");

    ExpectTrue(status.ok(), "out-of-storage gm copy record status");
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 3, "out-of-storage gm copy event count");
    if (events.size() < 3) {
        return;
    }
    ExpectEqInt(static_cast<int>(events[2].kind),
                static_cast<int>(tilexr::checker::EventKind::kDiagnostic),
                "out-of-storage gm copy diagnostic event kind");
    ExpectContains(events[2].detail, "exceeds registered storage",
                   "out-of-storage gm copy diagnostic detail");

    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(world.events());
    ExpectEqSize(findings.findings().size(), 1,
                 "out-of-storage gm copy unsupported finding count");
    if (findings.findings().empty()) {
        return;
    }
    ExpectEqInt(static_cast<int>(findings.findings()[0].kind),
                static_cast<int>(tilexr::checker::FindingKind::kUnsupportedApi),
                "out-of-storage gm copy unsupported finding kind");
    ExpectContains(findings.findings()[0].message, "registered checker range",
                   "out-of-storage gm copy unsupported message");
}

void TestTraceRuntimeAllowsSparseVirtualIpcWindow() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 1;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    tilexr::checker::RankWorld world =
        tilexr::checker::RankWorld::Create(1, 16, 16, 16);
    tilexr::checker::TraceRuntime runtime(&world, test_case);
    runtime.SetKernelContext(0, 0, 1);

    uint8_t user_input[16] = {0};
    uint8_t sparse_storage[8] = {0};
    runtime.AddRange(user_input, sizeof(user_input),
                     tilexr::checker::BufferRole::kUserInput, 0, 0);
    const uintptr_t virtual_base = 0x100000000000ULL;
    runtime.AddVirtualRange(virtual_base, sparse_storage, 1024, sizeof(sparse_storage),
                            tilexr::checker::BufferRole::kCommData, 0, 0);

    tilexr::checker::CheckerStatus status =
        runtime.RecordCopy(reinterpret_cast<void *>(virtual_base + 128), user_input, 16,
                           TileXR::Op::ADD, TILEXR_CHECKER_HERE,
                           "copy into sparse virtual ipc window");

    ExpectTrue(status.ok(), "sparse virtual copy record status");
    const std::vector<tilexr::checker::Event> &events = world.events().events();
    ExpectEqSize(events.size(), 2, "sparse virtual copy event count");
    tilexr::checker::FindingSet findings = tilexr::checker::CheckOrdering(world.events());
    ExpectEqSize(findings.findings().size(), 0,
                 "sparse virtual copy unsupported finding count");
}

}  // namespace

int main() {
    TestCopyMovesBytesAndLogsEvent();
    TestFlagEventsCaptureMagic();
    TestNullWorldAndUnsupportedRoleFail();
    TestRecordWriteAndBarrierEvents();
    TestEventsCaptureServerTopology();
    TestCheckerLocalShimIncludePathOnly();
    TestTraceAdapterKeepsAlgorithmShimsThin();
    TestAscendCPipePrimitivesRecordTraceEvents();
    TestTraceRuntimeReportsUnresolvedGmCopyAddress();
    TestRawAscendCDataCopyRecordsTraceRuntimeCopy();
    TestRawAscendCDataCopyReportsUnregisteredAddress();
    TestRawAscendCDataCopyPadRecordsContiguousTraceRuntimeCopy();
    TestRawAscendCDataCopyPadReportsUnsupportedStride();
    TestTraceRuntimeReportsCopyBeyondRegisteredStorage();
    TestTraceRuntimeAllowsSparseVirtualIpcWindow();
    return g_failures == 0 ? 0 : 1;
}
