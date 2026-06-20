#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kernel_operator.h"
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

void TestCheckerLocalShimIncludePathOnly() {
    const std::string checker_cmake = ReadFile(SourcePath("tests/checker/CMakeLists.txt"));
    const std::string tools_cmake = ReadFile(SourcePath("tools/checker/CMakeLists.txt"));

    ExpectContains(checker_cmake, "test_tilexr_checker_shim_events",
                   "checker test target wired");
    ExpectContains(checker_cmake, "tools/checker/shim", "checker shim include path wired");
    ExpectNotContains(tools_cmake, "tools/checker/shim", "checker core should not export shim");
}

}  // namespace

int main() {
    TestCopyMovesBytesAndLogsEvent();
    TestFlagEventsCaptureMagic();
    TestNullWorldAndUnsupportedRoleFail();
    TestRecordWriteAndBarrierEvents();
    TestCheckerLocalShimIncludePathOnly();
    return g_failures == 0 ? 0 : 1;
}
