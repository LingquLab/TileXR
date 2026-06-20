#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

#include "tilexr/checker/event.h"
#include "tilexr/checker/memory.h"
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

void ExpectEqU64(uint64_t actual, uint64_t expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

void ExpectEqI32(int32_t actual, int32_t expected, const char *message) {
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

void ExpectNull(const void *ptr, const char *message) {
    if (ptr != nullptr) {
        std::cerr << message << ": expected null\n";
        ++g_failures;
    }
}

void ExpectPtrEq(const void *actual, const void *expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}
}  // namespace

int main() {
    tilexr::checker::RankWorld world2 =
        tilexr::checker::RankWorld::Create(2, 16, 16, 32);
    tilexr::checker::RankWorld world4 =
        tilexr::checker::RankWorld::Create(4, 8, 8, 64);

    ExpectEqInt(world2.rank_size(), 2, "rank world with 2 ranks");
    ExpectEqInt(world4.rank_size(), 4, "rank world with 4 ranks");

    ExpectEqInt(world2.HostArgs(0).rank, 0, "rank 0 host args rank");
    ExpectEqInt(world2.HostArgs(1).rank, 1, "rank 1 host args rank");

    ExpectTrue(world2.UserInput(0).WriteInt32(0, 123).ok(),
               "write rank 0 user input int32");
    int32_t input_rank0 = 0;
    int32_t input_rank1 = 0;
    ExpectTrue(world2.UserInput(0).ReadInt32(0, &input_rank0).ok(),
               "read back rank 0 user input int32");
    ExpectTrue(world2.UserInput(1).ReadInt32(0, &input_rank1).ok(),
               "read rank 1 untouched user input int32");
    ExpectEqI32(input_rank0, 123, "rank 0 user input stores data");
    ExpectEqI32(input_rank1, 0, "rank 1 user input remains isolated");

    const uint64_t magic1 = world2.NextMagic();
    const uint64_t magic2 = world2.NextMagic();
    ExpectEqU64(magic1, 1, "first magic value");
    ExpectEqU64(magic2, 2, "second magic value");

    tilexr::checker::ByteBuffer buffer(4);
    ExpectTrue(!buffer.WriteBytes(2, "abcd", 3).ok(), "write bytes bounds failure");
    int32_t ignored = 0;
    ExpectTrue(!buffer.ReadInt32(1, &ignored).ok(), "read int32 bounds failure");
    const size_t int32_count_max =
        std::numeric_limits<size_t>::max() / sizeof(int32_t);
    ExpectTrue(!buffer.WriteInt32(int32_count_max + 1, 7).ok(),
               "write int32 overflow failure");
    ExpectTrue(!buffer.ReadInt32(int32_count_max + 1, &ignored).ok(),
               "read int32 overflow failure");

    tilexr::checker::Event first;
    first.kind = tilexr::checker::EventKind::kDiagnostic;
    first.rank = 0;
    world2.events().Add(first);

    tilexr::checker::Event second;
    second.kind = tilexr::checker::EventKind::kBarrier;
    second.rank = 1;
    world2.events().Add(second);

    const std::vector<tilexr::checker::Event> &events = world2.events().events();
    ExpectEqInt(static_cast<int>(events.size()), 2, "event count");
    ExpectEqU64(events[0].id, 1, "first event id");
    ExpectEqU64(events[1].id, 2, "second event id");
    world2.events().Clear();
    world2.events().Add(first);
    ExpectEqInt(static_cast<int>(world2.events().events().size()), 1,
                "event count after clear");
    ExpectEqU64(world2.events().events()[0].id, 1, "event id resets after clear");

    ExpectEqInt(world4.HostArgs(0).localRank, 0, "rank 0 localRank");
    ExpectEqInt(world4.HostArgs(3).localRank, 3, "rank 3 localRank");
    ExpectEqInt(world4.HostArgs(0).rankSize, 4, "rank 0 rankSize");
    ExpectEqInt(world4.HostArgs(3).localRankSize, 4, "rank 3 localRankSize");
    ExpectEqU64(world4.HostArgs(2).extraFlag, 0, "extraFlag defaults to zero");
    ExpectNull(world4.HostArgs(1).udmaInfoPtr, "udmaInfoPtr defaults null");
    ExpectNull(world4.HostArgs(1).udmaRegistryPtr, "udmaRegistryPtr defaults null");
    ExpectNull(world4.HostArgs(1).sdmaWorkspacePtr, "sdmaWorkspacePtr defaults null");

    ExpectEqI32(world4.HostArgs(0).magics[0], 0, "magics[0] zeroed");
    ExpectEqI32(world4.HostArgs(0).magics[3], 0, "magics[3] zeroed");
    ExpectTrue(world4.HostArgs(2).sendCountMatrix[0] == 0,
               "sendCountMatrix[0] zeroed");
    ExpectTrue(world4.HostArgs(2).sendCountMatrix[4 * 3 + 1] == 0,
               "sendCountMatrix[row][col] zeroed");

    ExpectEqSize(world4.CommFlag(0, 0).size(), sizeof(uint64_t),
                 "comm flag size is uint64_t");
    ExpectEqSize(world4.CommFlag(3, 1).size(), sizeof(uint64_t),
                 "comm flag size is uint64_t for any slot");
    ExpectTrue(world4.CommFlag(0, 0).WriteInt32(0, 55).ok(), "write first comm flag");
    int32_t flag00 = 0;
    int32_t flag01 = 0;
    ExpectTrue(world4.CommFlag(0, 0).ReadInt32(0, &flag00).ok(), "read first comm flag");
    ExpectTrue(world4.CommFlag(0, 1).ReadInt32(0, &flag01).ok(),
               "read isolated comm flag");
    ExpectEqI32(flag00, 55, "comm flag stores value");
    ExpectEqI32(flag01, 0, "comm flag slots are isolated");

    ExpectPtrEq(world4.HostArgs(0).peerMems[0], world4.CommData(0, 0).data().data(),
                "peerMems self mapping");
    ExpectPtrEq(world4.HostArgs(0).peerMems[3], world4.CommData(3, 0).data().data(),
                "peerMems peer mapping");
    ExpectPtrEq(world4.HostArgs(2).peerMems[1], world4.CommData(1, 2).data().data(),
                "peerMems maps peer slot to backing storage");

    return g_failures == 0 ? 0 : 1;
}
