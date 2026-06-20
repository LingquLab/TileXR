#include <cstdint>
#include <iostream>

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

    return g_failures == 0 ? 0 : 1;
}
