#ifndef TILEXR_PERF_TRACE_H
#define TILEXR_PERF_TRACE_H

#include <cstddef>
#include <cstdint>

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define TILEXR_PERF_TRACE_INLINE __attribute__((always_inline)) inline __aicore__
#else
#define TILEXR_PERF_TRACE_INLINE inline
#endif

namespace TileXR {

constexpr uint32_t TILEXR_PERF_TRACE_MAGIC = 0x54585054u; // TXPT
constexpr uint32_t TILEXR_PERF_TRACE_VERSION = 1u;
constexpr uint32_t TILEXR_PERF_MAX_STAGE_NAME = 32u;
constexpr uint32_t TILEXR_PERF_STAGE_COUNT = 7u;
constexpr uint32_t TILEXR_PERF_TRACE_STATS_OFFSET = 128u;

enum class PerfChipClass : uint32_t {
    GENERIC = 0,
    A5 = 1,
};

enum class PerfStageCategory : uint32_t {
    TOTAL = 0,
    COPY = 1,
    WAIT = 2,
    SYNC = 3,
    BARRIER = 4,
};

enum class PerfBarrierPolicy : uint32_t {
    NO_BARRIER = 0,
    END_BARRIER_ONLY = 1,
    BARRIERED = 2,
};

enum class PerfStageId : uint32_t {
    KERNEL_TOTAL = 0,
    CHUNK_TOTAL = 1,
    POST_SYNC = 2,
    LOCAL_INPUT_TO_IPC = 3,
    FLAG_POLL_WAIT = 4,
    PEER_IPC_TO_OUTPUT = 5,
    CHUNK_BARRIER = 6,
};

struct TileXRPerfTraceHeader {
    uint32_t magic = TILEXR_PERF_TRACE_MAGIC;
    uint32_t version = TILEXR_PERF_TRACE_VERSION;
    uint32_t headerSize = sizeof(TileXRPerfTraceHeader);
    // Stage descriptions are static schema metadata; this records their ABI size.
    uint32_t stageDescSize = sizeof(uint32_t) * 4 + TILEXR_PERF_MAX_STAGE_NAME;
    uint32_t coreStageStatsSize = sizeof(uint32_t) * 4 + sizeof(uint64_t) * 10;
    uint32_t flags = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t blockDim = 0;
    uint32_t maxCoreCount = 0;
    uint32_t stageCount = TILEXR_PERF_STAGE_COUNT;
    uint32_t cycleToUsDivisor = 50;
    uint64_t launchId = 0;
    uint64_t messageBytes = 0;
    uint32_t opType = 0;
    uint32_t dataType = 255;
    uint64_t statsOffset = 0;
    uint64_t statsBytes = 0;
};

struct TileXRPerfStageDesc {
    uint32_t stageId = 0;
    PerfStageCategory category = PerfStageCategory::TOTAL;
    PerfBarrierPolicy barrierPolicy = PerfBarrierPolicy::NO_BARRIER;
    uint32_t displayOrder = 0;
    char name[TILEXR_PERF_MAX_STAGE_NAME] = {};
};

struct TileXRPerfCoreStageStats {
    uint32_t rank = 0;
    uint32_t core = 0;
    uint32_t stageId = 0;
    uint32_t reserved = 0;
    uint64_t count = 0;
    uint64_t sumCycles = 0;
    uint64_t minCycles = 0;
    uint64_t maxCycles = 0;
    uint64_t firstStartCycle = 0;
    uint64_t lastEndCycle = 0;
    uint64_t aux0 = 0;
    uint64_t aux1 = 0;
    uint64_t aux2 = 0;
    uint64_t aux3 = 0;
};

TILEXR_PERF_TRACE_INLINE size_t PerfTraceStatsOffset(uint32_t rank, uint32_t core, uint32_t stage,
                                                     uint32_t maxCoreCount, uint32_t stageCount)
{
    return (static_cast<size_t>(rank) * maxCoreCount * stageCount) +
        (static_cast<size_t>(core) * stageCount) + stage;
}

TILEXR_PERF_TRACE_INLINE uint32_t PerfTraceCycleDivisor(PerfChipClass chipClass)
{
    return chipClass == PerfChipClass::A5 ? 1000u : 50u;
}

#if !defined(__CCE__) || !defined(__CCE_IS_AICORE__)
inline double PerfTraceCyclesToUs(uint64_t cycles, uint32_t divisor)
{
    return divisor == 0 ? 0.0 : static_cast<double>(cycles) / static_cast<double>(divisor);
}
#endif

} // namespace TileXR

#undef TILEXR_PERF_TRACE_INLINE

#endif // TILEXR_PERF_TRACE_H
