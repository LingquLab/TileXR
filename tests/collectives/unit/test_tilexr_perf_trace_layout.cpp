#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "tilexr_perf_trace.h"

namespace {

int g_failures = 0;

template <typename T>
void CheckEq(const char *label, T actual, T expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckTrue(const char *label, bool value)
{
    if (!value) {
        std::cerr << label << " expected true" << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    CheckEq("trace magic", TileXR::TILEXR_PERF_TRACE_MAGIC, 0x54585054u);
    CheckEq("trace version", TileXR::TILEXR_PERF_TRACE_VERSION, 1u);
    CheckEq("max stage name", TileXR::TILEXR_PERF_MAX_STAGE_NAME, 32u);
    CheckEq("stage count", TileXR::TILEXR_PERF_STAGE_COUNT, 7u);
    CheckEq("a5 divisor", TileXR::PerfTraceCycleDivisor(TileXR::PerfChipClass::A5), 1000u);
    CheckEq("generic divisor", TileXR::PerfTraceCycleDivisor(TileXR::PerfChipClass::GENERIC), 50u);
    CheckEq("cycles to us", TileXR::PerfTraceCyclesToUs(2500, 50), 50.0);
    CheckEq("stats offset",
            TileXR::PerfTraceStatsOffset(1, 3, 2, 8, TileXR::TILEXR_PERF_STAGE_COUNT),
            static_cast<size_t>(1 * 8 * TileXR::TILEXR_PERF_STAGE_COUNT +
                                3 * TileXR::TILEXR_PERF_STAGE_COUNT + 2));

    CheckTrue("header standard layout", std::is_standard_layout<TileXR::TileXRPerfTraceHeader>::value);
    CheckTrue("stage desc standard layout", std::is_standard_layout<TileXR::TileXRPerfStageDesc>::value);
    CheckTrue("core stage stats standard layout", std::is_standard_layout<TileXR::TileXRPerfCoreStageStats>::value);
    CheckTrue("header trivially copyable", std::is_trivially_copyable<TileXR::TileXRPerfTraceHeader>::value);
    CheckTrue("stage desc trivially copyable", std::is_trivially_copyable<TileXR::TileXRPerfStageDesc>::value);
    CheckTrue("core stage stats trivially copyable",
              std::is_trivially_copyable<TileXR::TileXRPerfCoreStageStats>::value);
    CheckEq("header sizeof", sizeof(TileXR::TileXRPerfTraceHeader), static_cast<size_t>(88));
    CheckEq("stage desc sizeof", sizeof(TileXR::TileXRPerfStageDesc), static_cast<size_t>(48));
    CheckEq("core stage stats sizeof", sizeof(TileXR::TileXRPerfCoreStageStats), static_cast<size_t>(80));

    CheckEq("header magic offset", offsetof(TileXR::TileXRPerfTraceHeader, magic), static_cast<size_t>(0));
    CheckEq("header version offset", offsetof(TileXR::TileXRPerfTraceHeader, version), static_cast<size_t>(4));
    CheckEq("header headerSize offset", offsetof(TileXR::TileXRPerfTraceHeader, headerSize), static_cast<size_t>(8));
    CheckEq("header stageDescSize offset", offsetof(TileXR::TileXRPerfTraceHeader, stageDescSize),
            static_cast<size_t>(12));
    CheckEq("header coreStageStatsSize offset", offsetof(TileXR::TileXRPerfTraceHeader, coreStageStatsSize),
            static_cast<size_t>(16));
    CheckEq("header flags offset", offsetof(TileXR::TileXRPerfTraceHeader, flags), static_cast<size_t>(20));
    CheckEq("header rank offset", offsetof(TileXR::TileXRPerfTraceHeader, rank), static_cast<size_t>(24));
    CheckEq("header rankSize offset", offsetof(TileXR::TileXRPerfTraceHeader, rankSize), static_cast<size_t>(28));
    CheckEq("header blockDim offset", offsetof(TileXR::TileXRPerfTraceHeader, blockDim), static_cast<size_t>(32));
    CheckEq("header maxCoreCount offset", offsetof(TileXR::TileXRPerfTraceHeader, maxCoreCount),
            static_cast<size_t>(36));
    CheckEq("header stageCount offset", offsetof(TileXR::TileXRPerfTraceHeader, stageCount),
            static_cast<size_t>(40));
    CheckEq("header cycleToUsDivisor offset", offsetof(TileXR::TileXRPerfTraceHeader, cycleToUsDivisor),
            static_cast<size_t>(44));
    CheckEq("header launchId offset", offsetof(TileXR::TileXRPerfTraceHeader, launchId), static_cast<size_t>(48));
    CheckEq("header messageBytes offset", offsetof(TileXR::TileXRPerfTraceHeader, messageBytes),
            static_cast<size_t>(56));
    CheckEq("header opType offset", offsetof(TileXR::TileXRPerfTraceHeader, opType), static_cast<size_t>(64));
    CheckEq("header dataType offset", offsetof(TileXR::TileXRPerfTraceHeader, dataType), static_cast<size_t>(68));
    CheckEq("header statsOffset offset", offsetof(TileXR::TileXRPerfTraceHeader, statsOffset),
            static_cast<size_t>(72));
    CheckEq("header statsBytes offset", offsetof(TileXR::TileXRPerfTraceHeader, statsBytes), static_cast<size_t>(80));

    CheckEq("stage desc stageId offset", offsetof(TileXR::TileXRPerfStageDesc, stageId), static_cast<size_t>(0));
    CheckEq("stage desc category offset", offsetof(TileXR::TileXRPerfStageDesc, category), static_cast<size_t>(4));
    CheckEq("stage desc barrierPolicy offset", offsetof(TileXR::TileXRPerfStageDesc, barrierPolicy),
            static_cast<size_t>(8));
    CheckEq("stage desc displayOrder offset", offsetof(TileXR::TileXRPerfStageDesc, displayOrder),
            static_cast<size_t>(12));
    CheckEq("stage desc name offset", offsetof(TileXR::TileXRPerfStageDesc, name), static_cast<size_t>(16));

    CheckEq("core stage stats rank offset", offsetof(TileXR::TileXRPerfCoreStageStats, rank), static_cast<size_t>(0));
    CheckEq("core stage stats core offset", offsetof(TileXR::TileXRPerfCoreStageStats, core), static_cast<size_t>(4));
    CheckEq("core stage stats stageId offset", offsetof(TileXR::TileXRPerfCoreStageStats, stageId),
            static_cast<size_t>(8));
    CheckEq("core stage stats reserved offset", offsetof(TileXR::TileXRPerfCoreStageStats, reserved),
            static_cast<size_t>(12));
    CheckEq("core stage stats count offset", offsetof(TileXR::TileXRPerfCoreStageStats, count),
            static_cast<size_t>(16));
    CheckEq("core stage stats sumCycles offset", offsetof(TileXR::TileXRPerfCoreStageStats, sumCycles),
            static_cast<size_t>(24));
    CheckEq("core stage stats minCycles offset", offsetof(TileXR::TileXRPerfCoreStageStats, minCycles),
            static_cast<size_t>(32));
    CheckEq("core stage stats maxCycles offset", offsetof(TileXR::TileXRPerfCoreStageStats, maxCycles),
            static_cast<size_t>(40));
    CheckEq("core stage stats firstStartCycle offset", offsetof(TileXR::TileXRPerfCoreStageStats, firstStartCycle),
            static_cast<size_t>(48));
    CheckEq("core stage stats lastEndCycle offset", offsetof(TileXR::TileXRPerfCoreStageStats, lastEndCycle),
            static_cast<size_t>(56));
    CheckEq("core stage stats aux0 offset", offsetof(TileXR::TileXRPerfCoreStageStats, aux0),
            static_cast<size_t>(64));
    CheckEq("core stage stats aux1 offset", offsetof(TileXR::TileXRPerfCoreStageStats, aux1),
            static_cast<size_t>(72));

    TileXR::TileXRPerfTraceHeader header {};
    CheckEq("default header magic", header.magic, TileXR::TILEXR_PERF_TRACE_MAGIC);
    CheckEq("default header version", header.version, TileXR::TILEXR_PERF_TRACE_VERSION);
    CheckEq("default header size", header.headerSize,
            static_cast<uint32_t>(sizeof(TileXR::TileXRPerfTraceHeader)));
    CheckEq("default stage desc size", header.stageDescSize,
            static_cast<uint32_t>(sizeof(TileXR::TileXRPerfStageDesc)));
    CheckEq("default core stage stats size", header.coreStageStatsSize,
            static_cast<uint32_t>(sizeof(TileXR::TileXRPerfCoreStageStats)));
    CheckTrue("stats carry raw cycles", offsetof(TileXR::TileXRPerfCoreStageStats, sumCycles) > 0);
    CheckTrue("stats carry timeline bounds", offsetof(TileXR::TileXRPerfCoreStageStats, lastEndCycle) >
        offsetof(TileXR::TileXRPerfCoreStageStats, firstStartCycle));
    return g_failures == 0 ? 0 : 1;
}
