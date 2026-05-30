#include <cstddef>
#include <cstdint>
#include <iostream>

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
    CheckEq("stage count", TileXR::TILEXR_PERF_STAGE_COUNT, 7u);
    CheckEq("a5 divisor", TileXR::PerfTraceCycleDivisor(TileXR::PerfChipClass::A5), 1000u);
    CheckEq("generic divisor", TileXR::PerfTraceCycleDivisor(TileXR::PerfChipClass::GENERIC), 50u);
    CheckEq("cycles to us", TileXR::PerfTraceCyclesToUs(2500, 50), 50.0);
    CheckEq("stats offset",
            TileXR::PerfTraceStatsOffset(1, 3, 2, 8, TileXR::TILEXR_PERF_STAGE_COUNT),
            static_cast<size_t>(1 * 8 * TileXR::TILEXR_PERF_STAGE_COUNT +
                                3 * TileXR::TILEXR_PERF_STAGE_COUNT + 2));

    TileXR::TileXRPerfTraceHeader header {};
    header.magic = TileXR::TILEXR_PERF_TRACE_MAGIC;
    header.version = TileXR::TILEXR_PERF_TRACE_VERSION;
    header.headerSize = sizeof(TileXR::TileXRPerfTraceHeader);
    header.stageDescSize = sizeof(TileXR::TileXRPerfStageDesc);
    header.coreStageStatsSize = sizeof(TileXR::TileXRPerfCoreStageStats);
    CheckTrue("header size is recorded", header.headerSize >= sizeof(uint32_t) * 8);
    CheckTrue("stats carry raw cycles", offsetof(TileXR::TileXRPerfCoreStageStats, sumCycles) > 0);
    CheckTrue("stats carry timeline bounds", offsetof(TileXR::TileXRPerfCoreStageStats, lastEndCycle) >
        offsetof(TileXR::TileXRPerfCoreStageStats, firstStartCycle));
    return g_failures == 0 ? 0 : 1;
}
