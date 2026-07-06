#ifndef TILEXR_COLLECTIVES_HOST_PERF_TRACE_REPORT_H
#define TILEXR_COLLECTIVES_HOST_PERF_TRACE_REPORT_H

#include <cstdint>
#include <string>
#include <vector>

#include "tilexr_perf_trace.h"
#include "tilexr_types.h"

namespace TileXRCollectives {
namespace Host {

struct PerfStageSummary {
    uint32_t stageId = 0;
    std::string stageName;
    uint64_t count = 0;
    uint64_t sumCycles = 0;
    uint64_t maxCycles = 0;
    uint32_t maxRank = 0;
    uint32_t maxCore = 0;
    double sumUs = 0.0;
};

struct PerfReportOptions {
    std::string outputDir;
    bool emitAiPrompt = false;
    bool incomplete = false;
    std::string incompleteReason;
};

const char *PerfStageName(uint32_t stageId);

std::vector<PerfStageSummary> SummarizePerfTrace(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats);

std::vector<std::string> AnalyzePerfTrace(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<PerfStageSummary> &summaries);

int WritePerfTraceReports(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats,
    const PerfReportOptions &options);

} // namespace Host
} // namespace TileXRCollectives

#endif // TILEXR_COLLECTIVES_HOST_PERF_TRACE_REPORT_H
