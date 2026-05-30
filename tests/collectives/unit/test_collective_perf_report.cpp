#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "perf_trace_report.h"
#include "tilexr_perf_trace.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

template <typename T>
void CheckEq(const char *label, T actual, T expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

std::string ReadFile(const std::string &path)
{
    std::ifstream input(path.c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << path << std::endl;
        ++g_failures;
        return {};
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string &path, const std::string &needle)
{
    const std::string text = ReadFile(path);
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected " << path << " to contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckTextContains(const std::string &label, const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected " << label << " to contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckFileMissingOrDoesNotContain(const std::string &path, const std::string &needle)
{
    std::ifstream input(path.c_str());
    if (!input.is_open()) {
        return;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string text = buffer.str();
    if (text.find(needle) != std::string::npos) {
        std::cerr << "expected " << path << " not to retain stale content: " << needle << std::endl;
        ++g_failures;
    }
}

TileXR::TileXRPerfCoreStageStats MakeStat(uint32_t rank, uint32_t core, TileXR::PerfStageId stage,
                                          uint64_t count, uint64_t sumCycles, uint64_t maxCycles)
{
    TileXR::TileXRPerfCoreStageStats stat {};
    stat.rank = rank;
    stat.core = core;
    stat.stageId = static_cast<uint32_t>(stage);
    stat.count = count;
    stat.sumCycles = sumCycles;
    stat.minCycles = sumCycles / count;
    stat.maxCycles = maxCycles;
    stat.firstStartCycle = 100000 + rank * 1000 + core * 100 + static_cast<uint32_t>(stage);
    stat.lastEndCycle = stat.firstStartCycle + sumCycles;
    stat.aux0 = 0x1000 + rank;
    stat.aux1 = 0x2000 + core;
    return stat;
}

void TestPerfReportSummariesAndFiles()
{
    const std::string outputDir = "/tmp/tilexr_perf_report_test";
    std::remove((outputDir + "/trace.json").c_str());
    std::remove((outputDir + "/summary.csv").c_str());
    std::remove((outputDir + "/analysis.md").c_str());
    std::remove((outputDir + "/report.html").c_str());
    std::remove((outputDir + "/ai_prompt.md").c_str());

    TileXR::TileXRPerfTraceHeader header {};
    header.rankSize = 2;
    header.maxCoreCount = 2;
    header.blockDim = 2;
    header.stageCount = TileXR::TILEXR_PERF_STAGE_COUNT;
    header.cycleToUsDivisor = 50;
    header.opType = static_cast<uint32_t>(TileXR::TileXRType::ALL_GATHER);
    header.messageBytes = 64ull * 1024ull * 1024ull;

    std::vector<TileXR::TileXRPerfCoreStageStats> stats(
        static_cast<size_t>(header.rankSize) * header.maxCoreCount * header.stageCount);

    const size_t waitOffset = TileXR::PerfTraceStatsOffset(
        0, 0, static_cast<uint32_t>(TileXR::PerfStageId::FLAG_POLL_WAIT),
        header.maxCoreCount, header.stageCount);
    stats[waitOffset] = MakeStat(0, 0, TileXR::PerfStageId::FLAG_POLL_WAIT, 1, 1000, 1000);

    const size_t peerOffset = TileXR::PerfTraceStatsOffset(
        1, 1, static_cast<uint32_t>(TileXR::PerfStageId::PEER_IPC_TO_OUTPUT),
        header.maxCoreCount, header.stageCount);
    stats[peerOffset] = MakeStat(1, 1, TileXR::PerfStageId::PEER_IPC_TO_OUTPUT, 2, 4000, 2500);

    const auto summaries = TileXRCollectives::Host::SummarizePerfTrace(header, stats);
    CHECK_TRUE(!summaries.empty());
    if (!summaries.empty()) {
        CheckEq("top stage", summaries.front().stageName, std::string("peer_ipc_to_output"));
    }

    const auto findings = TileXRCollectives::Host::AnalyzePerfTrace(header, summaries);
    CHECK_TRUE(!findings.empty());
    if (!findings.empty()) {
        CHECK_TRUE(findings.front().find("bottleneck") != std::string::npos);
    }

    TileXRCollectives::Host::PerfReportOptions options {};
    options.outputDir = outputDir;
    options.emitAiPrompt = true;
    CheckEq("WritePerfTraceReports",
            TileXRCollectives::Host::WritePerfTraceReports(header, stats, options),
            TileXR::TILEXR_SUCCESS);

    CheckContains(options.outputDir + "/trace.json", "\"raw_cycles\"");
    CheckContains(options.outputDir + "/trace.json", "\"first_start_cycle\"");
    CheckContains(options.outputDir + "/trace.json", "\"last_end_cycle\"");
    CheckContains(options.outputDir + "/trace.json", "\"sum_us\"");
    const std::string traceJson = ReadFile(options.outputDir + "/trace.json");
    CheckTextContains("trace.json", traceJson,
                      "\"stage\": \"flag_poll_wait\", \"stage_id\": 4, \"count\": 1, \"raw_cycles\": 1000");
    CheckTextContains("trace.json", traceJson,
                      "\"min_cycles\": 1000, \"max_cycles\": 1000, \"first_start_cycle\": 100004");
    CheckTextContains("trace.json", traceJson,
                      "\"last_end_cycle\": 101004, \"aux0\": 4096, \"aux1\": 8192, \"sum_us\": 20");
    CheckTextContains("trace.json", traceJson,
                      "\"stage\": \"peer_ipc_to_output\", \"stage_id\": 5, \"count\": 2, \"raw_cycles\": 4000");
    CheckTextContains("trace.json", traceJson,
                      "\"min_cycles\": 2000, \"max_cycles\": 2500, \"first_start_cycle\": 101105");
    CheckTextContains("trace.json", traceJson,
                      "\"last_end_cycle\": 105105, \"aux0\": 4097, \"aux1\": 8193, \"sum_us\": 80");
    CheckContains(options.outputDir + "/summary.csv", "stage,rank,core");
    CheckContains(options.outputDir + "/analysis.md", "bottleneck");
    CheckContains(options.outputDir + "/report.html", "Bottleneck First");
    CheckContains(options.outputDir + "/report.html", "Timeline");
    CheckContains(options.outputDir + "/report.html", "first_start_cycle");
    const std::string htmlReport = ReadFile(options.outputDir + "/report.html");
    CheckTextContains("report.html", htmlReport,
                      "peer_ipc_to_output</td><td>101105</td><td>105105</td><td>4000</td><td>4000</td><td>80");
    CheckContains(options.outputDir + "/ai_prompt.md", "TileXR collective profiling");

    options.outputDir = "/tmp/tilexr_perf_report_test_nested/a/b";
    options.emitAiPrompt = true;
    CheckEq("WritePerfTraceReports nested output",
            TileXRCollectives::Host::WritePerfTraceReports(header, stats, options),
            TileXR::TILEXR_SUCCESS);
    CheckContains(options.outputDir + "/trace.json", "\"raw_cycles\"");

    std::ofstream stalePrompt((options.outputDir + "/ai_prompt.md").c_str(), std::ios::out | std::ios::trunc);
    stalePrompt << "stale prompt";
    stalePrompt.close();
    options.emitAiPrompt = false;
    CheckEq("WritePerfTraceReports without ai prompt",
            TileXRCollectives::Host::WritePerfTraceReports(header, stats, options),
            TileXR::TILEXR_SUCCESS);
    CheckFileMissingOrDoesNotContain(options.outputDir + "/ai_prompt.md", "stale prompt");

    TileXR::TileXRPerfTraceHeader invalidHeader = header;
    invalidHeader.stageCount = TileXR::TILEXR_PERF_STAGE_COUNT + 1;
    CheckEq("WritePerfTraceReports rejects invalid stage count",
            TileXRCollectives::Host::WritePerfTraceReports(invalidHeader, stats, options),
            TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    invalidHeader = header;
    invalidHeader.cycleToUsDivisor = 0;
    CheckEq("WritePerfTraceReports rejects invalid cycle divisor",
            TileXRCollectives::Host::WritePerfTraceReports(invalidHeader, stats, options),
            TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestPerfReportSummariesAndFiles();
    return g_failures == 0 ? 0 : 1;
}
