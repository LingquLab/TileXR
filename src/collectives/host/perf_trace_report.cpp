#include "perf_trace_report.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace TileXRCollectives {
namespace Host {
namespace {

bool EnsureOutputDir(const std::string &outputDir)
{
    if (outputDir.empty()) {
        return false;
    }

    std::string current;
    size_t pos = 0;
    if (outputDir[0] == '/') {
        current = "/";
        pos = 1;
    }

    while (pos <= outputDir.size()) {
        const size_t next = outputDir.find('/', pos);
        const std::string component = outputDir.substr(
            pos, next == std::string::npos ? std::string::npos : next - pos);
        pos = next == std::string::npos ? outputDir.size() + 1 : next + 1;
        if (component.empty()) {
            continue;
        }

        if (!current.empty() && current[current.size() - 1] != '/') {
            current += "/";
        }
        current += component;

        struct stat st {};
        if (stat(current.c_str(), &st) == 0) {
            if (!S_ISDIR(st.st_mode)) {
                return false;
            }
            continue;
        }

        if (errno != ENOENT) {
            return false;
        }

        if (mkdir(current.c_str(), 0755) != 0) {
            if (errno != EEXIST) {
                return false;
            }
            if (stat(current.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
                return false;
            }
        }
    }

    return true;
}

bool WriteTextFile(const std::string &path, const std::string &text)
{
    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }

    output << text;
    output.flush();
    if (output.fail() || output.bad()) {
        return false;
    }
    output.close();
    return !output.fail() && !output.bad();
}

bool RemoveIfExists(const std::string &path)
{
    if (std::remove(path.c_str()) == 0) {
        return true;
    }
    return errno == ENOENT;
}

std::string JoinPath(const std::string &dir, const std::string &file)
{
    if (!dir.empty() && dir[dir.size() - 1] == '/') {
        return dir + file;
    }
    return dir + "/" + file;
}

std::string OpName(uint32_t opType)
{
    const auto it = TileXR::TILEXR_TYPE2NAME.find(static_cast<TileXR::TileXRType>(opType));
    if (it != TileXR::TILEXR_TYPE2NAME.end()) {
        return it->second;
    }
    return "Unknown";
}

std::string EscapeJson(const std::string &value)
{
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

std::string EscapeHtml(const std::string &value)
{
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '&':
                out << "&amp;";
                break;
            case '<':
                out << "&lt;";
                break;
            case '>':
                out << "&gt;";
                break;
            case '"':
                out << "&quot;";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

std::vector<TileXR::TileXRPerfCoreStageStats> NonEmptyStats(
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats)
{
    std::vector<TileXR::TileXRPerfCoreStageStats> result;
    for (const auto &stat : stats) {
        if (stat.count != 0 || stat.sumCycles != 0 || stat.maxCycles != 0) {
            result.push_back(stat);
        }
    }
    return result;
}

double StatSumUs(const TileXR::TileXRPerfTraceHeader &header, const TileXR::TileXRPerfCoreStageStats &stat)
{
    return TileXR::PerfTraceCyclesToUs(stat.sumCycles, header.cycleToUsDivisor);
}

bool IsValidTraceHeader(const TileXR::TileXRPerfTraceHeader &header)
{
    return header.magic == TileXR::TILEXR_PERF_TRACE_MAGIC &&
        header.version == TileXR::TILEXR_PERF_TRACE_VERSION &&
        header.stageCount > 0 &&
        header.stageCount <= TileXR::TILEXR_PERF_STAGE_COUNT &&
        header.cycleToUsDivisor != 0;
}

std::string BuildTraceJson(const TileXR::TileXRPerfTraceHeader &header,
                           const std::vector<TileXR::TileXRPerfCoreStageStats> &stats)
{
    const auto nonEmptyStats = NonEmptyStats(stats);
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"tilexr_perf_trace_report.v1\",\n";
    out << "  \"op_type\": " << header.opType << ",\n";
    out << "  \"op_name\": \"" << EscapeJson(OpName(header.opType)) << "\",\n";
    out << "  \"rank_size\": " << header.rankSize << ",\n";
    out << "  \"max_core_count\": " << header.maxCoreCount << ",\n";
    out << "  \"block_dim\": " << header.blockDim << ",\n";
    out << "  \"stage_count\": " << header.stageCount << ",\n";
    out << "  \"cycle_to_us_divisor\": " << header.cycleToUsDivisor << ",\n";
    out << "  \"message_bytes\": " << header.messageBytes << ",\n";
    out << "  \"stats\": [\n";
    for (size_t i = 0; i < nonEmptyStats.size(); ++i) {
        const auto &stat = nonEmptyStats[i];
        out << "    {\"rank\": " << stat.rank
            << ", \"core\": " << stat.core
            << ", \"stage\": \"" << EscapeJson(PerfStageName(stat.stageId)) << "\""
            << ", \"stage_id\": " << stat.stageId
            << ", \"count\": " << stat.count
            << ", \"raw_cycles\": " << stat.sumCycles
            << ", \"min_cycles\": " << stat.minCycles
            << ", \"max_cycles\": " << stat.maxCycles
            << ", \"first_start_cycle\": " << stat.firstStartCycle
            << ", \"last_end_cycle\": " << stat.lastEndCycle
            << ", \"aux0\": " << stat.aux0
            << ", \"aux1\": " << stat.aux1
            << ", \"sum_us\": " << StatSumUs(header, stat)
            << "}";
        if (i + 1 != nonEmptyStats.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

std::string BuildSummaryCsv(const std::vector<PerfStageSummary> &summaries)
{
    std::ostringstream out;
    out << "stage,rank,core,count,sum_cycles,max_cycles,sum_us,rank_meaning,core_meaning\n";
    for (const auto &summary : summaries) {
        out << summary.stageName << ","
            << summary.maxRank << ","
            << summary.maxCore << ","
            << summary.count << ","
            << summary.sumCycles << ","
            << summary.maxCycles << ","
            << summary.sumUs << ","
            << "max,max\n";
    }
    return out.str();
}

std::string BuildAnalysisMarkdown(const TileXR::TileXRPerfTraceHeader &header,
                                  const std::vector<PerfStageSummary> &summaries,
                                  const std::vector<std::string> &findings)
{
    std::ostringstream out;
    out << "# TileXR Collective Perf Analysis\n\n";
    out << "- Operation: " << OpName(header.opType) << "\n";
    out << "- Message bytes: " << header.messageBytes << "\n";
    out << "- Rank size: " << header.rankSize << "\n";
    out << "- Block dim: " << header.blockDim << "\n\n";
    out << "## Findings\n\n";
    for (const auto &finding : findings) {
        out << "- " << finding << "\n";
    }
    out << "\n## Stage Summary\n\n";
    out << "| Stage | Count | Sum cycles | Sum us | Max rank | Max core |\n";
    out << "| --- | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto &summary : summaries) {
        out << "| " << summary.stageName
            << " | " << summary.count
            << " | " << summary.sumCycles
            << " | " << summary.sumUs
            << " | " << summary.maxRank
            << " | " << summary.maxCore
            << " |\n";
    }
    return out.str();
}

std::string BuildHtmlReport(const TileXR::TileXRPerfTraceHeader &header,
                            const std::vector<PerfStageSummary> &summaries,
                            const std::vector<std::string> &findings,
                            const std::vector<TileXR::TileXRPerfCoreStageStats> &stats)
{
    const auto nonEmptyStats = NonEmptyStats(stats);
    std::ostringstream out;
    out << "<!doctype html>\n<html><head><meta charset=\"utf-8\">";
    out << "<title>TileXR Collective Perf Report</title>";
    out << "<style>body{font-family:sans-serif;margin:24px;line-height:1.45}";
    out << "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 8px}";
    out << "th{background:#f4f4f4;text-align:left}</style></head><body>\n";
    out << "<h1>TileXR Collective Perf Report</h1>\n";
    out << "<h2>Bottleneck First</h2>\n";
    out << "<p>Operation: " << EscapeHtml(OpName(header.opType)) << ", message bytes: "
        << header.messageBytes << "</p>\n";
    out << "<ul>\n";
    for (const auto &finding : findings) {
        out << "<li>" << EscapeHtml(finding) << "</li>\n";
    }
    out << "</ul>\n";
    out << "<table><thead><tr><th>Stage</th><th>Count</th><th>Sum cycles</th>";
    out << "<th>Sum us</th><th>Max rank</th><th>Max core</th></tr></thead><tbody>\n";
    for (const auto &summary : summaries) {
        out << "<tr><td>" << EscapeHtml(summary.stageName)
            << "</td><td>" << summary.count
            << "</td><td>" << summary.sumCycles
            << "</td><td>" << summary.sumUs
            << "</td><td>" << summary.maxRank
            << "</td><td>" << summary.maxCore
            << "</td></tr>\n";
    }
    out << "</tbody></table>\n";
    out << "<h2>Timeline</h2>\n";
    out << "<table><thead><tr><th>Rank</th><th>Core</th><th>Stage</th>";
    out << "<th>first_start_cycle</th><th>last_end_cycle</th>";
    out << "<th>duration_cycles</th><th>raw_cycles</th><th>sum_us</th>";
    out << "</tr></thead><tbody>\n";
    for (const auto &stat : nonEmptyStats) {
        const uint64_t durationCycles = stat.lastEndCycle >= stat.firstStartCycle ?
            stat.lastEndCycle - stat.firstStartCycle : 0;
        out << "<tr><td>" << stat.rank
            << "</td><td>" << stat.core
            << "</td><td>" << EscapeHtml(PerfStageName(stat.stageId))
            << "</td><td>" << stat.firstStartCycle
            << "</td><td>" << stat.lastEndCycle
            << "</td><td>" << durationCycles
            << "</td><td>" << stat.sumCycles
            << "</td><td>" << StatSumUs(header, stat)
            << "</td></tr>\n";
    }
    out << "</tbody></table>\n";
    out << "</body></html>\n";
    return out.str();
}

std::string BuildAiPrompt(const TileXR::TileXRPerfTraceHeader &header,
                          const std::vector<PerfStageSummary> &summaries,
                          const std::vector<std::string> &findings)
{
    std::ostringstream out;
    out << "# TileXR collective profiling\n\n";
    out << "Analyze this TileXR collective profiling summary and suggest host/kernel tuning next steps.\n\n";
    out << "Operation: " << OpName(header.opType) << "\n";
    out << "Message bytes: " << header.messageBytes << "\n";
    out << "Cycle-to-us divisor: " << header.cycleToUsDivisor << "\n\n";
    out << "Findings:\n";
    for (const auto &finding : findings) {
        out << "- " << finding << "\n";
    }
    out << "\nStage summaries:\n";
    for (const auto &summary : summaries) {
        out << "- " << summary.stageName << ": " << summary.sumCycles
            << " cycles, " << summary.sumUs << " us, count " << summary.count << "\n";
    }
    return out.str();
}

} // namespace

const char *PerfStageName(uint32_t stageId)
{
    switch (static_cast<TileXR::PerfStageId>(stageId)) {
        case TileXR::PerfStageId::KERNEL_TOTAL:
            return "kernel_total";
        case TileXR::PerfStageId::CHUNK_TOTAL:
            return "chunk_total";
        case TileXR::PerfStageId::POST_SYNC:
            return "post_sync";
        case TileXR::PerfStageId::LOCAL_INPUT_TO_IPC:
            return "local_input_to_ipc";
        case TileXR::PerfStageId::FLAG_POLL_WAIT:
            return "flag_poll_wait";
        case TileXR::PerfStageId::PEER_IPC_TO_OUTPUT:
            return "peer_ipc_to_output";
        case TileXR::PerfStageId::CHUNK_BARRIER:
            return "chunk_barrier";
        default:
            return "unknown";
    }
}

std::vector<PerfStageSummary> SummarizePerfTrace(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats)
{
    const uint32_t stageCount = std::min(header.stageCount, TileXR::TILEXR_PERF_STAGE_COUNT);
    if (stageCount == 0) {
        return {};
    }

    std::vector<PerfStageSummary> summaries(stageCount);
    for (uint32_t stageId = 0; stageId < stageCount; ++stageId) {
        summaries[stageId].stageId = stageId;
        summaries[stageId].stageName = PerfStageName(stageId);
    }

    for (const auto &stat : stats) {
        if (stat.count == 0 || stat.stageId >= stageCount) {
            continue;
        }

        auto &summary = summaries[stat.stageId];
        summary.count += stat.count;
        summary.sumCycles += stat.sumCycles;
        summary.sumUs += TileXR::PerfTraceCyclesToUs(stat.sumCycles, header.cycleToUsDivisor);
        if (stat.maxCycles > summary.maxCycles) {
            summary.maxCycles = stat.maxCycles;
            summary.maxRank = stat.rank;
            summary.maxCore = stat.core;
        }
    }

    std::vector<PerfStageSummary> nonEmpty;
    for (const auto &summary : summaries) {
        if (summary.count != 0) {
            nonEmpty.push_back(summary);
        }
    }

    std::sort(nonEmpty.begin(), nonEmpty.end(),
        [](const PerfStageSummary &lhs, const PerfStageSummary &rhs) {
            if (lhs.sumCycles != rhs.sumCycles) {
                return lhs.sumCycles > rhs.sumCycles;
            }
            return lhs.stageId < rhs.stageId;
        });
    return nonEmpty;
}

std::vector<std::string> AnalyzePerfTrace(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<PerfStageSummary> &summaries)
{
    (void)header;
    if (summaries.empty()) {
        return { "No profiling samples were recorded." };
    }

    std::vector<std::string> findings;
    {
        std::ostringstream finding;
        finding << "Top bottleneck stage is " << summaries.front().stageName
                << " with " << summaries.front().sumUs << " us recorded.";
        findings.push_back(finding.str());
    }

    for (const auto &summary : summaries) {
        if (summary.stageId == static_cast<uint32_t>(TileXR::PerfStageId::FLAG_POLL_WAIT)) {
            std::ostringstream finding;
            finding << "flag_poll_wait consumed " << summary.sumUs
                    << " us; inspect peer readiness and flag polling balance.";
            findings.push_back(finding.str());
        } else if (summary.stageId == static_cast<uint32_t>(TileXR::PerfStageId::PEER_IPC_TO_OUTPUT)) {
            std::ostringstream finding;
            finding << "peer_ipc_to_output consumed " << summary.sumUs
                    << " us; inspect IPC read/write bandwidth and chunk sizing.";
            findings.push_back(finding.str());
        }
    }
    return findings;
}

int WritePerfTraceReports(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats,
    const PerfReportOptions &options)
{
    if (!IsValidTraceHeader(header)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    if (!EnsureOutputDir(options.outputDir)) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }

    const auto summaries = SummarizePerfTrace(header, stats);
    const auto findings = AnalyzePerfTrace(header, summaries);

    if (!WriteTextFile(JoinPath(options.outputDir, "trace.json"), BuildTraceJson(header, stats)) ||
        !WriteTextFile(JoinPath(options.outputDir, "summary.csv"), BuildSummaryCsv(summaries)) ||
        !WriteTextFile(JoinPath(options.outputDir, "analysis.md"), BuildAnalysisMarkdown(header, summaries, findings)) ||
        !WriteTextFile(JoinPath(options.outputDir, "report.html"), BuildHtmlReport(header, summaries, findings, stats))) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }

    if (options.emitAiPrompt &&
        !WriteTextFile(JoinPath(options.outputDir, "ai_prompt.md"), BuildAiPrompt(header, summaries, findings))) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    if (!options.emitAiPrompt && !RemoveIfExists(JoinPath(options.outputDir, "ai_prompt.md"))) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }

    return TileXR::TILEXR_SUCCESS;
}

} // namespace Host
} // namespace TileXRCollectives
