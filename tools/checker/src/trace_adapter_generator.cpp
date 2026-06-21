#include "tilexr/checker/trace_adapter_generator.h"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace tilexr {
namespace checker {

namespace {

struct SourceSymbolObservation {
    std::string symbol;
    std::vector<int> lines;
    std::vector<std::string> snippets;
    std::string note;
};

struct SourceObservation {
    bool available = false;
    std::vector<SourceSymbolObservation> known_hook_hits;
    std::vector<SourceSymbolObservation> manual_review_candidates;
};

struct SymbolPattern {
    const char *symbol;
    const char *pattern;
    const char *note;
};

const SymbolPattern kKnownHookPatterns[] = {
    {"CpGM2GMPingPong", "CpGM2GMPingPong", "covered by collective_trace_adapter macro"},
    {"SetSyncFlag", "SetSyncFlag", "covered by collective_trace_adapter macro or SyncCollectives shim"},
    {"WaitSyncFlag", "WaitSyncFlag", "covered by collective_trace_adapter macro or SyncCollectives shim"},
    {"AscendC::SetFlag", "AscendC::SetFlag", "covered by kernel_operator pipe trace hook"},
    {"AscendC::WaitFlag", "AscendC::WaitFlag", "covered by kernel_operator pipe trace hook"},
    {"AscendC::PipeBarrier", "AscendC::PipeBarrier", "covered by kernel_operator pipe trace hook"},
    {"PipeBarrier", "PipeBarrier", "covered by kernel_operator pipe trace hook when routed through AscendC shim"},
};

const SymbolPattern kManualReviewPatterns[] = {
    {"DataCopy", "DataCopy", "raw AscendC DataCopy is not a GM2GM oracle event unless wrapped by a traced helper"},
    {"DataCopyPad", "DataCopyPad", "raw AscendC DataCopyPad is not a GM2GM oracle event unless wrapped by a traced helper"},
    {"SetAtomic", "SetAtomic", "atomic accumulation semantics need op-specific oracle review"},
    {"SetAtomicAdd", "SetAtomicAdd", "atomic add accumulation semantics need op-specific oracle review"},
    {"SetAtomicMax", "SetAtomicMax", "atomic max accumulation semantics need op-specific oracle review"},
    {"SetAtomicMin", "SetAtomicMin", "atomic min accumulation semantics need op-specific oracle review"},
};

bool EmptyOrContainsQuote(const std::string &text) {
    return text.empty() || text.find('"') != std::string::npos;
}

bool IsIdentifier(const std::string &text) {
    if (text.empty()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(text[0]);
    if (!std::isalpha(first) && text[0] != '_') {
        return false;
    }
    for (size_t i = 1; i < text.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (!std::isalnum(ch) && text[i] != '_') {
            return false;
        }
    }
    return true;
}

bool IsIdentifierChar(char ch) {
    const unsigned char value = static_cast<unsigned char>(ch);
    return std::isalnum(value) || ch == '_';
}

bool SymbolBoundaryMatchAt(const std::string &line, const std::string &pattern,
                           size_t pos) {
    if (pos > 0 && IsIdentifierChar(line[pos - 1])) {
        return false;
    }
    const size_t end = pos + pattern.size();
    if (end < line.size() && IsIdentifierChar(line[end])) {
        return false;
    }
    return true;
}

bool ContainsSymbolPattern(const std::string &line, const std::string &pattern) {
    size_t pos = line.find(pattern);
    while (pos != std::string::npos) {
        if (SymbolBoundaryMatchAt(line, pattern, pos)) {
            return true;
        }
        pos = line.find(pattern, pos + 1);
    }
    return false;
}

std::string BaseName(const std::string &path) {
    const std::string::size_type pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

std::string StripHeaderExtension(const std::string &path) {
    std::string name = BaseName(path);
    const std::string::size_type pos = name.rfind('.');
    if (pos != std::string::npos) {
        name = name.substr(0, pos);
    }
    return name;
}

std::string InferTargetHeader(const std::string &source_file) {
    const std::string collective_prefix = "src/collectives/kernels/";
    if (source_file.find(collective_prefix) == 0) {
        return source_file.substr(collective_prefix.size());
    }

    const std::string collective_marker = "/src/collectives/kernels/";
    const std::string::size_type collective_pos = source_file.find(collective_marker);
    if (collective_pos != std::string::npos) {
        return source_file.substr(collective_pos + collective_marker.size());
    }

    const std::string ep_prefix = "src/ep/";
    if (source_file.find(ep_prefix) == 0) {
        return BaseName(source_file);
    }

    const std::string ep_marker = "/src/ep/";
    if (source_file.find(ep_marker) != std::string::npos) {
        return BaseName(source_file);
    }

    return BaseName(source_file);
}

std::string GuardName(const std::string &adapter_name) {
    std::string guard = "TILEXR_CHECKER_";
    for (size_t i = 0; i < adapter_name.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(adapter_name[i]);
        if (std::isalnum(ch)) {
            guard += static_cast<char>(std::toupper(ch));
        } else {
            guard += "_";
        }
    }
    guard += "_TRACE_SHIM_H";
    return guard;
}

std::string AdapterNamespace(const std::string &adapter_name) {
    std::string name = "trace_adapter_";
    for (size_t i = 0; i < adapter_name.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(adapter_name[i]);
        if (std::isalnum(ch) || adapter_name[i] == '_') {
            name += static_cast<char>(std::tolower(ch));
        } else {
            name += "_";
        }
    }
    if (name == "trace_adapter_") {
        name += "generated";
    }
    return name;
}

bool IsProductionSourcePath(const std::string &path) {
    return path.find("src/collectives/") == 0 || path.find("/src/collectives/") != std::string::npos ||
           path.find("src/ep/") == 0 || path.find("/src/ep/") != std::string::npos;
}

std::string Trim(const std::string &text) {
    std::string::size_type begin = 0;
    while (begin < text.size() &&
           std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    std::string::size_type end = text.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string StripCommentsAndStringsFromLine(const std::string &line,
                                            bool *in_block_comment) {
    std::string cleaned;
    bool in_string = false;
    bool in_char = false;
    bool escaped = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        const char next = i + 1 < line.size() ? line[i + 1] : '\0';

        if (in_block_comment != nullptr && *in_block_comment) {
            if (ch == '*' && next == '/') {
                *in_block_comment = false;
                cleaned += "  ";
                ++i;
            } else {
                cleaned += ' ';
            }
            continue;
        }

        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            cleaned += ' ';
            continue;
        }

        if (in_char) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '\'') {
                in_char = false;
            }
            cleaned += ' ';
            continue;
        }

        if (ch == '/' && next == '/') {
            cleaned.append(line.size() - i, ' ');
            break;
        }
        if (ch == '/' && next == '*') {
            if (in_block_comment != nullptr) {
                *in_block_comment = true;
            }
            cleaned += "  ";
            ++i;
            continue;
        }
        if (ch == '"') {
            in_string = true;
            cleaned += ' ';
            continue;
        }
        if (ch == '\'') {
            in_char = true;
            cleaned += ' ';
            continue;
        }

        cleaned += ch;
    }
    return cleaned;
}

void AddObservationLine(const SymbolPattern &pattern, int line,
                        const std::string &snippet,
                        std::vector<SourceSymbolObservation> *observations) {
    for (size_t i = 0; i < observations->size(); ++i) {
        if ((*observations)[i].symbol == pattern.symbol) {
            (*observations)[i].lines.push_back(line);
            (*observations)[i].snippets.push_back(snippet);
            return;
        }
    }

    SourceSymbolObservation observation;
    observation.symbol = pattern.symbol;
    observation.note = pattern.note;
    observation.lines.push_back(line);
    observation.snippets.push_back(snippet);
    observations->push_back(observation);
}

SourceObservation ObserveSourceSymbols(const std::string &source_file) {
    SourceObservation observation;
    std::ifstream input(source_file.c_str());
    if (!input.is_open()) {
        return observation;
    }

    observation.available = true;
    std::string line;
    int line_no = 0;
    bool in_block_comment = false;
    while (std::getline(input, line)) {
        ++line_no;
        const std::string snippet = Trim(line);
        const std::string scan_line =
            StripCommentsAndStringsFromLine(line, &in_block_comment);
        for (size_t i = 0; i < sizeof(kKnownHookPatterns) / sizeof(kKnownHookPatterns[0]); ++i) {
            if (std::string(kKnownHookPatterns[i].symbol) == "PipeBarrier" &&
                ContainsSymbolPattern(scan_line, "AscendC::PipeBarrier")) {
                continue;
            }
            if (ContainsSymbolPattern(scan_line, kKnownHookPatterns[i].pattern)) {
                AddObservationLine(kKnownHookPatterns[i], line_no,
                                   snippet,
                                   &observation.known_hook_hits);
            }
        }
        for (size_t i = 0; i < sizeof(kManualReviewPatterns) / sizeof(kManualReviewPatterns[0]); ++i) {
            if (ContainsSymbolPattern(scan_line, kManualReviewPatterns[i].pattern)) {
                AddObservationLine(kManualReviewPatterns[i], line_no,
                                   snippet,
                                   &observation.manual_review_candidates);
            }
        }
    }
    return observation;
}

std::string EscapeJson(const std::string &text) {
    std::string escaped;
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

void RenderLineArray(const std::vector<int> &lines, std::ostream *out) {
    (*out) << "[";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i != 0) {
            (*out) << ",";
        }
        (*out) << lines[i];
    }
    (*out) << "]";
}

void RenderSymbolObservations(const std::vector<SourceSymbolObservation> &observations,
                              std::ostream *out) {
    (*out) << "[";
    for (size_t i = 0; i < observations.size(); ++i) {
        if (i != 0) {
            (*out) << ",";
        }
        (*out) << "{"
               << "\"symbol\":\"" << EscapeJson(observations[i].symbol) << "\","
               << "\"lines\":";
        RenderLineArray(observations[i].lines, out);
        (*out) << ",\"occurrences\":[";
        for (size_t j = 0; j < observations[i].lines.size(); ++j) {
            if (j != 0) {
                (*out) << ",";
            }
            (*out) << "{\"line\":" << observations[i].lines[j]
                   << ",\"snippet\":\"";
            if (j < observations[i].snippets.size()) {
                (*out) << EscapeJson(observations[i].snippets[j]);
            }
            (*out) << "\"}";
        }
        (*out) << "]"
               << ",\"note\":\"" << EscapeJson(observations[i].note) << "\""
               << "}";
    }
    (*out) << "]";
}

std::string ManualReviewActionForSymbol(const std::string &symbol) {
    if (symbol == "source_unavailable") {
        return "fix the source path or provide an explicit unsupported gate before installing this adapter";
    }
    if (symbol == "coverage_decision") {
        return "add a traced runner/materializer coverage decision or explicit unsupported gate";
    }
    if (symbol == "DataCopy" || symbol == "DataCopyPad") {
        return "add a trace wrapper or explicit unsupported gate for " + symbol;
    }
    if (symbol.find("SetAtomic") == 0) {
        return "add an op-specific atomic oracle/materializer or explicit unsupported gate for " +
               symbol;
    }
    return "add an op-specific trace model or explicit unsupported gate for " + symbol;
}

std::string ManualReviewReasonForSymbol(const std::string &symbol) {
    if (symbol == "source_unavailable") {
        return "source file could not be read";
    }
    if (symbol == "coverage_decision") {
        return "no recognized checker trace hook was observed";
    }
    return "";
}

void RenderManualReviewActions(const std::string &source_file,
                               const std::vector<SourceSymbolObservation> &observations,
                               bool source_unavailable,
                               bool needs_coverage_decision,
                               std::ostream *out) {
    (*out) << "[";
    bool first = true;
    if (source_unavailable) {
        (*out) << "{"
               << "\"symbol\":\"source_unavailable\","
               << "\"source_file\":\"" << EscapeJson(source_file) << "\","
               << "\"line\":0,"
               << "\"action\":\""
               << EscapeJson(ManualReviewActionForSymbol("source_unavailable"))
               << "\","
               << "\"status\":\"pending\","
               << "\"reason\":\"source file could not be read\""
               << "}";
        first = false;
    }
    if (needs_coverage_decision) {
        (*out) << "{"
               << "\"symbol\":\"coverage_decision\","
               << "\"source_file\":\"" << EscapeJson(source_file) << "\","
               << "\"line\":0,"
               << "\"action\":\""
               << EscapeJson(ManualReviewActionForSymbol("coverage_decision"))
               << "\","
               << "\"status\":\"pending\","
               << "\"reason\":\"no recognized checker trace hook was observed\""
               << "}";
        first = false;
    }
    for (size_t i = 0; i < observations.size(); ++i) {
        for (size_t j = 0; j < observations[i].lines.size(); ++j) {
            if (!first) {
                (*out) << ",";
            }
            first = false;
            (*out) << "{"
                   << "\"symbol\":\"" << EscapeJson(observations[i].symbol) << "\","
                   << "\"source_file\":\"" << EscapeJson(source_file) << "\","
                   << "\"line\":" << observations[i].lines[j] << ","
                   << "\"action\":\""
                   << EscapeJson(ManualReviewActionForSymbol(observations[i].symbol))
                   << "\","
                   << "\"status\":\"pending\""
                   << "}";
        }
    }
    (*out) << "]";
}

void RenderManualReviewPlanMarkdown(
        const SourceObservation &observation,
        std::ostream *out) {
    const bool needs_coverage_decision =
        observation.available && observation.known_hook_hits.empty();
    if (!observation.available) {
        (*out) << "## Manual Review Required\n\n"
               << "- `source_unavailable` line 0: "
               << ManualReviewActionForSymbol("source_unavailable")
               << " (`status: pending`; source file could not be read)\n\n";
        return;
    }
    if (observation.manual_review_candidates.empty() && !needs_coverage_decision) {
        (*out) << "## Manual Review\n\n"
               << "No raw DataCopy/DataCopyPad or SetAtomic* candidates were found by "
               << "the lightweight source scan.\n\n";
        return;
    }

    (*out) << "## Manual Review Required\n\n"
           << "The source scan found primitives that are not automatically covered by "
           << "the generic trace adapter. Keep the manifest `required_actions` entries "
           << "as `pending` until each item has an explicit checker-side decision.\n\n";
    if (needs_coverage_decision) {
        (*out) << "- `coverage_decision` line 0: "
               << ManualReviewActionForSymbol("coverage_decision")
               << " (`status: pending`; no recognized checker trace hook was observed)\n";
    }
    for (size_t i = 0; i < observation.manual_review_candidates.size(); ++i) {
        const SourceSymbolObservation &candidate =
            observation.manual_review_candidates[i];
        for (size_t j = 0; j < candidate.lines.size(); ++j) {
            (*out) << "- `" << candidate.symbol << "` line "
                   << candidate.lines[j] << ": "
                   << ManualReviewActionForSymbol(candidate.symbol)
                   << " (`status: pending`)\n";
        }
    }
    (*out) << "\n";
}

void RenderMetadataHookArrays(const std::vector<SourceSymbolObservation> &observations,
                              const std::string &prefix, std::ostream *out) {
    for (size_t i = 0; i < observations.size(); ++i) {
        (*out) << "static const int " << prefix << i << "Lines[] = {";
        for (size_t j = 0; j < observations[i].lines.size(); ++j) {
            if (j != 0) {
                (*out) << ", ";
            }
            (*out) << observations[i].lines[j];
        }
        (*out) << "};\n";
    }
}

void RenderMetadataHookObservations(const std::vector<SourceSymbolObservation> &observations,
                                    const std::string &array_name,
                                    const std::string &prefix,
                                    std::ostream *out) {
    if (observations.empty()) {
        return;
    }
    (*out) << "static const TraceAdapterHookObservation " << array_name << "[] = {\n";
    for (size_t i = 0; i < observations.size(); ++i) {
        (*out) << "    {\"" << EscapeJson(observations[i].symbol) << "\", "
               << prefix << i << "Lines, "
               << observations[i].lines.size() << "}";
        if (i + 1 != observations.size()) {
            (*out) << ",";
        }
        (*out) << "\n";
    }
    (*out) << "};\n";
}

void RenderTraceAdapterMetadataBlock(const TraceAdapterSpec &spec,
                                     const SourceObservation &observation,
                                     const std::string &guard,
                                     std::ostream *out) {
    const std::string ns = AdapterNamespace(spec.adapter_name);
    (*out) << "namespace tilexr {\n"
           << "namespace checker {\n"
           << "namespace " << ns << " {\n\n"
           << "static const int kKnownHookCount = "
           << observation.known_hook_hits.size() << ";\n"
           << "static const int kManualReviewCandidateCount = "
           << observation.manual_review_candidates.size() << ";\n";
    RenderMetadataHookArrays(observation.known_hook_hits, "kKnownHook", out);
    RenderMetadataHookObservations(observation.known_hook_hits, "kKnownHooks",
                                   "kKnownHook", out);
    RenderMetadataHookArrays(observation.manual_review_candidates,
                             "kManualReviewCandidate", out);
    RenderMetadataHookObservations(observation.manual_review_candidates,
                                   "kManualReviewCandidates",
                                   "kManualReviewCandidate", out);
    (*out) << "\ninline const TraceAdapterMetadata &Metadata()\n"
           << "{\n"
           << "    static const TraceAdapterMetadata metadata = {\n"
           << "        \"" << EscapeJson(spec.adapter_name) << "\",\n"
           << "        \"" << EscapeJson(spec.source_file) << "\",\n"
           << "        \"" << EscapeJson(spec.target_header) << "\",\n"
           << "        \"" << EscapeJson(guard) << "\",\n"
           << "        ";
    if (observation.known_hook_hits.empty()) {
        (*out) << "nullptr";
    } else {
        (*out) << "kKnownHooks";
    }
    (*out) << ", " << observation.known_hook_hits.size() << ",\n"
           << "        ";
    if (observation.manual_review_candidates.empty()) {
        (*out) << "nullptr";
    } else {
        (*out) << "kManualReviewCandidates";
    }
    (*out) << ", " << observation.manual_review_candidates.size() << "\n"
           << "    };\n"
           << "    return metadata;\n"
           << "}\n\n"
           << "inline bool Audit(const char **reason)\n"
           << "{\n"
           << "    return AuditTraceAdapterMetadata(Metadata(), reason);\n"
           << "}\n\n"
           << "}  // namespace " << ns << "\n"
           << "}  // namespace checker\n"
           << "}  // namespace tilexr\n";
}

const char *CoverageEventForSymbol(const std::string &symbol) {
    if (symbol == "CpGM2GMPingPong") {
        return "COPY";
    }
    if (symbol == "SetSyncFlag") {
        return "FLAG_STORE";
    }
    if (symbol == "WaitSyncFlag") {
        return "FLAG_WAIT";
    }
    if (symbol == "AscendC::SetFlag") {
        return "PIPE_SET";
    }
    if (symbol == "AscendC::WaitFlag") {
        return "PIPE_WAIT";
    }
    if (symbol == "AscendC::PipeBarrier" || symbol == "PipeBarrier") {
        return "PIPE_BARRIER";
    }
    return nullptr;
}

std::string CandidateMacroForSymbol(const std::string &symbol) {
    if (symbol == "AscendC::SetFlag") {
        return "AscendC::SetFlag";
    }
    if (symbol == "AscendC::WaitFlag") {
        return "AscendC::WaitFlag";
    }
    if (symbol == "AscendC::PipeBarrier") {
        return "AscendC::PipeBarrier";
    }
    return symbol;
}

void RenderRequiredEventCoverageSeed(
        const std::string &source_file,
        const std::vector<SourceSymbolObservation> &observations,
        std::ostream *out) {
    (*out) << "[";
    bool first = true;
    for (size_t i = 0; i < observations.size(); ++i) {
        const char *event_kind = CoverageEventForSymbol(observations[i].symbol);
        if (event_kind == nullptr) {
            continue;
        }
        for (size_t j = 0; j < observations[i].lines.size(); ++j) {
            if (!first) {
                (*out) << ",";
            }
            first = false;
            (*out) << "{\"kind\":\"" << event_kind << "\","
                   << "\"source_file\":\"" << EscapeJson(source_file) << "\","
                   << "\"source_line\":" << observations[i].lines[j] << "}";
        }
    }
    (*out) << "]";
}

TraceSourceObservationItem MakeTraceSourceObservationItem(
        const SourceSymbolObservation &observation,
        bool manual_review_required) {
    TraceSourceObservationItem item;
    item.symbol = observation.symbol;
    item.note = observation.note;
    item.lines = observation.lines;
    item.snippets = observation.snippets;
    item.manual_review_required = manual_review_required;
    item.macro_name = CandidateMacroForSymbol(observation.symbol);
    const char *event = CoverageEventForSymbol(observation.symbol);
    if (event != nullptr) {
        item.event = event;
    } else {
        item.event = "manual_review";
    }
    return item;
}

TraceSourceObservationItem MakeSyntheticTraceSourceAction(
        const std::string &symbol) {
    TraceSourceObservationItem item;
    item.symbol = symbol;
    item.event = "manual_review";
    item.lines.push_back(0);
    item.note = ManualReviewReasonForSymbol(symbol);
    item.manual_review_required = true;
    return item;
}

void RenderSemiAutomaticHookCandidates(
        const TraceSourceAnalysis &analysis,
        std::ostream *out) {
    (*out) << "[";
    bool first = true;
    for (size_t i = 0; i < analysis.auto_hook_candidates.size(); ++i) {
        const TraceSourceObservationItem &candidate =
            analysis.auto_hook_candidates[i];
        for (size_t j = 0; j < candidate.lines.size(); ++j) {
            if (!first) {
                (*out) << ",";
            }
            first = false;
            (*out) << "{"
                   << "\"symbol\":\"" << EscapeJson(candidate.symbol) << "\","
                   << "\"event\":\"" << EscapeJson(candidate.event) << "\","
                   << "\"source_file\":\"" << EscapeJson(analysis.source_file) << "\","
                   << "\"line\":" << candidate.lines[j] << ","
                   << "\"macro\":\"" << EscapeJson(candidate.macro_name) << "\","
                   << "\"status\":\"auto_traceable\","
                   << "\"location\":\"checker shim/header macro\"";
            if (j < candidate.snippets.size()) {
                (*out) << ",\"snippet\":\""
                       << EscapeJson(candidate.snippets[j]) << "\"";
            }
            (*out) << "}";
        }
    }
    (*out) << "]";
}

void RenderTraceSourceAnalysisItems(
        const std::vector<TraceSourceObservationItem> &items,
        const std::string &source_file,
        bool manual_actions,
        std::ostream *out) {
    (*out) << "[";
    bool first = true;
    for (size_t i = 0; i < items.size(); ++i) {
        const TraceSourceObservationItem &item = items[i];
        for (size_t j = 0; j < item.lines.size(); ++j) {
            if (!first) {
                (*out) << ",";
            }
            first = false;
            (*out) << "{"
                   << "\"symbol\":\"" << EscapeJson(item.symbol) << "\","
                   << "\"event\":\"" << EscapeJson(item.event) << "\","
                   << "\"source_file\":\"" << EscapeJson(source_file) << "\","
                   << "\"line\":" << item.lines[j];
            if (manual_actions) {
                (*out) << ",\"action\":\""
                       << EscapeJson(ManualReviewActionForSymbol(item.symbol))
                       << "\",\"status\":\"pending\"";
                const std::string reason = item.note.empty()
                                               ? ManualReviewReasonForSymbol(item.symbol)
                                               : item.note;
                if (!reason.empty()) {
                    (*out) << ",\"reason\":\"" << EscapeJson(reason) << "\"";
                }
            } else {
                (*out) << ",\"macro\":\"" << EscapeJson(item.macro_name) << "\","
                       << "\"status\":\"auto_traceable\"";
            }
            if (j < item.snippets.size()) {
                (*out) << ",\"snippet\":\"" << EscapeJson(item.snippets[j]) << "\"";
            }
            (*out) << "}";
        }
    }
    (*out) << "]";
}

size_t CountTraceSourceOccurrences(
        const std::vector<TraceSourceObservationItem> &items) {
    size_t count = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        count += items[i].lines.size();
    }
    return count;
}

void RenderSemiAutomaticTraceSection(const TraceSourceAnalysis &analysis,
                                     std::ostream *out) {
    (*out) << "\"semi_automatic_trace\":{"
           << "\"method\":\"header_macro_shim\","
           << "\"source_file\":\"" << EscapeJson(analysis.source_file) << "\","
           << "\"source_scan_file\":\"" << EscapeJson(analysis.source_scan_file) << "\","
           << "\"header_stub\":{"
           << "\"shim_header\":\"" << EscapeJson(analysis.shim_header) << "\","
           << "\"generic_adapter\":\"tilexr/checker/collective_trace_adapter.h\","
           << "\"source_define\":\"TILEXR_CHECKER_TRACE_SOURCE_FILE\","
           << "\"target_header_define\":\"TILEXR_CHECKER_TRACE_TARGET_HEADER\","
           << "\"target_header\":\"" << EscapeJson(analysis.target_header) << "\""
           << "},"
           << "\"auto_hook_candidates\":";
    RenderSemiAutomaticHookCandidates(analysis, out);
    (*out) << ",\"manual_review_actions\":";
    RenderTraceSourceAnalysisItems(analysis.manual_review_candidates,
                                   analysis.source_file, true, out);
    (*out) << "}";
}

TraceSourceAnalysis BuildTraceSourceAnalysis(
        const TraceAdapterSpec &spec,
        const std::string &shim_header,
        const SourceObservation &observation) {
    TraceSourceAnalysis analysis;
    analysis.adapter_name = spec.adapter_name;
    analysis.source_file = spec.source_file;
    analysis.source_scan_file = spec.source_scan_file;
    analysis.target_header = spec.target_header;
    analysis.shim_header = shim_header;
    analysis.available = observation.available;
    analysis.manual_review_required =
        !observation.available || !observation.manual_review_candidates.empty() ||
        (observation.available && observation.known_hook_hits.empty());
    for (size_t i = 0; i < observation.known_hook_hits.size(); ++i) {
        analysis.auto_hook_candidates.push_back(
            MakeTraceSourceObservationItem(observation.known_hook_hits[i], false));
    }
    for (size_t i = 0; i < observation.manual_review_candidates.size(); ++i) {
        analysis.manual_review_candidates.push_back(
            MakeTraceSourceObservationItem(observation.manual_review_candidates[i], true));
    }
    if (!observation.available) {
        analysis.manual_review_candidates.push_back(
            MakeSyntheticTraceSourceAction("source_unavailable"));
    } else if (observation.known_hook_hits.empty()) {
        analysis.manual_review_candidates.push_back(
            MakeSyntheticTraceSourceAction("coverage_decision"));
    }
    return analysis;
}

std::string JoinObservationSymbols(const std::vector<SourceSymbolObservation> &observations) {
    std::ostringstream out;
    for (size_t i = 0; i < observations.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << observations[i].symbol;
    }
    return out.str();
}

CheckerStatus WriteFile(const std::string &path, const std::string &content,
                        const std::string &kind) {
    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        return CheckerStatus::Fail("failed to open " + kind + " output file: " + path);
    }
    output << content;
    output.close();
    if (!output) {
        return CheckerStatus::Fail("failed to write " + kind + " output file: " + path);
    }
    return CheckerStatus::Ok();
}

}  // namespace

CheckerStatus NormalizeTraceAdapterSpec(TraceAdapterSpec *spec) {
    if (spec == nullptr) {
        return CheckerStatus::Fail("trace adapter spec pointer is null");
    }
    if (spec->source_file.empty() || spec->source_file.find('"') != std::string::npos) {
        return CheckerStatus::Unsupported(
            "trace adapter source file must be non-empty and quote-free");
    }
    if (spec->adapter_name.empty()) {
        spec->adapter_name = StripHeaderExtension(spec->source_file);
    }
    if (spec->target_header.empty()) {
        spec->target_header = InferTargetHeader(spec->source_file);
    }
    if (spec->source_scan_file.empty()) {
        spec->source_scan_file = spec->source_file;
    }
    if (EmptyOrContainsQuote(spec->adapter_name) ||
        EmptyOrContainsQuote(spec->target_header) ||
        spec->source_scan_file.find('"') != std::string::npos) {
        return CheckerStatus::Unsupported(
            "trace adapter inferred fields must be non-empty and quote-free");
    }
    return CheckerStatus::Ok();
}

CheckerStatus AnalyzeTraceSource(const TraceAdapterSpec &spec,
                                 const std::string &shim_header,
                                 TraceSourceAnalysis *analysis) {
    if (analysis == nullptr) {
        return CheckerStatus::Fail("trace source analysis pointer is null");
    }
    TraceAdapterSpec normalized = spec;
    CheckerStatus normalize_status = NormalizeTraceAdapterSpec(&normalized);
    if (!normalize_status.ok()) {
        return normalize_status;
    }
    if (EmptyOrContainsQuote(shim_header)) {
        return CheckerStatus::Unsupported(
            "trace source analysis shim header must be non-empty and quote-free");
    }
    const SourceObservation observation =
        ObserveSourceSymbols(normalized.source_scan_file);
    *analysis = BuildTraceSourceAnalysis(normalized, shim_header, observation);
    return CheckerStatus::Ok();
}

CheckerStatus RenderTraceSourceAnalysisJson(const TraceSourceAnalysis &analysis,
                                            std::string *output) {
    if (output == nullptr) {
        return CheckerStatus::Fail("trace source analysis json output pointer is null");
    }
    std::ostringstream out;
    out << "{"
        << "\"mode\":\"trace_source_analysis\","
        << "\"adapter_name\":\"" << EscapeJson(analysis.adapter_name) << "\","
        << "\"source_file\":\"" << EscapeJson(analysis.source_file) << "\","
        << "\"source_scan_file\":\"" << EscapeJson(analysis.source_scan_file) << "\","
        << "\"target_header\":\"" << EscapeJson(analysis.target_header) << "\","
        << "\"available\":" << (analysis.available ? "true" : "false") << ","
        << "\"manual_review_required\":"
        << (analysis.manual_review_required ? "true" : "false") << ",";
    RenderSemiAutomaticTraceSection(analysis, &out);
    out << ",\"auto_hook_candidates\":";
    RenderTraceSourceAnalysisItems(analysis.auto_hook_candidates,
                                   analysis.source_file, false, &out);
    out << ",\"manual_review_actions\":";
    RenderTraceSourceAnalysisItems(analysis.manual_review_candidates,
                                   analysis.source_file, true, &out);
    out << "}";
    *output = out.str();
    return CheckerStatus::Ok();
}

CheckerStatus RenderTraceSourceAnalysisText(const TraceSourceAnalysis &analysis,
                                            std::string *output) {
    if (output == nullptr) {
        return CheckerStatus::Fail("trace source analysis text output pointer is null");
    }
    std::ostringstream out;
    out << "trace source: " << analysis.source_file << "\n"
        << "scan file: " << analysis.source_scan_file << "\n"
        << "header stub: " << analysis.shim_header << " -> "
        << analysis.target_header << "\n"
        << "available: " << (analysis.available ? "yes" : "no") << "\n"
        << "auto hook symbols: " << analysis.auto_hook_candidates.size() << "\n"
        << "auto hook occurrences: "
        << CountTraceSourceOccurrences(analysis.auto_hook_candidates) << "\n";
    for (size_t i = 0; i < analysis.auto_hook_candidates.size(); ++i) {
        const TraceSourceObservationItem &item = analysis.auto_hook_candidates[i];
        for (size_t j = 0; j < item.lines.size(); ++j) {
            out << "  " << item.symbol << " line " << item.lines[j]
                << " -> " << item.event << " via " << item.macro_name << "\n";
        }
    }
    out << "manual review symbols: " << analysis.manual_review_candidates.size() << "\n"
        << "manual review actions: "
        << CountTraceSourceOccurrences(analysis.manual_review_candidates) << "\n";
    for (size_t i = 0; i < analysis.manual_review_candidates.size(); ++i) {
        const TraceSourceObservationItem &item = analysis.manual_review_candidates[i];
        for (size_t j = 0; j < item.lines.size(); ++j) {
            out << "  " << item.symbol << " line " << item.lines[j]
                << " -> " << ManualReviewActionForSymbol(item.symbol);
            const std::string reason = item.note.empty()
                                           ? ManualReviewReasonForSymbol(item.symbol)
                                           : item.note;
            if (!reason.empty()) {
                out << " (" << reason << ")";
            }
            out << "\n";
        }
    }
    *output = out.str();
    return CheckerStatus::Ok();
}

CheckerStatus RenderTraceAdapter(const TraceAdapterSpec &spec, std::string *output) {
    if (output == nullptr) {
        return CheckerStatus::Fail("trace adapter output pointer is null");
    }
    TraceAdapterSpec normalized = spec;
    CheckerStatus normalize_status = NormalizeTraceAdapterSpec(&normalized);
    if (!normalize_status.ok()) {
        return normalize_status;
    }
    if (EmptyOrContainsQuote(normalized.adapter_name) ||
        EmptyOrContainsQuote(normalized.source_file) ||
        EmptyOrContainsQuote(normalized.target_header)) {
        return CheckerStatus::Unsupported("trace adapter spec fields must be non-empty and quote-free");
    }

    const std::string guard = GuardName(normalized.adapter_name);
    const SourceObservation source_observation =
        ObserveSourceSymbols(normalized.source_scan_file);
    std::ostringstream out;
    out << "#ifndef " << guard << "\n"
        << "#define " << guard << "\n\n"
        << "#define TILEXR_CHECKER_TRACE_SOURCE_FILE \"" << normalized.source_file << "\"\n"
        << "#define TILEXR_CHECKER_TRACE_TARGET_HEADER \"" << normalized.target_header << "\"\n"
        << "#include \"tilexr/checker/collective_trace_adapter.h\"\n\n";
    RenderTraceAdapterMetadataBlock(normalized, source_observation, guard, &out);
    out
        << "#undef TILEXR_CHECKER_TRACE_TARGET_HEADER\n"
        << "#undef TILEXR_CHECKER_TRACE_SOURCE_FILE\n\n"
        << "#endif  // " << guard << "\n";
    *output = out.str();
    return CheckerStatus::Ok();
}

CheckerStatus WriteTraceAdapterFile(const TraceAdapterSpec &spec, const std::string &path) {
    if (path.empty()) {
        return CheckerStatus::Unsupported("trace adapter output path is empty");
    }
    if (IsProductionSourcePath(path)) {
        return CheckerStatus::Unsupported(
            "trace adapter output must be a checker shim path, not a production source path");
    }

    std::string content;
    CheckerStatus status = RenderTraceAdapter(spec, &content);
    if (!status.ok()) {
        return status;
    }

    return WriteFile(path, content, "trace adapter");
}

CheckerStatus RenderTraceAdapterManifest(const TraceAdapterSpec &spec,
                                         const std::string &shim_header,
                                         std::string *output) {
    if (output == nullptr) {
        return CheckerStatus::Fail("trace adapter manifest output pointer is null");
    }
    TraceAdapterSpec normalized = spec;
    CheckerStatus normalize_status = NormalizeTraceAdapterSpec(&normalized);
    if (!normalize_status.ok()) {
        return normalize_status;
    }
    if (EmptyOrContainsQuote(normalized.adapter_name) ||
        EmptyOrContainsQuote(normalized.source_file) ||
        EmptyOrContainsQuote(normalized.target_header) ||
        EmptyOrContainsQuote(shim_header)) {
        return CheckerStatus::Unsupported(
            "trace adapter manifest fields must be non-empty and quote-free");
    }

    std::ostringstream out;
    const SourceObservation source_observation =
        ObserveSourceSymbols(normalized.source_scan_file);
    const bool needs_coverage_decision =
        source_observation.available && source_observation.known_hook_hits.empty();
    if (normalized.strict_source_observation && !source_observation.available) {
        return CheckerStatus::Unsupported("strict source observation could not read source file: " +
                                          normalized.source_file);
    }
    if (normalized.strict_source_observation &&
        !source_observation.manual_review_candidates.empty()) {
        return CheckerStatus::Unsupported(
            "strict source observation found manual review candidates: " +
            JoinObservationSymbols(source_observation.manual_review_candidates));
    }
    if (normalized.strict_source_observation &&
        needs_coverage_decision) {
        return CheckerStatus::Unsupported(
            "strict source observation found no recognized checker trace hook");
    }
    out << "{"
        << "\"adapter_name\":\"" << EscapeJson(normalized.adapter_name) << "\","
        << "\"source_file\":\"" << EscapeJson(normalized.source_file) << "\","
        << "\"target_header\":\"" << EscapeJson(normalized.target_header) << "\","
        << "\"shim_header\":\"" << EscapeJson(shim_header) << "\","
        << "\"include_guard\":\"" << GuardName(normalized.adapter_name) << "\","
        << "\"metadata_api\":{"
        << "\"namespace\":\"tilexr::checker::" << AdapterNamespace(normalized.adapter_name) << "\","
        << "\"metadata\":\"Metadata\","
        << "\"audit\":\"Audit\""
        << "},"
        << "\"generated_outputs\":{"
        << "\"adapter_header\":\"" << EscapeJson(shim_header) << "\","
        << "\"manifest\":\"generated by --manifest-output\","
        << "\"runner_skeleton\":\"generated when --runner-output is set\","
        << "\"header_probe\":\"generated when --probe-output is set\""
        << "},"
        << "\"trace_hooks\":["
        << "{\"symbol\":\"CpGM2GMPingPong\",\"event\":\"READ/COPY/WRITE\","
        << "\"location\":\"TILEXR_CHECKER_TRACE_SOURCE_FILE\"},"
        << "{\"symbol\":\"SetSyncFlag\",\"event\":\"FLAG_STORE\","
        << "\"location\":\"TILEXR_CHECKER_TRACE_SOURCE_FILE\"},"
        << "{\"symbol\":\"WaitSyncFlag\",\"event\":\"FLAG_WAIT\","
        << "\"location\":\"TILEXR_CHECKER_TRACE_SOURCE_FILE\"},"
        << "{\"symbol\":\"AscendC::SetFlag\",\"event\":\"PIPE_SET\","
        << "\"location\":\"__builtin_FILE/__builtin_LINE\"},"
        << "{\"symbol\":\"AscendC::WaitFlag\",\"event\":\"PIPE_WAIT\","
        << "\"location\":\"__builtin_FILE/__builtin_LINE\"},"
        << "{\"symbol\":\"AscendC::PipeBarrier\",\"event\":\"PIPE_BARRIER\","
        << "\"location\":\"__builtin_FILE/__builtin_LINE\"}"
        << "],"
        << "\"manual_review_required\":"
        << (source_observation.available &&
            source_observation.manual_review_candidates.empty() &&
            !needs_coverage_decision ? "false" : "true") << ","
        << "\"required_actions\":";
    RenderManualReviewActions(normalized.source_file,
                              source_observation.manual_review_candidates,
                              !source_observation.available,
                              !source_observation.available ? false : needs_coverage_decision,
                              &out);
    out << ","
        << "\"source_observation\":{"
        << "\"available\":" << (source_observation.available ? "true" : "false") << ","
        << "\"known_hook_hits\":";
    RenderSymbolObservations(source_observation.known_hook_hits, &out);
    out << ",\"manual_review_candidates\":";
    RenderSymbolObservations(source_observation.manual_review_candidates, &out);
    out << "},";
    const TraceSourceAnalysis analysis =
        BuildTraceSourceAnalysis(normalized, shim_header, source_observation);
    RenderSemiAutomaticTraceSection(analysis, &out);
    out << ","
        << "\"installed_trace_seed\":{"
        << "\"required_event_coverage\":";
    RenderRequiredEventCoverageSeed(normalized.source_file,
                                    source_observation.known_hook_hits, &out);
    out << "},"
        << "\"source_preservation\":{"
        << "\"production_source_must_not_change\":true,"
        << "\"forbidden_output_roots\":[\"src/collectives/\",\"src/ep/\"]"
        << "},"
        << "\"runner_integration\":{"
        << "\"include_shim_header\":\"" << EscapeJson(shim_header) << "\","
        << "\"register_virtual_ranges\":\"RegisterCollectiveTraceRanges\","
        << "\"install_peer_mems\":\"InstallCollectiveTracePeerMems\","
        << "\"set_runtime\":\"TraceRuntime::SetCurrent\","
        << "\"materialize_oracle\":\"add an op-specific materializer before CompareInt32Output\","
        << "\"checklist\":["
        << "\"source_preservation_checked\","
        << "\"trace_ranges_registered\","
        << "\"peer_mems_restored\","
        << "\"block_schedule_defined\","
        << "\"operator_init_process_wired\","
        << "\"oracle_materializer_reviewed\""
        << "]"
        << "},"
        << "\"validation_commands\":["
        << "\"source scripts/common_env.sh && cmake --build build-checker --target tilexr_checker_all -j$(nproc)\","
        << "\"source scripts/common_env.sh && ctest --test-dir build-checker -R checker --output-on-failure\","
        << "\"git diff -- src/collectives src/ep\""
        << "]"
        << "}";
    *output = out.str();
    return CheckerStatus::Ok();
}

CheckerStatus WriteTraceAdapterManifestFile(const TraceAdapterSpec &spec,
                                            const std::string &shim_header,
                                            const std::string &path) {
    if (path.empty()) {
        return CheckerStatus::Unsupported("trace adapter manifest output path is empty");
    }
    if (IsProductionSourcePath(path)) {
        return CheckerStatus::Unsupported(
            "trace adapter manifest output must be a checker output path, not a production source path");
    }

    std::string content;
    CheckerStatus status = RenderTraceAdapterManifest(spec, shim_header, &content);
    if (!status.ok()) {
        return status;
    }
    return WriteFile(path, content, "trace adapter manifest");
}

CheckerStatus RenderTraceRunnerSkeleton(const TraceRunnerSkeletonSpec &spec,
                                        std::string *output) {
    if (output == nullptr) {
        return CheckerStatus::Fail("trace runner skeleton output pointer is null");
    }
    if (!IsIdentifier(spec.function_name) || !IsIdentifier(spec.materializer_name) ||
        EmptyOrContainsQuote(spec.shim_include)) {
        return CheckerStatus::Unsupported(
            "trace runner skeleton fields must use valid identifiers and quote-free include paths");
    }

    std::ostringstream out;
    out << "#include \"tilexr/checker/collective_trace_runner.h\"\n"
        << "#include \"tilexr/checker/trace_runtime.h\"\n"
        << "#include \"" << spec.shim_include << "\"\n\n"
        << "#include <cstddef>\n"
        << "#include <cstdint>\n"
        << "#include <vector>\n\n"
        << "namespace tilexr {\n"
        << "namespace checker {\n\n"
        << "namespace {\n\n"
        << "struct RunnerIntegrationChecklist {\n"
        << "    bool source_preservation_checked = true;\n"
        << "    bool trace_ranges_registered = true;\n"
        << "    bool peer_mems_restored = true;\n"
        << "    bool block_schedule_defined = false;\n"
        << "    bool operator_init_process_wired = false;\n"
        << "    bool oracle_materializer_reviewed = false;\n"
        << "};\n\n"
        << "bool RunnerIntegrationReady(const RunnerIntegrationChecklist &checklist) {\n"
        << "    return checklist.source_preservation_checked &&\n"
        << "           checklist.trace_ranges_registered &&\n"
        << "           checklist.peer_mems_restored &&\n"
        << "           checklist.block_schedule_defined &&\n"
        << "           checklist.operator_init_process_wired &&\n"
        << "           checklist.oracle_materializer_reviewed;\n"
        << "}\n\n"
        << "}  // namespace\n\n"
        << "CheckerStatus " << spec.function_name
        << "(RankWorld *world, const CheckerCase &test_case) {\n"
        << "    if (world == nullptr) {\n"
        << "        return CheckerStatus::Fail(\"rank world is null\");\n"
        << "    }\n"
        << "    RunnerIntegrationChecklist checklist;\n"
        << "    TraceRuntime runtime(world, test_case);\n"
        << "    const size_t output_bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);\n"
        << "    RegisterCollectiveTraceRanges(world, test_case, &runtime, output_bytes);\n"
        << "    const std::vector<std::vector<GM_ADDR> > original_peer_mems =\n"
        << "        InstallCollectiveTracePeerMems(world, test_case);\n"
        << "    TraceRuntime::SetCurrent(&runtime);\n\n"
        << "    // TODO(checker-runner): define the production block schedule and set\n"
        << "    // checklist.block_schedule_defined after choosing block_num/block_idx order.\n"
        << "    // TODO(checker-runner): instantiate the production operator from the shim header,\n"
        << "    // set runtime.SetKernelContext(rank, block_idx, block_num), call Init/Process,\n"
        << "    // and set checklist.operator_init_process_wired.\n"
        << "    // TODO(checker-runner): review the oracle materializer against the operator semantics,\n"
        << "    // then set checklist.oracle_materializer_reviewed before wiring this into\n"
        << "    // CollectiveExecutor.\n\n"
        << "    runtime." << spec.materializer_name << "();\n"
        << "    TraceRuntime::SetCurrent(nullptr);\n"
        << "    RestoreCollectiveTracePeerMems(world, test_case, original_peer_mems);\n"
        << "    if (!RunnerIntegrationReady(checklist)) {\n"
        << "        return CheckerStatus::Unsupported(\n"
        << "            \"generated trace runner still has unchecked integration items\");\n"
        << "    }\n"
        << "    return CheckerStatus::Ok();\n"
        << "}\n\n"
        << "}  // namespace checker\n"
        << "}  // namespace tilexr\n";
    *output = out.str();
    return CheckerStatus::Ok();
}

CheckerStatus WriteTraceRunnerSkeletonFile(const TraceRunnerSkeletonSpec &spec,
                                           const std::string &path) {
    if (path.empty()) {
        return CheckerStatus::Unsupported("trace runner skeleton output path is empty");
    }
    if (IsProductionSourcePath(path)) {
        return CheckerStatus::Unsupported(
            "trace runner skeleton output must be a checker output path, not a production source path");
    }

    std::string content;
    CheckerStatus status = RenderTraceRunnerSkeleton(spec, &content);
    if (!status.ok()) {
        return status;
    }
    return WriteFile(path, content, "trace runner skeleton");
}

CheckerStatus RenderTraceHeaderProbeSkeleton(const TraceHeaderProbeSpec &spec,
                                             std::string *output) {
    if (output == nullptr) {
        return CheckerStatus::Fail("trace header probe output pointer is null");
    }
    if (EmptyOrContainsQuote(spec.adapter_name) ||
        EmptyOrContainsQuote(spec.shim_include) ||
        spec.expected_source_file.find('"') != std::string::npos ||
        spec.expected_target_header.find('"') != std::string::npos) {
        return CheckerStatus::Unsupported(
            "trace header probe fields must be non-empty and quote-free");
    }

    const std::string ns = AdapterNamespace(spec.adapter_name);
    const std::string guard = GuardName(spec.adapter_name);
    std::ostringstream out;
    out << "#include <cstring>\n"
        << "#include <type_traits>\n\n"
        << "#include \"" << EscapeJson(spec.shim_include) << "\"\n\n"
        << "int main() {\n"
        << "    const tilexr::checker::TraceAdapterMetadata &metadata =\n"
        << "        tilexr::checker::" << ns << "::Metadata();\n"
        << "    const char *reason = nullptr;\n"
        << "    if (!tilexr::checker::" << ns << "::Audit(&reason)) {\n"
        << "        return 1;\n"
        << "    }\n"
        << "    if (reason != nullptr) {\n"
        << "        return 2;\n"
        << "    }\n"
        << "    if (metadata.adapter_name == nullptr || metadata.source_file == nullptr ||\n"
        << "        metadata.target_header == nullptr || metadata.include_guard == nullptr) {\n"
        << "        return 3;\n"
        << "    }\n"
        << "    if (std::strcmp(metadata.adapter_name, \""
        << EscapeJson(spec.adapter_name) << "\") != 0) {\n"
        << "        return 4;\n"
        << "    }\n";
    if (!spec.expected_source_file.empty()) {
        out << "    if (std::strcmp(metadata.source_file, \""
            << EscapeJson(spec.expected_source_file) << "\") != 0) {\n"
            << "        return 5;\n"
            << "    }\n";
    }
    if (!spec.expected_target_header.empty()) {
        out << "    if (std::strcmp(metadata.target_header, \""
            << EscapeJson(spec.expected_target_header) << "\") != 0) {\n"
            << "        return 6;\n"
            << "    }\n";
    }
    out << "    if (std::strcmp(metadata.include_guard, \""
        << EscapeJson(guard) << "\") != 0) {\n"
        << "        return 7;\n"
        << "    }\n"
        << "    if (metadata.known_hook_count <\n"
        << "        tilexr::checker::" << ns << "::kKnownHookCount) {\n"
        << "        return 8;\n"
        << "    }\n"
        << "    if (metadata.manual_review_candidate_count <\n"
        << "        tilexr::checker::" << ns << "::kManualReviewCandidateCount) {\n"
        << "        return 9;\n"
        << "    }\n"
        << "    return 0;\n"
        << "}\n";
    *output = out.str();
    return CheckerStatus::Ok();
}

CheckerStatus WriteTraceHeaderProbeSkeletonFile(const TraceHeaderProbeSpec &spec,
                                                const std::string &path) {
    if (path.empty()) {
        return CheckerStatus::Unsupported("trace header probe output path is empty");
    }
    if (IsProductionSourcePath(path)) {
        return CheckerStatus::Unsupported(
            "trace header probe output must be a checker output path, not a production source path");
    }

    std::string content;
    CheckerStatus status = RenderTraceHeaderProbeSkeleton(spec, &content);
    if (!status.ok()) {
        return status;
    }
    return WriteFile(path, content, "trace header probe");
}

CheckerStatus RenderTraceOnboardingPlan(const TraceAdapterSpec &adapter_spec,
                                        const TraceRunnerSkeletonSpec &runner_spec,
                                        const std::string &shim_header,
                                        const std::string &manifest_path,
                                        const std::string &runner_path,
                                        std::string *output) {
    if (output == nullptr) {
        return CheckerStatus::Fail("trace onboarding output pointer is null");
    }

    TraceAdapterSpec normalized = adapter_spec;
    CheckerStatus normalize_status = NormalizeTraceAdapterSpec(&normalized);
    if (!normalize_status.ok()) {
        return normalize_status;
    }
    if (EmptyOrContainsQuote(shim_header) || EmptyOrContainsQuote(manifest_path) ||
        EmptyOrContainsQuote(runner_path) ||
        !IsIdentifier(runner_spec.function_name) ||
        !IsIdentifier(runner_spec.materializer_name)) {
        return CheckerStatus::Unsupported(
            "trace onboarding fields must be non-empty, quote-free, and use valid identifiers");
    }

    const SourceObservation source_observation =
        ObserveSourceSymbols(normalized.source_scan_file);
    std::ostringstream out;
    out << "# Trace Onboarding: " << normalized.adapter_name << "\n\n"
        << "Production source: `" << normalized.source_file << "`\n"
        << "Target include: `" << normalized.target_header << "`\n"
        << "Shim header: `" << shim_header << "`\n"
        << "Manifest: `" << manifest_path << "`\n"
        << "Runner skeleton: `" << runner_path << "`\n\n"
        << "## Integration Steps\n\n"
        << "1. Move or copy the generated shim under `tools/checker/shim/tilexr/checker/` "
        << "without editing the production header.\n"
        << "2. Add the runner source to `tools/checker/CMakeLists.txt` and declare `"
        << runner_spec.function_name << "` in `tools/checker/include/tilexr/checker/"
        << "collective_trace_runner.h`.\n"
        << "3. Implement the block schedule, production operator construction, and "
        << "`Init`/`Process` calls in `" << runner_path << "`.\n"
        << "4. Review or implement `" << runner_spec.materializer_name
        << "` so the output oracle matches the production operator semantics.\n"
        << "5. Wire the algorithm selection in `CollectiveExecutor` only after "
        << "`RunnerIntegrationChecklist` is fully satisfied.\n"
        << "6. Add an `AlgorithmId` entry, `ToString(AlgorithmId)` text, and "
        << "`ParseAlgorithm` CLI mapping when this adapter introduces a new "
        << "named production algorithm.\n"
        << "7. Move the generated header probe into `tests/checker/unit/`, add it to "
        << "`tests/checker/CMakeLists.txt`, and keep its metadata/audit checks in the "
        << "installed manifest as `probe_test`.\n"
        << "8. Add or update `test_tilexr_checker_*` coverage for selector conditions, "
        << "trace events, oracle output, and source locations.\n\n"
        << "Copy the generated manifest's `installed_trace_seed.required_event_coverage` "
        << "entries into `tools/checker/installed_traces/<adapter>_trace_manifest.json` "
        << "after reviewing the production call sites. Keep only events that must be "
        << "present for the installed runner to be considered correctly traced.\n\n"
        << "Installed manifest skeleton: "
        << "`tools/checker/installed_traces/<adapter>_trace_manifest.json` should name "
        << "`adapter_name`, `source_file`, `shim_path`, `runner_source`, "
        << "`runner_function`, `probe_test`, `probe_cmake_entry`, `algorithm_enum`, "
        << "`materializer_name`, `schedule_name`, `schedule_function`, `op`, "
        << "`algorithm`, representative `smoke_cases`, and "
        << "`required_event_coverage`.\n\n"
        << "";
    RenderManualReviewPlanMarkdown(source_observation, &out);
    out
        << "## Required Verification\n\n"
        << "- `source scripts/common_env.sh && build-checker/tools/checker/"
        << "tilexr_checker --verify-trace-bundle --adapter-name "
        << normalized.adapter_name << " --output-dir <generated-checker-dir> "
        << "--repo-root . --verification-report <generated-checker-dir>/verification.json`\n"
        << "- `source scripts/common_env.sh && build-checker/tools/checker/"
        << "tilexr_checker --verify-installed-trace --adapter-name "
        << normalized.adapter_name << " --repo-root . --verification-report "
        << "/tmp/tilexr-checker-" << normalized.adapter_name
        << "-installed-verification.json`\n"
        << "- `source scripts/common_env.sh && cmake --build build-checker --target "
        << "tilexr_checker_all -j$(nproc)`\n"
        << "- `source scripts/common_env.sh && ctest --test-dir build-checker -R "
        << "checker --output-on-failure`\n"
        << "- `git diff -- src/collectives src/ep`\n\n"
        << "## Source Preservation\n\n"
        << "Do not edit production sources for checker-only symbols. Checker support "
        << "must stay in generated shims, runner code, tests, and checker metadata.\n";
    *output = out.str();
    return CheckerStatus::Ok();
}

CheckerStatus WriteTraceOnboardingPlanFile(const TraceAdapterSpec &adapter_spec,
                                           const TraceRunnerSkeletonSpec &runner_spec,
                                           const std::string &shim_header,
                                           const std::string &manifest_path,
                                           const std::string &runner_path,
                                           const std::string &path) {
    if (path.empty()) {
        return CheckerStatus::Unsupported("trace onboarding output path is empty");
    }
    if (IsProductionSourcePath(path)) {
        return CheckerStatus::Unsupported(
            "trace onboarding output must be a checker output path, not a production source path");
    }

    std::string content;
    CheckerStatus status = RenderTraceOnboardingPlan(adapter_spec, runner_spec, shim_header,
                                                     manifest_path, runner_path, &content);
    if (!status.ok()) {
        return status;
    }
    return WriteFile(path, content, "trace onboarding plan");
}

}  // namespace checker
}  // namespace tilexr
