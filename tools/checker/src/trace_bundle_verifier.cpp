#include "tilexr/checker/trace_bundle_verifier.h"

#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "tilexr/checker/executor.h"

namespace tilexr {
namespace checker {
namespace {

std::string JoinPath(const std::string &dir, const std::string &name) {
    if (dir.empty()) {
        return name;
    }
    if (dir[dir.size() - 1] == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

bool ReadTextFile(const std::string &path, std::string *content) {
    std::ifstream input(path.c_str());
    if (!input.is_open()) {
        return false;
    }
    std::ostringstream out;
    out << input.rdbuf();
    *content = out.str();
    return true;
}

void AddIfMissing(const std::string &content, const std::string &needle,
                  const std::string &section, std::vector<std::string> *missing) {
    if (content.find(needle) == std::string::npos) {
        missing->push_back(section);
    }
}

void AddUncheckedIfFalse(const std::string &content, const std::string &item,
                         std::vector<std::string> *unchecked) {
    const std::string false_pattern = item + " = false";
    if (content.find(false_pattern) != std::string::npos) {
        unchecked->push_back(item);
    }
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

size_t SkipJsonWhitespace(const std::string &content, size_t pos) {
    while (pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[pos])) != 0) {
        ++pos;
    }
    return pos;
}

bool FindJsonValueStart(const std::string &content, const std::string &key,
                        size_t *value_start) {
    if (value_start == nullptr) {
        return false;
    }
    const std::string quoted_key = "\"" + key + "\"";
    const std::string::size_type key_pos = content.find(quoted_key);
    if (key_pos == std::string::npos) {
        return false;
    }
    const std::string::size_type colon = content.find(':', key_pos + quoted_key.size());
    if (colon == std::string::npos) {
        return false;
    }
    *value_start = SkipJsonWhitespace(content, colon + 1);
    return *value_start < content.size();
}

bool ExtractJsonString(const std::string &content, const std::string &key,
                       std::string *value) {
    size_t pos = 0;
    if (value == nullptr || !FindJsonValueStart(content, key, &pos) ||
        content[pos] != '"') {
        return false;
    }
    ++pos;
    std::string parsed;
    while (pos < content.size()) {
        const char ch = content[pos++];
        if (ch == '"') {
            *value = parsed;
            return true;
        }
        if (ch == '\\') {
            if (pos >= content.size()) {
                return false;
            }
            const char escaped = content[pos++];
            switch (escaped) {
                case '"':
                case '\\':
                case '/':
                    parsed += escaped;
                    break;
                case 'n':
                    parsed += '\n';
                    break;
                case 'r':
                    parsed += '\r';
                    break;
                case 't':
                    parsed += '\t';
                    break;
                default:
                    return false;
            }
        } else {
            parsed += ch;
        }
    }
    return false;
}

bool ExtractJsonInt64(const std::string &content, const std::string &key,
                      int64_t *value) {
    size_t pos = 0;
    if (value == nullptr || !FindJsonValueStart(content, key, &pos)) {
        return false;
    }
    const char *begin = content.c_str() + pos;
    char *end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(begin, &end, 10);
    if (errno != 0 || end == begin) {
        return false;
    }
    *value = static_cast<int64_t>(parsed);
    return true;
}

bool ExtractJsonInt(const std::string &content, const std::string &key,
                    int *value) {
    int64_t parsed = 0;
    if (value == nullptr || !ExtractJsonInt64(content, key, &parsed) ||
        parsed < INT_MIN || parsed > INT_MAX) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

std::string RenderSummary(const TraceBundleVerificationResult &result) {
    std::ostringstream out;
    out << "trace bundle verification: "
        << (result.complete ? "complete" : "incomplete");
    if (result.probe_compile_attempted) {
        out << "\nprobe_compile: "
            << (result.probe_compile_passed ? "PASS" : "FAIL");
    }
    if (!result.missing_artifacts.empty()) {
        out << "\nmissing_artifacts:";
        for (size_t i = 0; i < result.missing_artifacts.size(); ++i) {
            out << " " << result.missing_artifacts[i];
        }
    }
    if (!result.missing_manifest_sections.empty()) {
        out << "\nmissing_manifest_sections:";
        for (size_t i = 0; i < result.missing_manifest_sections.size(); ++i) {
            out << " " << result.missing_manifest_sections[i];
        }
    }
    if (!result.unchecked_items.empty()) {
        out << "\nunchecked_items:";
        for (size_t i = 0; i < result.unchecked_items.size(); ++i) {
            out << " " << result.unchecked_items[i];
        }
    }
    return out.str();
}

struct InstalledAlgorithmSpec {
    struct SmokeCase {
        int rank_size = 0;
        int64_t count = 0;
        int server_count = 0;
        int64_t bs = 0;
        int64_t h = 0;
        int64_t top_k = 0;
        int64_t moe_expert_num = 0;
        TileXR::TileXRDataType data_type = TileXR::TILEXR_DATA_TYPE_RESERVED;
    };
    std::string adapter_name;
    std::string manifest_path;
    std::string source_file;
    std::string runner_function;
    std::string shim_path;
    std::string runner_source;
    std::string probe_test;
    std::string probe_cmake_entry;
    std::string algorithm_enum;
    std::string materializer_name;
    std::string schedule_name;
    std::string schedule_function;
    CollectiveOp op = CollectiveOp::kAllGather;
    AlgorithmId algorithm = AlgorithmId::kDefault;
    int rank_size = 0;
    int64_t count = 0;
    int server_count = 0;
    TileXR::TileXRDataType data_type = TileXR::TILEXR_DATA_TYPE_RESERVED;
    std::vector<SmokeCase> smoke_cases;
    struct RequiredEventCoverage {
        EventKind kind = EventKind::kDiagnostic;
        std::string source_file;
        int source_line = 0;
    };
    std::vector<RequiredEventCoverage> required_event_coverage;
};

struct SourceScanHit {
    std::string symbol;
    int line = 0;
    bool manual_review_required = false;
};

struct SourceScanPattern {
    const char *symbol;
    const char *pattern;
    bool manual_review_required;
};

const SourceScanPattern kSourceScanPatterns[] = {
    {"CpGM2GMPingPong", "CpGM2GMPingPong", false},
    {"SetSyncFlag", "SetSyncFlag", false},
    {"WaitSyncFlag", "WaitSyncFlag", false},
    {"AscendC::SetFlag", "AscendC::SetFlag", false},
    {"AscendC::WaitFlag", "AscendC::WaitFlag", false},
    {"AscendC::PipeBarrier", "AscendC::PipeBarrier", false},
    {"PipeBarrier", "PipeBarrier", false},
    {"DataCopy", "DataCopy", true},
    {"DataCopyPad", "DataCopyPad", true},
    {"SetAtomic", "SetAtomic", true},
    {"SetAtomicAdd", "SetAtomicAdd", true},
    {"SetAtomicMax", "SetAtomicMax", true},
    {"SetAtomicMin", "SetAtomicMin", true},
};

std::string InstalledManifestPath(const std::string &adapter_name) {
    return "tools/checker/installed_traces/" + adapter_name +
           "_trace_manifest.json";
}

bool ParseManifestOp(const std::string &text, CollectiveOp *op) {
    if (op == nullptr) {
        return false;
    }
    if (text == "allgather") {
        *op = CollectiveOp::kAllGather;
        return true;
    }
    if (text == "allreduce") {
        *op = CollectiveOp::kAllReduce;
        return true;
    }
    if (text == "ep_dispatch") {
        *op = CollectiveOp::kEpDispatch;
        return true;
    }
    if (text == "ep_combine") {
        *op = CollectiveOp::kEpCombine;
        return true;
    }
    return false;
}

bool IsEpManifestOp(CollectiveOp op) {
    return op == CollectiveOp::kEpDispatch || op == CollectiveOp::kEpCombine;
}

bool IsSourceAlignedOracleOp(CollectiveOp op) {
    return op == CollectiveOp::kEpDispatch;
}

bool ParseManifestDataType(const std::string &text, TileXR::TileXRDataType *data_type) {
    if (data_type == nullptr) {
        return false;
    }
    if (text == "int32") {
        *data_type = TileXR::TILEXR_DATA_TYPE_INT32;
        return true;
    }
    if (text == "fp16") {
        *data_type = TileXR::TILEXR_DATA_TYPE_FP16;
        return true;
    }
    if (text == "bfp16") {
        *data_type = TileXR::TILEXR_DATA_TYPE_BFP16;
        return true;
    }
    return false;
}

bool IsIdentifierChar(char ch) {
    const unsigned char value = static_cast<unsigned char>(ch);
    return std::isalnum(value) || ch == '_';
}

bool IsDigitChar(char ch) {
    return std::isdigit(static_cast<unsigned char>(ch)) != 0;
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

bool ContainsLineNumberToken(const std::string &content, int line) {
    const std::string token = std::to_string(line);
    size_t pos = content.find(token);
    while (pos != std::string::npos) {
        const bool before_ok = pos == 0 || !IsDigitChar(content[pos - 1]);
        const size_t end = pos + token.size();
        const bool after_ok = end >= content.size() || !IsDigitChar(content[end]);
        if (before_ok && after_ok) {
            return true;
        }
        pos = content.find(token, pos + 1);
    }
    return false;
}

void ScanSourceHooks(const std::string &source,
                     std::vector<SourceScanHit> *hits) {
    if (hits == nullptr) {
        return;
    }
    std::istringstream input(source);
    std::string line;
    int line_no = 0;
    bool in_block_comment = false;
    while (std::getline(input, line)) {
        ++line_no;
        const std::string scan_line =
            StripCommentsAndStringsFromLine(line, &in_block_comment);
        for (size_t i = 0; i < sizeof(kSourceScanPatterns) /
                                 sizeof(kSourceScanPatterns[0]); ++i) {
            if (std::string(kSourceScanPatterns[i].symbol) == "PipeBarrier" &&
                ContainsSymbolPattern(scan_line, "AscendC::PipeBarrier")) {
                continue;
            }
            if (std::string(kSourceScanPatterns[i].symbol) == "DataCopy" &&
                ContainsSymbolPattern(scan_line, "DataCopyPad")) {
                continue;
            }
            if (ContainsSymbolPattern(scan_line, kSourceScanPatterns[i].pattern)) {
                SourceScanHit hit;
                hit.symbol = kSourceScanPatterns[i].symbol;
                hit.line = line_no;
                hit.manual_review_required =
                    kSourceScanPatterns[i].manual_review_required;
                hits->push_back(hit);
            }
        }
    }
}

void CheckInstalledSourceScan(const std::string &repo_root,
                              const InstalledAlgorithmSpec &spec,
                              std::vector<std::string> *missing) {
    if (missing == nullptr) {
        return;
    }
    std::string source;
    if (!ReadTextFile(JoinPath(repo_root, spec.source_file), &source)) {
        missing->push_back("source_scan:source_unavailable:" + spec.source_file);
        return;
    }
    std::string shim;
    if (!ReadTextFile(JoinPath(repo_root, spec.shim_path), &shim)) {
        missing->push_back("source_scan:shim_unavailable:" + spec.shim_path);
        return;
    }

    std::vector<SourceScanHit> hits;
    ScanSourceHooks(source, &hits);
    bool found_known_hook = false;
    for (size_t i = 0; i < hits.size(); ++i) {
        const SourceScanHit &hit = hits[i];
        if (hit.manual_review_required) {
            std::ostringstream item;
            item << "source_scan:manual_review:" << hit.symbol << ":" << hit.line;
            missing->push_back(item.str());
            continue;
        }
        found_known_hook = true;
        const bool has_symbol = shim.find("\"" + hit.symbol + "\"") != std::string::npos;
        if (!has_symbol || !ContainsLineNumberToken(shim, hit.line)) {
            std::ostringstream item;
            item << "source_scan:" << hit.symbol << ":" << hit.line;
            missing->push_back(item.str());
        }
    }
    if (!found_known_hook && hits.empty()) {
        missing->push_back("source_scan:coverage_decision:" + spec.source_file);
    }
}

bool ParseManifestEventKind(const std::string &text, EventKind *kind) {
    if (kind == nullptr) {
        return false;
    }
    if (text == "COPY") {
        *kind = EventKind::kCopy;
        return true;
    }
    if (text == "READ") {
        *kind = EventKind::kRead;
        return true;
    }
    if (text == "WRITE") {
        *kind = EventKind::kWrite;
        return true;
    }
    if (text == "FLAG_STORE") {
        *kind = EventKind::kFlagStore;
        return true;
    }
    if (text == "FLAG_WAIT") {
        *kind = EventKind::kFlagWait;
        return true;
    }
    if (text == "PIPE_SET") {
        *kind = EventKind::kPipeSet;
        return true;
    }
    if (text == "PIPE_WAIT") {
        *kind = EventKind::kPipeWait;
        return true;
    }
    if (text == "PIPE_BARRIER") {
        *kind = EventKind::kPipeBarrier;
        return true;
    }
    if (text == "BARRIER") {
        *kind = EventKind::kBarrier;
        return true;
    }
    return false;
}

const char *EventKindManifestName(EventKind kind) {
    switch (kind) {
        case EventKind::kCopy:
            return "COPY";
        case EventKind::kRead:
            return "READ";
        case EventKind::kWrite:
            return "WRITE";
        case EventKind::kFlagStore:
            return "FLAG_STORE";
        case EventKind::kFlagWait:
            return "FLAG_WAIT";
        case EventKind::kPipeSet:
            return "PIPE_SET";
        case EventKind::kPipeWait:
            return "PIPE_WAIT";
        case EventKind::kPipeBarrier:
            return "PIPE_BARRIER";
        case EventKind::kBarrier:
            return "BARRIER";
        case EventKind::kRankStart:
            return "RANK_START";
        case EventKind::kRankEnd:
            return "RANK_END";
        case EventKind::kDiagnostic:
            return "DIAGNOSTIC";
    }
    return "UNKNOWN";
}

bool ParseManifestAlgorithm(const std::string &text, AlgorithmId *algorithm) {
    if (algorithm == nullptr) {
        return false;
    }
    if (text == "default") {
        *algorithm = AlgorithmId::kDefault;
        return true;
    }
    if (text == "allgather_hierarchy_double_ring") {
        *algorithm = AlgorithmId::kAllGatherHierarchyDoubleRing;
        return true;
    }
    if (text == "allreduce_big_data") {
        *algorithm = AlgorithmId::kAllReduceBigData;
        return true;
    }
    return false;
}

void RequireJsonString(const std::string &manifest, const std::string &key,
                       std::string *value,
                       std::vector<std::string> *missing) {
    if (!ExtractJsonString(manifest, key, value)) {
        missing->push_back("manifest:" + key);
    }
}

void RequireJsonInt(const std::string &manifest, const std::string &key,
                    int *value, std::vector<std::string> *missing) {
    if (!ExtractJsonInt(manifest, key, value)) {
        missing->push_back("manifest:" + key);
    }
}

void RequireJsonInt64(const std::string &manifest, const std::string &key,
                      int64_t *value, std::vector<std::string> *missing) {
    if (!ExtractJsonInt64(manifest, key, value)) {
        missing->push_back("manifest:" + key);
    }
}

size_t FindMatchingJsonBracket(const std::string &content, size_t open_pos,
                               char open_ch, char close_ch) {
    if (open_pos >= content.size() || content[open_pos] != open_ch) {
        return std::string::npos;
    }
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t pos = open_pos; pos < content.size(); ++pos) {
        const char ch = content[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch == open_ch) {
            ++depth;
        } else if (ch == close_ch) {
            --depth;
            if (depth == 0) {
                return pos;
            }
        }
    }
    return std::string::npos;
}

void AddPendingManualReviewActions(const std::string &manifest,
                                   std::vector<std::string> *unchecked) {
    if (unchecked == nullptr) {
        return;
    }
    size_t array_pos = 0;
    if (!FindJsonValueStart(manifest, "required_actions", &array_pos) ||
        array_pos >= manifest.size() || manifest[array_pos] != '[') {
        return;
    }
    const size_t array_end = FindMatchingJsonBracket(manifest, array_pos, '[', ']');
    if (array_end == std::string::npos) {
        unchecked->push_back("manual_review:required_actions");
        return;
    }

    size_t pos = array_pos + 1;
    while (pos < array_end) {
        pos = SkipJsonWhitespace(manifest, pos);
        if (pos >= array_end) {
            break;
        }
        if (manifest[pos] == ',') {
            ++pos;
            continue;
        }
        if (manifest[pos] != '{') {
            unchecked->push_back("manual_review:required_actions");
            return;
        }
        const size_t object_end = FindMatchingJsonBracket(manifest, pos, '{', '}');
        if (object_end == std::string::npos || object_end > array_end) {
            unchecked->push_back("manual_review:required_actions");
            return;
        }
        const std::string object = manifest.substr(pos, object_end - pos + 1);
        std::string status;
        if (!ExtractJsonString(object, "status", &status)) {
            unchecked->push_back("manual_review:missing_status");
        } else if (status != "handled" && status != "waived" &&
                   status != "unsupported_gate") {
            std::string symbol;
            int line = 0;
            if (!ExtractJsonString(object, "symbol", &symbol)) {
                symbol = "unknown";
            }
            if (!ExtractJsonInt(object, "line", &line)) {
                line = 0;
            }
            unchecked->push_back("manual_review:" + symbol + ":" +
                                 std::to_string(line));
        }
        pos = object_end + 1;
    }
}

void ParseRequiredEventCoverage(const std::string &manifest,
                                std::vector<InstalledAlgorithmSpec::RequiredEventCoverage> *coverage,
                                std::vector<std::string> *missing) {
    if (coverage == nullptr || missing == nullptr) {
        return;
    }
    size_t array_pos = 0;
    if (!FindJsonValueStart(manifest, "required_event_coverage", &array_pos)) {
        missing->push_back("manifest:required_event_coverage");
        return;
    }
    if (array_pos >= manifest.size() || manifest[array_pos] != '[') {
        missing->push_back("manifest:required_event_coverage");
        return;
    }
    const size_t array_end = FindMatchingJsonBracket(manifest, array_pos, '[', ']');
    if (array_end == std::string::npos) {
        missing->push_back("manifest:required_event_coverage");
        return;
    }

    size_t pos = array_pos + 1;
    int entry_index = 0;
    while (pos < array_end) {
        pos = SkipJsonWhitespace(manifest, pos);
        if (pos >= array_end) {
            break;
        }
        if (manifest[pos] == ',') {
            ++pos;
            continue;
        }
        if (manifest[pos] != '{') {
            missing->push_back("manifest:required_event_coverage");
            return;
        }
        const size_t object_end = FindMatchingJsonBracket(manifest, pos, '{', '}');
        if (object_end == std::string::npos || object_end > array_end) {
            missing->push_back("manifest:required_event_coverage");
            return;
        }
        const std::string object = manifest.substr(pos, object_end - pos + 1);
        InstalledAlgorithmSpec::RequiredEventCoverage item;
        std::string kind;
        bool ok = true;
        if (!ExtractJsonString(object, "kind", &kind) ||
            !ParseManifestEventKind(kind, &item.kind)) {
            missing->push_back("manifest:required_event_coverage[" +
                               std::to_string(entry_index) + "].kind");
            ok = false;
        }
        if (!ExtractJsonString(object, "source_file", &item.source_file)) {
            missing->push_back("manifest:required_event_coverage[" +
                               std::to_string(entry_index) + "].source_file");
            ok = false;
        }
        if (!ExtractJsonInt(object, "source_line", &item.source_line)) {
            missing->push_back("manifest:required_event_coverage[" +
                               std::to_string(entry_index) + "].source_line");
            ok = false;
        }
        if (ok) {
            coverage->push_back(item);
        }
        pos = object_end + 1;
        ++entry_index;
    }
    if (entry_index == 0) {
        missing->push_back("manifest:required_event_coverage");
    }
}

void ParseSmokeCases(const std::string &manifest,
                     CollectiveOp op,
                     TileXR::TileXRDataType default_data_type,
                     std::vector<InstalledAlgorithmSpec::SmokeCase> *smoke_cases,
                     std::vector<std::string> *missing) {
    if (smoke_cases == nullptr || missing == nullptr) {
        return;
    }
    size_t array_pos = 0;
    if (!FindJsonValueStart(manifest, "smoke_cases", &array_pos)) {
        return;
    }
    if (array_pos >= manifest.size() || manifest[array_pos] != '[') {
        missing->push_back("manifest:smoke_cases");
        return;
    }
    const size_t array_end = FindMatchingJsonBracket(manifest, array_pos, '[', ']');
    if (array_end == std::string::npos) {
        missing->push_back("manifest:smoke_cases");
        return;
    }

    size_t pos = array_pos + 1;
    int entry_index = 0;
    while (pos < array_end) {
        pos = SkipJsonWhitespace(manifest, pos);
        if (pos >= array_end) {
            break;
        }
        if (manifest[pos] == ',') {
            ++pos;
            continue;
        }
        if (manifest[pos] != '{') {
            missing->push_back("manifest:smoke_cases");
            return;
        }
        const size_t object_end = FindMatchingJsonBracket(manifest, pos, '{', '}');
        if (object_end == std::string::npos || object_end > array_end) {
            missing->push_back("manifest:smoke_cases");
            return;
        }
        const std::string object = manifest.substr(pos, object_end - pos + 1);
        InstalledAlgorithmSpec::SmokeCase item;
        bool ok = true;
        if (!ExtractJsonInt(object, "rank_size", &item.rank_size)) {
            missing->push_back("manifest:smoke_cases[" +
                               std::to_string(entry_index) + "].rank_size");
            ok = false;
        }
        if (!ExtractJsonInt(object, "server_count", &item.server_count)) {
            missing->push_back("manifest:smoke_cases[" +
                               std::to_string(entry_index) + "].server_count");
            ok = false;
        }
        if (IsEpManifestOp(op)) {
            if (!ExtractJsonInt64(object, "bs", &item.bs)) {
                missing->push_back("manifest:smoke_cases[" +
                                   std::to_string(entry_index) + "].bs");
                ok = false;
            }
            if (!ExtractJsonInt64(object, "h", &item.h)) {
                missing->push_back("manifest:smoke_cases[" +
                                   std::to_string(entry_index) + "].h");
                ok = false;
            }
            if (!ExtractJsonInt64(object, "top_k", &item.top_k)) {
                missing->push_back("manifest:smoke_cases[" +
                                   std::to_string(entry_index) + "].top_k");
                ok = false;
            }
            if (!ExtractJsonInt64(object, "moe_expert_num", &item.moe_expert_num)) {
                missing->push_back("manifest:smoke_cases[" +
                                   std::to_string(entry_index) + "].moe_expert_num");
                ok = false;
            }
            std::string data_type;
            if (ExtractJsonString(object, "datatype", &data_type)) {
                if (!ParseManifestDataType(data_type, &item.data_type)) {
                    missing->push_back("manifest:smoke_cases[" +
                                       std::to_string(entry_index) + "].datatype");
                    ok = false;
                }
            } else {
                item.data_type = default_data_type;
                if (item.data_type == TileXR::TILEXR_DATA_TYPE_RESERVED) {
                    missing->push_back("manifest:smoke_cases[" +
                                       std::to_string(entry_index) + "].datatype");
                    ok = false;
                }
            }
        } else {
            if (!ExtractJsonInt64(object, "count", &item.count)) {
                missing->push_back("manifest:smoke_cases[" +
                                   std::to_string(entry_index) + "].count");
                ok = false;
            }
            item.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
        }
        if (ok) {
            smoke_cases->push_back(item);
        }
        pos = object_end + 1;
        ++entry_index;
    }
    if (entry_index == 0) {
        missing->push_back("manifest:smoke_cases");
    }
}

CheckerStatus LoadInstalledAlgorithmSpec(const std::string &repo_root,
                                         const std::string &adapter_name,
                                         InstalledAlgorithmSpec *spec,
                                         std::vector<std::string> *missing) {
    if (spec == nullptr || missing == nullptr) {
        return CheckerStatus::Fail("installed trace manifest output pointer is null");
    }
    *spec = InstalledAlgorithmSpec();
    spec->adapter_name = adapter_name;
    spec->manifest_path = InstalledManifestPath(adapter_name);

    std::string manifest;
    if (!ReadTextFile(JoinPath(repo_root, spec->manifest_path), &manifest)) {
        missing->push_back(spec->manifest_path);
        return CheckerStatus::Unsupported("installed trace manifest is missing: " +
                                          spec->manifest_path);
    }

    RequireJsonString(manifest, "adapter_name", &spec->adapter_name, missing);
    RequireJsonString(manifest, "source_file", &spec->source_file, missing);
    RequireJsonString(manifest, "runner_function", &spec->runner_function, missing);
    RequireJsonString(manifest, "shim_path", &spec->shim_path, missing);
    RequireJsonString(manifest, "runner_source", &spec->runner_source, missing);
    RequireJsonString(manifest, "probe_test", &spec->probe_test, missing);
    RequireJsonString(manifest, "probe_cmake_entry", &spec->probe_cmake_entry, missing);
    RequireJsonString(manifest, "algorithm_enum", &spec->algorithm_enum, missing);
    RequireJsonString(manifest, "materializer_name", &spec->materializer_name, missing);
    RequireJsonString(manifest, "schedule_name", &spec->schedule_name, missing);
    RequireJsonString(manifest, "schedule_function", &spec->schedule_function, missing);
    std::string op;
    if (!ExtractJsonString(manifest, "op", &op) ||
        !ParseManifestOp(op, &spec->op)) {
        missing->push_back("manifest:op");
    }
    std::string algorithm;
    if (!ExtractJsonString(manifest, "algorithm", &algorithm) ||
        !ParseManifestAlgorithm(algorithm, &spec->algorithm)) {
        missing->push_back("manifest:algorithm");
    }
    std::string data_type;
    if (ExtractJsonString(manifest, "datatype", &data_type) &&
        !ParseManifestDataType(data_type, &spec->data_type)) {
        missing->push_back("manifest:datatype");
    }
    RequireJsonInt(manifest, "rank_size", &spec->rank_size, missing);
    if (!IsEpManifestOp(spec->op)) {
        RequireJsonInt64(manifest, "count", &spec->count, missing);
    }
    RequireJsonInt(manifest, "server_count", &spec->server_count, missing);
    ParseSmokeCases(manifest, spec->op, spec->data_type, &spec->smoke_cases, missing);
    if (!IsEpManifestOp(spec->op) && spec->smoke_cases.empty() &&
        spec->rank_size > 0 && spec->count > 0 && spec->server_count > 0) {
        InstalledAlgorithmSpec::SmokeCase fallback;
        fallback.rank_size = spec->rank_size;
        fallback.count = spec->count;
        fallback.server_count = spec->server_count;
        fallback.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
        spec->smoke_cases.push_back(fallback);
    }
    ParseRequiredEventCoverage(manifest, &spec->required_event_coverage, missing);
    if (spec->adapter_name != adapter_name) {
        missing->push_back("manifest:adapter_name:" + spec->adapter_name);
    }
    if (!missing->empty()) {
        return CheckerStatus::Unsupported("installed trace manifest is incomplete: " +
                                          spec->manifest_path);
    }
    return CheckerStatus::Ok();
}

void CheckRequiredEventCoverage(const EventLog &events,
                                const InstalledAlgorithmSpec &spec,
                                std::vector<std::string> *missing) {
    if (missing == nullptr) {
        return;
    }
    const std::vector<Event> &items = events.events();
    for (size_t i = 0; i < spec.required_event_coverage.size(); ++i) {
        const InstalledAlgorithmSpec::RequiredEventCoverage &required =
            spec.required_event_coverage[i];
        bool found = false;
        for (size_t j = 0; j < items.size(); ++j) {
            if (items[j].kind == required.kind &&
                items[j].source_file == required.source_file &&
                items[j].source_line == required.source_line) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::ostringstream item;
            item << "event_coverage:" << EventKindManifestName(required.kind)
                 << ":" << required.source_file << ":" << required.source_line;
            missing->push_back(item.str());
        }
    }
}

void RequireFileWithText(const std::string &repo_root, const std::string &path,
                         const std::string &needle, const std::string &missing_name,
                         std::vector<std::string> *missing) {
    std::string content;
    if (!ReadTextFile(JoinPath(repo_root, path), &content)) {
        missing->push_back(path);
        return;
    }
    if (!needle.empty() && content.find(needle) == std::string::npos) {
        missing->push_back(missing_name);
    }
}

bool IsSafeShellPath(const std::string &path) {
    if (path.empty()) {
        return false;
    }
    for (size_t i = 0; i < path.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(path[i]);
        if (ch < 32 || ch == '\'' || ch == '\n' || ch == '\r') {
            return false;
        }
    }
    return true;
}

std::string ShellQuotePath(const std::string &path) {
    return "'" + path + "'";
}

CheckerStatus RunTraceProbeCompile(const std::string &bundle_dir,
                                   const std::string &adapter_name,
                                   const std::string &repo_root,
                                   TraceBundleVerificationResult *result) {
    if (result == nullptr) {
        return CheckerStatus::Fail("trace bundle verification result pointer is null");
    }
    result->probe_compile_attempted = true;
    result->probe_compile_passed = false;

    const std::string probe_path =
        JoinPath(bundle_dir, adapter_name + "_trace_probe.cpp");
    const std::string probe_binary =
        JoinPath(bundle_dir, adapter_name + "_trace_probe.bin");
    const std::string probe_log =
        JoinPath(bundle_dir, adapter_name + "_trace_probe_compile.log");

    const std::string paths[] = {
        repo_root,
        bundle_dir,
        probe_path,
        probe_binary,
        probe_log,
        JoinPath(repo_root, "tools/checker/shim"),
        JoinPath(repo_root, "tools/checker/include"),
        JoinPath(repo_root, "src/include"),
        JoinPath(repo_root, "src/collectives/kernels"),
    };
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i) {
        if (!IsSafeShellPath(paths[i])) {
            result->missing_artifacts.push_back("probe_compile:unsafe_path");
            return CheckerStatus::Unsupported("trace probe compile path is unsafe");
        }
    }

    std::ostringstream cmd;
    cmd << "c++ -std=c++17"
        << " -I" << ShellQuotePath(bundle_dir)
        << " -I" << ShellQuotePath(JoinPath(repo_root, "tools/checker/shim"))
        << " -I" << ShellQuotePath(JoinPath(repo_root, "tools/checker/include"))
        << " -I" << ShellQuotePath(JoinPath(repo_root, "src/include"))
        << " -I" << ShellQuotePath(JoinPath(repo_root, "src/collectives/kernels"))
        << " " << ShellQuotePath(probe_path)
        << " -o " << ShellQuotePath(probe_binary)
        << " >" << ShellQuotePath(probe_log) << " 2>&1"
        << " && " << ShellQuotePath(probe_binary)
        << " >>" << ShellQuotePath(probe_log) << " 2>&1";

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        result->missing_artifacts.push_back("probe_compile:" + probe_log);
        return CheckerStatus::Unsupported("trace probe compile/run failed");
    }

    result->probe_compile_passed = true;
    unlink(probe_binary.c_str());
    unlink(probe_log.c_str());
    return CheckerStatus::Ok();
}

CheckerStatus RunInstalledProbeCompile(const InstalledAlgorithmSpec &spec,
                                       const std::string &repo_root,
                                       TraceBundleVerificationResult *result) {
    if (result == nullptr) {
        return CheckerStatus::Fail("installed trace verification result pointer is null");
    }
    result->probe_compile_attempted = true;
    result->probe_compile_passed = false;

    const std::string probe_path = JoinPath(repo_root, spec.probe_test);
    char binary_template[] = "/tmp/tilexr-installed-probe-XXXXXX";
    const int fd = mkstemp(binary_template);
    if (fd < 0) {
        result->missing_artifacts.push_back("probe_compile:mkstemp");
        return CheckerStatus::Fail("failed to create installed probe binary path");
    }
    close(fd);
    unlink(binary_template);
    const std::string probe_binary = binary_template;
    const std::string probe_log = probe_binary + ".log";

    const std::string paths[] = {
        repo_root,
        probe_path,
        probe_binary,
        probe_log,
        JoinPath(repo_root, "tools/checker/shim"),
        JoinPath(repo_root, "tools/checker/include"),
        JoinPath(repo_root, "src/include"),
        JoinPath(repo_root, "src/collectives/kernels"),
        JoinPath(repo_root, "src/collectives/host"),
    };
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i) {
        if (!IsSafeShellPath(paths[i])) {
            result->missing_artifacts.push_back("probe_compile:unsafe_path");
            return CheckerStatus::Unsupported("installed probe compile path is unsafe");
        }
    }

    std::ostringstream cmd;
    cmd << "c++ -std=c++17"
        << " -I" << ShellQuotePath(JoinPath(repo_root, "tools/checker/shim"))
        << " -I" << ShellQuotePath(JoinPath(repo_root, "tools/checker/include"))
        << " -I" << ShellQuotePath(JoinPath(repo_root, "src/include"))
        << " -I" << ShellQuotePath(JoinPath(repo_root, "src/collectives/kernels"))
        << " -I" << ShellQuotePath(JoinPath(repo_root, "src/collectives/host"))
        << " " << ShellQuotePath(probe_path)
        << " -o " << ShellQuotePath(probe_binary)
        << " >" << ShellQuotePath(probe_log) << " 2>&1"
        << " && " << ShellQuotePath(probe_binary)
        << " >>" << ShellQuotePath(probe_log) << " 2>&1";

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        result->missing_artifacts.push_back("probe_compile:" + probe_log);
        return CheckerStatus::Unsupported("installed trace probe compile/run failed");
    }

    result->probe_compile_passed = true;
    unlink(probe_binary.c_str());
    unlink(probe_log.c_str());
    return CheckerStatus::Ok();
}

std::string RenderInstalledSummary(const TraceBundleVerificationResult &result,
                                   const InstalledAlgorithmSpec &spec) {
    std::ostringstream out;
    out << "installed algorithm verification: "
        << (result.complete ? "complete" : "incomplete")
        << "\nalgorithm: " << spec.adapter_name
        << "\nmanifest: " << spec.manifest_path
        << "\nrunner: " << spec.runner_function
        << "\nmaterializer: " << spec.materializer_name
        << "\nschedule: " << spec.schedule_name
        << "\nexecutor: CollectiveExecutor"
        << "\nprobe: " << spec.probe_test
        << "\nprobe_compile: "
        << (result.probe_compile_passed ? "PASS" : "FAIL")
        << "\nruntime_smoke: " << (result.runtime_smoke_passed ? "PASS" : "FAIL")
        << "\nruntime_smoke_cases: " << result.runtime_smoke_case_count
        << "\nruntime_events: " << result.runtime_smoke_event_count;
    if (!result.missing_artifacts.empty()) {
        out << "\nmissing_or_incomplete:";
        for (size_t i = 0; i < result.missing_artifacts.size(); ++i) {
            out << " " << result.missing_artifacts[i];
        }
    }
    return out.str();
}

CheckerStatus RunInstalledSmoke(const InstalledAlgorithmSpec &spec,
                                TraceBundleVerificationResult *result) {
    EventLog combined_events;
    result->runtime_smoke_case_count = 0;
    result->runtime_smoke_event_count = 0;
    result->runtime_smoke_passed = true;

    for (size_t i = 0; i < spec.smoke_cases.size(); ++i) {
        const InstalledAlgorithmSpec::SmokeCase &smoke_case = spec.smoke_cases[i];
        CheckerCase test_case;
        test_case.op = spec.op;
        test_case.algorithm = spec.algorithm;
        test_case.rank_size = smoke_case.rank_size;
        test_case.server_count = smoke_case.server_count;
        test_case.count = smoke_case.count;
        test_case.bs = smoke_case.bs;
        test_case.h = smoke_case.h;
        test_case.top_k = smoke_case.top_k;
        test_case.moe_expert_num = smoke_case.moe_expert_num;
        test_case.data_type = smoke_case.data_type == TileXR::TILEXR_DATA_TYPE_RESERVED
            ? TileXR::TILEXR_DATA_TYPE_INT32
            : smoke_case.data_type;
        test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
        test_case.scheduler = SchedulerMode::kRoundRobin;
        test_case.magic = 0xC001CA5EULL + i;

        CheckerStatus case_status = ValidateCase(test_case);
        if (!case_status.ok()) {
            result->runtime_smoke_passed = false;
            result->missing_artifacts.push_back(std::string("runtime_smoke_case:") +
                                                case_status.message);
            return case_status;
        }

        size_t input_bytes = 0;
        size_t output_bytes = 0;
        size_t comm_data_bytes = 0;
        if (IsEpManifestOp(test_case.op)) {
            const size_t element_size = ElementSize(test_case.data_type);
            input_bytes = static_cast<size_t>(test_case.bs * test_case.h) * element_size;
            output_bytes = EpDispatchOutputBytes(test_case);
            comm_data_bytes = output_bytes;
        } else {
            input_bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
            const size_t output_elements =
                test_case.op == CollectiveOp::kAllGather
                    ? static_cast<size_t>(test_case.rank_size) *
                          static_cast<size_t>(test_case.count)
                    : static_cast<size_t>(test_case.count);
            output_bytes = output_elements * sizeof(int32_t);
            comm_data_bytes = input_bytes;
        }
        RankWorld world = RankWorld::Create(test_case.rank_size, input_bytes, output_bytes,
                                            comm_data_bytes);
        world.ConfigureServers(test_case.server_count);

        CollectiveExecutor executor;
        RunResult run = executor.Run(&world, test_case);
        result->runtime_smoke_event_count += run.event_count;
        ++result->runtime_smoke_case_count;

        const bool passed = run.status.ok() && run.findings.findings().empty() &&
                            run.mismatches.empty() && run.event_count > 0;
        if (!passed) {
            result->runtime_smoke_passed = false;
            std::ostringstream reason;
            reason << "runtime_smoke_case[" << i << "]:" << run.status.message
                   << ":events=" << run.event_count
                   << ":findings=" << run.findings.findings().size()
                   << ":mismatches=" << run.mismatches.size();
            result->missing_artifacts.push_back(reason.str());
            return CheckerStatus::Unsupported("installed trace runtime smoke failed");
        }

        const std::vector<Event> &events = world.events().events();
        for (size_t event_index = 0; event_index < events.size(); ++event_index) {
            combined_events.Add(events[event_index]);
        }
    }

    CheckRequiredEventCoverage(combined_events, spec, &result->missing_artifacts);
    if (!result->missing_artifacts.empty()) {
        return CheckerStatus::Unsupported("installed trace event coverage failed");
    }
    return CheckerStatus::Ok();
}

}  // namespace

CheckerStatus VerifyTraceBundle(const std::string &bundle_dir,
                                const std::string &adapter_name,
                                TraceBundleVerificationResult *result) {
    if (result == nullptr) {
        return CheckerStatus::Fail("trace bundle verification result pointer is null");
    }
    *result = TraceBundleVerificationResult();
    if (bundle_dir.empty() || adapter_name.empty()) {
        return CheckerStatus::Unsupported("trace bundle directory and adapter name are required");
    }

    const std::string shim_name = adapter_name + "_trace_shim.h";
    const std::string manifest_name = adapter_name + "_trace_manifest.json";
    const std::string runner_name = adapter_name + "_trace_runner.cpp";
    const std::string probe_name = adapter_name + "_trace_probe.cpp";
    const std::string onboarding_name = adapter_name + "_trace_onboarding.md";

    std::string shim;
    std::string manifest;
    std::string runner;
    std::string probe;
    std::string onboarding;
    if (!ReadTextFile(JoinPath(bundle_dir, shim_name), &shim)) {
        result->missing_artifacts.push_back(shim_name);
    }
    if (!ReadTextFile(JoinPath(bundle_dir, manifest_name), &manifest)) {
        result->missing_artifacts.push_back(manifest_name);
    }
    if (!ReadTextFile(JoinPath(bundle_dir, runner_name), &runner)) {
        result->missing_artifacts.push_back(runner_name);
    }
    if (!ReadTextFile(JoinPath(bundle_dir, probe_name), &probe)) {
        result->missing_artifacts.push_back(probe_name);
    }
    if (!ReadTextFile(JoinPath(bundle_dir, onboarding_name), &onboarding)) {
        result->missing_artifacts.push_back(onboarding_name);
    }

    if (!manifest.empty()) {
        AddIfMissing(manifest, "\"source_preservation\"", "source_preservation",
                     &result->missing_manifest_sections);
        AddIfMissing(manifest, "\"runner_integration\"", "runner_integration",
                     &result->missing_manifest_sections);
        AddIfMissing(manifest, "\"checklist\"", "checklist",
                     &result->missing_manifest_sections);
        AddIfMissing(manifest, "\"metadata_api\"", "metadata_api",
                     &result->missing_manifest_sections);
        if (manifest.find("\"generated_outputs\"") != std::string::npos) {
            AddIfMissing(manifest, "\"semi_automatic_trace\"",
                         "semi_automatic_trace",
                         &result->missing_manifest_sections);
            AddIfMissing(manifest, "\"header_stub\"", "semi_automatic_trace:header_stub",
                         &result->missing_manifest_sections);
            AddIfMissing(manifest, "\"auto_hook_candidates\"",
                         "semi_automatic_trace:auto_hook_candidates",
                         &result->missing_manifest_sections);
            AddIfMissing(manifest, "\"manual_review_actions\"",
                         "semi_automatic_trace:manual_review_actions",
                         &result->missing_manifest_sections);
        }
        AddPendingManualReviewActions(manifest, &result->unchecked_items);
    }
    if (!shim.empty()) {
        AddIfMissing(shim, "TraceAdapterMetadata", "shim:TraceAdapterMetadata",
                     &result->missing_artifacts);
        AddIfMissing(shim, "Metadata()", "shim:Metadata",
                     &result->missing_artifacts);
        AddIfMissing(shim, "Audit(const char **reason)", "shim:Audit",
                     &result->missing_artifacts);
    }
    if (!runner.empty()) {
        AddUncheckedIfFalse(runner, "block_schedule_defined", &result->unchecked_items);
        AddUncheckedIfFalse(runner, "operator_init_process_wired", &result->unchecked_items);
        AddUncheckedIfFalse(runner, "oracle_materializer_reviewed", &result->unchecked_items);
    }
    if (!probe.empty()) {
        const std::string adapter_ns = AdapterNamespace(adapter_name);
        AddIfMissing(probe, adapter_ns + "::Metadata()", "probe:Metadata",
                     &result->missing_artifacts);
        AddIfMissing(probe, adapter_ns + "::Audit(&reason)", "probe:Audit",
                     &result->missing_artifacts);
    }

    result->complete = result->missing_artifacts.empty() &&
                       result->missing_manifest_sections.empty() &&
                       result->unchecked_items.empty();
    result->summary = RenderSummary(*result);
    if (!result->complete) {
        return CheckerStatus::Unsupported("trace bundle has incomplete checker-side integration");
    }
    return CheckerStatus::Ok();
}

CheckerStatus VerifyTraceBundle(const std::string &bundle_dir,
                                const std::string &adapter_name,
                                const std::string &repo_root,
                                TraceBundleVerificationResult *result) {
    CheckerStatus static_status = VerifyTraceBundle(bundle_dir, adapter_name, result);
    if (repo_root.empty()) {
        result->missing_artifacts.push_back("probe_compile:repo_root");
        result->complete = false;
        result->summary = RenderSummary(*result);
        return CheckerStatus::Unsupported("trace bundle repo root is required for probe compile");
    }

    CheckerStatus probe_status = CheckerStatus::Ok();
    if (result->missing_artifacts.empty()) {
        probe_status = RunTraceProbeCompile(bundle_dir, adapter_name, repo_root, result);
    }
    result->complete = static_status.ok() &&
                       probe_status.ok() &&
                       result->missing_artifacts.empty() &&
                       result->missing_manifest_sections.empty() &&
                       result->unchecked_items.empty();
    result->summary = RenderSummary(*result);
    if (!result->complete) {
        if (!probe_status.ok()) {
            return probe_status;
        }
        return static_status.ok()
                   ? CheckerStatus::Unsupported(
                         "trace bundle has incomplete checker-side integration")
                   : static_status;
    }
    return CheckerStatus::Ok();
}

CheckerStatus VerifyInstalledTraceAlgorithm(const std::string &repo_root,
                                            const std::string &adapter_name,
                                            TraceBundleVerificationResult *result) {
    if (result == nullptr) {
        return CheckerStatus::Fail("installed trace algorithm result pointer is null");
    }
    *result = TraceBundleVerificationResult();
    if (repo_root.empty() || adapter_name.empty()) {
        return CheckerStatus::Unsupported("repo root and adapter name are required");
    }

    InstalledAlgorithmSpec spec;
    CheckerStatus manifest_status =
        LoadInstalledAlgorithmSpec(repo_root, adapter_name, &spec,
                                   &result->missing_artifacts);

    if (manifest_status.ok()) {
        if (IsSourceAlignedOracleOp(spec.op)) {
            RequireFileWithText(repo_root, spec.source_file, std::string(),
                                spec.source_file, &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.runner_source, spec.runner_function,
                                spec.runner_source + ":" + spec.runner_function,
                                &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.runner_source, spec.materializer_name,
                                spec.runner_source + ":materializer:" +
                                    spec.materializer_name,
                                &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.runner_source, spec.schedule_function,
                                spec.runner_source + ":schedule:" +
                                    spec.schedule_function,
                                &result->missing_artifacts);
        } else {
            RequireFileWithText(repo_root, spec.shim_path, "collective_trace_adapter.h",
                                spec.shim_path, &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.shim_path, "TraceAdapterMetadata",
                                "shim:TraceAdapterMetadata", &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.shim_path, "Metadata()",
                                "shim:Metadata", &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.shim_path, "Audit(const char **reason)",
                                "shim:Audit", &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.shim_path, spec.source_file,
                                "shim:source_file:" + spec.source_file,
                                &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.runner_source, spec.runner_function,
                                spec.runner_source + ":" + spec.runner_function,
                                &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.runner_source,
                                std::string("ApplyMaterializer(\"") +
                                    spec.materializer_name + "\")",
                                spec.runner_source + ":materializer:" +
                                    spec.materializer_name,
                                &result->missing_artifacts);
            RequireFileWithText(repo_root, spec.runner_source, spec.schedule_function,
                                spec.runner_source + ":schedule:" +
                                    spec.schedule_function,
                                &result->missing_artifacts);
            CheckInstalledSourceScan(repo_root, spec, &result->missing_artifacts);
        }
        RequireFileWithText(repo_root,
                            "tools/checker/include/tilexr/checker/collective_trace_runner.h",
                            spec.runner_function, "collective_trace_runner.h:" +
                            spec.runner_function, &result->missing_artifacts);
        RequireFileWithText(repo_root, "tools/checker/src/executor.cpp",
                            spec.algorithm_enum,
                            std::string("CollectiveExecutor:") + spec.algorithm_enum,
                            &result->missing_artifacts);
        RequireFileWithText(repo_root, spec.probe_test, "static_assert",
                            spec.probe_test, &result->missing_artifacts);
        RequireFileWithText(repo_root, "tests/checker/CMakeLists.txt",
                            spec.probe_cmake_entry,
                            std::string("tests/checker/CMakeLists.txt:") +
                                spec.probe_cmake_entry,
                            &result->missing_artifacts);
    }

    if (result->missing_artifacts.empty() && manifest_status.ok()) {
        (void)RunInstalledProbeCompile(spec, repo_root, result);
    }

    if (result->missing_artifacts.empty() && manifest_status.ok() &&
        result->probe_compile_passed) {
        (void)RunInstalledSmoke(spec, result);
    }

    result->complete = result->missing_artifacts.empty() &&
                       result->probe_compile_passed &&
                       result->runtime_smoke_passed;
    result->summary = RenderInstalledSummary(*result, spec);
    if (!result->complete) {
        if (!manifest_status.ok()) {
            return manifest_status;
        }
        return CheckerStatus::Unsupported("installed trace algorithm is incomplete");
    }
    return CheckerStatus::Ok();
}

}  // namespace checker
}  // namespace tilexr
