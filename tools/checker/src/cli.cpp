#include "tilexr/checker/cli.h"

#include <cerrno>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "tilexr/checker/diagnostics.h"
#include "tilexr/checker/oracle.h"
#include "tilexr/checker/report.h"
#include "tilexr/checker/trace_adapter_generator.h"
#include "tilexr/checker/trace_bundle_verifier.h"
#include "tilexr/checker/world.h"

namespace tilexr {
namespace checker {

namespace {

const char *StatusLabel(const CheckerStatusCode code) {
    switch (code) {
        case CheckerStatusCode::kOk:
            return "PASS";
        case CheckerStatusCode::kUnsupported:
            return "UNSUPPORTED";
        case CheckerStatusCode::kFail:
            return "FAIL";
        case CheckerStatusCode::kInconclusive:
            return "INCONCLUSIVE";
        case CheckerStatusCode::kInternalError:
            return "ERROR";
    }
    return "ERROR";
}

const char *EventKindName(EventKind kind) {
    switch (kind) {
        case EventKind::kRankStart:
            return "RANK_START";
        case EventKind::kRankEnd:
            return "RANK_END";
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
        case EventKind::kDiagnostic:
            return "DIAGNOSTIC";
    }
    return "UNKNOWN";
}

std::string BufferReportName(BufferRole role) {
    switch (role) {
        case BufferRole::kUserInput:
            return "UserInput";
        case BufferRole::kUserOutput:
            return "UserOutput";
        case BufferRole::kCommFlag:
            return "CommFlag";
        case BufferRole::kCommData:
            return "CommData";
        case BufferRole::kLocalUb:
            return "LocalUb";
        case BufferRole::kMetadata:
            return "Metadata";
        case BufferRole::kRegisteredCommBuffer:
            return "RegisteredCommBuffer";
    }
    return "Unknown";
}

const char *ServerScopeName(int server, int peer_server) {
    if (server < 0 || peer_server < 0) {
        return "unknown";
    }
    return server == peer_server ? "same_server" : "cross_server";
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

std::string JoinPath(const std::string &dir, const std::string &name) {
    if (dir.empty()) {
        return name;
    }
    if (dir[dir.size() - 1] == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

std::string BaseName(const std::string &path) {
    const std::string::size_type pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

std::string ParentDir(const std::string &path) {
    const std::string::size_type pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string PascalWord(const std::string &word) {
    if (word == "allgather") {
        return "AllGather";
    }
    if (word == "allreduce") {
        return "AllReduce";
    }
    if (word == "alltoall") {
        return "AllToAll";
    }
    if (word == "ep") {
        return "Ep";
    }
    std::string out;
    bool capitalize = true;
    for (size_t i = 0; i < word.size(); ++i) {
        const char ch = word[i];
        if (ch == '_' || ch == '-' || ch == '.') {
            capitalize = true;
            continue;
        }
        if (capitalize && ch >= 'a' && ch <= 'z') {
            out += static_cast<char>(ch - 'a' + 'A');
        } else {
            out += ch;
        }
        capitalize = false;
    }
    return out;
}

std::string PascalName(const std::string &name) {
    std::string out;
    std::string word;
    for (size_t i = 0; i <= name.size(); ++i) {
        const char ch = i < name.size() ? name[i] : '_';
        if (ch == '_' || ch == '-' || ch == '.') {
            if (!word.empty()) {
                out += PascalWord(word);
                word.clear();
            }
        } else {
            word += ch;
        }
    }
    return out;
}

CheckerStatus ParseInt(const char *text, int *value) {
    if (text == nullptr || value == nullptr) {
        return CheckerStatus::Unsupported("missing integer argument");
    }
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return CheckerStatus::Unsupported(std::string("invalid integer: ") + text);
    }
    if (parsed < INT_MIN || parsed > INT_MAX) {
        return CheckerStatus::Unsupported(std::string("integer out of range: ") + text);
    }
    *value = static_cast<int>(parsed);
    return CheckerStatus::Ok();
}

CheckerStatus ParseInt64(const char *text, int64_t *value) {
    if (text == nullptr || value == nullptr) {
        return CheckerStatus::Unsupported("missing int64 argument");
    }
    char *end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return CheckerStatus::Unsupported(std::string("invalid int64: ") + text);
    }
    *value = static_cast<int64_t>(parsed);
    return CheckerStatus::Ok();
}

CheckerStatus ParseOp(const std::string &text, CollectiveOp *op) {
    if (text == "allgather") {
        *op = CollectiveOp::kAllGather;
        return CheckerStatus::Ok();
    }
    if (text == "allreduce") {
        *op = CollectiveOp::kAllReduce;
        return CheckerStatus::Ok();
    }
    if (text == "ep_dispatch") {
        *op = CollectiveOp::kEpDispatch;
        return CheckerStatus::Ok();
    }
    if (text == "ep_combine") {
        *op = CollectiveOp::kEpCombine;
        return CheckerStatus::Ok();
    }
    return CheckerStatus::Unsupported("unsupported op: " + text);
}

CheckerStatus ParseScheduler(const std::string &text, SchedulerMode *mode) {
    if (text == "serial") {
        *mode = SchedulerMode::kSerial;
        return CheckerStatus::Ok();
    }
    if (text == "round_robin") {
        *mode = SchedulerMode::kRoundRobin;
        return CheckerStatus::Ok();
    }
    return CheckerStatus::Unsupported("unsupported scheduler: " + text);
}

CheckerStatus ParseDatatype(const std::string &text, TileXR::TileXRDataType *data_type) {
    if (text == "int32") {
        *data_type = TileXR::TILEXR_DATA_TYPE_INT32;
        return CheckerStatus::Ok();
    }
    if (text == "fp16") {
        *data_type = TileXR::TILEXR_DATA_TYPE_FP16;
        return CheckerStatus::Ok();
    }
    if (text == "bf16" || text == "bfp16") {
        *data_type = TileXR::TILEXR_DATA_TYPE_BFP16;
        return CheckerStatus::Ok();
    }
    return CheckerStatus::Unsupported("unsupported datatype: " + text);
}

CheckerStatus ParseReduceOp(const std::string &text, TileXR::TileXRReduceOp *reduce_op) {
    if (text == "sum") {
        *reduce_op = TileXR::TILEXR_REDUCE_SUM;
        return CheckerStatus::Ok();
    }
    return CheckerStatus::Unsupported("unsupported reduce-op: " + text);
}

CheckerStatus ParseAlgorithm(const std::string &text, AlgorithmId *algorithm) {
    if (text == "default") {
        *algorithm = AlgorithmId::kDefault;
        return CheckerStatus::Ok();
    }
    if (text == "allgather_hierarchy_double_ring") {
        *algorithm = AlgorithmId::kAllGatherHierarchyDoubleRing;
        return CheckerStatus::Ok();
    }
    if (text == "allreduce_big_data") {
        *algorithm = AlgorithmId::kAllReduceBigData;
        return CheckerStatus::Ok();
    }
    return CheckerStatus::Unsupported("unsupported algorithm: " + text);
}

CheckerStatus RequireValue(int argc, const char *const *argv, int index, const char **value) {
    if (index + 1 >= argc) {
        return CheckerStatus::Unsupported(std::string("missing value for ") + argv[index]);
    }
    *value = argv[index + 1];
    return CheckerStatus::Ok();
}

CheckerStatus EnsureDirectory(const std::string &output_dir) {
    if (output_dir.empty()) {
        return CheckerStatus::Unsupported("output directory is empty");
    }

    std::string current;
    if (output_dir[0] == '/') {
        current = "/";
    }

    std::string::size_type start = output_dir[0] == '/' ? 1 : 0;
    while (start <= output_dir.size()) {
        const std::string::size_type end = output_dir.find('/', start);
        const std::string part = output_dir.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (!part.empty()) {
            if (!current.empty() && current[current.size() - 1] != '/') {
                current += "/";
            }
            current += part;
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                return CheckerStatus::Fail("failed to create output directory " + current + ": " +
                                           std::strerror(errno));
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return CheckerStatus::Ok();
}

CheckerStatus ValidateCliRunnableShape(const CheckerCase &test_case) {
    if (test_case.rank_size <= 0) {
        return CheckerStatus::Unsupported("rank_size must be positive");
    }
    if (test_case.server_count <= 0) {
        return CheckerStatus::Unsupported("server_count must be positive");
    }
    if (test_case.server_count > test_case.rank_size) {
        return CheckerStatus::Unsupported("server_count must not exceed rank_size");
    }
    if (test_case.rank_size % test_case.server_count != 0) {
        return CheckerStatus::Unsupported("rank_size must be divisible by server_count");
    }
    if (test_case.op == CollectiveOp::kEpDispatch ||
        test_case.op == CollectiveOp::kEpCombine) {
        if (test_case.bs <= 0 || test_case.h <= 0 || test_case.top_k <= 0 ||
            test_case.moe_expert_num <= 0) {
            return CheckerStatus::Unsupported(
                std::string(ToString(test_case.op)) +
                " requires positive bs, h, top_k, and moe_expert_num");
        }
    } else if (test_case.count <= 0) {
        return CheckerStatus::Unsupported("count must be positive");
    }
    return CheckerStatus::Ok();
}

CheckerStatus WriteTextFile(const std::string &path, const std::string &content) {
    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        return CheckerStatus::Fail("failed to open " + path + " for writing");
    }
    output << content;
    output.close();
    if (!output) {
        return CheckerStatus::Fail("failed to write " + path);
    }
    return CheckerStatus::Ok();
}

bool HasFindingKind(const FindingSet &findings, FindingKind kind) {
    const std::vector<Finding> &items = findings.findings();
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].kind == kind) {
            return true;
        }
    }
    return false;
}

std::string Trim(const std::string &text) {
    std::string::size_type begin = 0;
    while (begin < text.size() &&
           (text[begin] == ' ' || text[begin] == '\t' ||
            text[begin] == '\r' || text[begin] == '\n')) {
        ++begin;
    }
    std::string::size_type end = text.size();
    while (end > begin &&
           (text[end - 1] == ' ' || text[end - 1] == '\t' ||
            text[end - 1] == '\r' || text[end - 1] == '\n')) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool FindJsonValueStart(const std::string &json, const std::string &key,
                        size_t *value_start) {
    if (value_start == nullptr) {
        return false;
    }
    const std::string quoted_key = "\"" + key + "\"";
    const std::string::size_type key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        return false;
    }
    const std::string::size_type colon =
        json.find(':', key_pos + quoted_key.size());
    if (colon == std::string::npos) {
        return false;
    }
    size_t pos = colon + 1;
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\t' ||
            json[pos] == '\r' || json[pos] == '\n')) {
        ++pos;
    }
    if (pos >= json.size()) {
        return false;
    }
    *value_start = pos;
    return true;
}

bool FindJsonFieldValue(const std::string &json, const std::string &key,
                        std::string *value) {
    if (value == nullptr) {
        return false;
    }
    size_t pos = 0;
    if (!FindJsonValueStart(json, key, &pos)) {
        return false;
    }
    if (json[pos] == '"') {
        std::string out;
        bool escaped = false;
        for (size_t i = pos + 1; i < json.size(); ++i) {
            const char ch = json[i];
            if (escaped) {
                switch (ch) {
                    case 'n':
                        out += '\n';
                        break;
                    case 'r':
                        out += '\r';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    case '\\':
                        out += '\\';
                        break;
                    case '"':
                        out += '"';
                        break;
                    default:
                        out += ch;
                        break;
                }
                escaped = false;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                continue;
            }
            if (ch == '"') {
                *value = out;
                return true;
            }
            out += ch;
        }
        return false;
    }

    const std::string::size_type end = json.find_first_of(",}", pos);
    if (end == std::string::npos) {
        return false;
    }
    *value = Trim(json.substr(pos, end - pos));
    return true;
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

bool ReadTextFile(const std::string &path, std::string *content) {
    if (content == nullptr) {
        return false;
    }
    std::ifstream input(path.c_str());
    if (!input.is_open()) {
        return false;
    }
    std::ostringstream out;
    out << input.rdbuf();
    *content = out.str();
    return true;
}

CheckerStatus ReadJsonIntField(const std::string &json, const std::string &key,
                               int *value) {
    std::string text;
    if (!FindJsonFieldValue(json, key, &text)) {
        return CheckerStatus::Fail("event trace missing integer field: " + key);
    }
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        parsed < INT_MIN || parsed > INT_MAX) {
        return CheckerStatus::Fail("event trace invalid integer field: " + key);
    }
    *value = static_cast<int>(parsed);
    return CheckerStatus::Ok();
}

CheckerStatus ReadJsonUint64Field(const std::string &json, const std::string &key,
                                  uint64_t *value) {
    std::string text;
    if (!FindJsonFieldValue(json, key, &text)) {
        return CheckerStatus::Fail("event trace missing uint64 field: " + key);
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        return CheckerStatus::Fail("event trace invalid uint64 field: " + key);
    }
    *value = static_cast<uint64_t>(parsed);
    return CheckerStatus::Ok();
}

CheckerStatus ReadJsonSizeField(const std::string &json, const std::string &key,
                                size_t *value) {
    uint64_t parsed = 0;
    CheckerStatus status = ReadJsonUint64Field(json, key, &parsed);
    if (!status.ok()) {
        return status;
    }
    *value = static_cast<size_t>(parsed);
    return CheckerStatus::Ok();
}

CheckerStatus ReadJsonBoolField(const std::string &json, const std::string &key,
                                bool *value) {
    std::string text;
    if (!FindJsonFieldValue(json, key, &text)) {
        return CheckerStatus::Fail("event trace missing bool field: " + key);
    }
    if (text == "true") {
        *value = true;
        return CheckerStatus::Ok();
    }
    if (text == "false") {
        *value = false;
        return CheckerStatus::Ok();
    }
    return CheckerStatus::Fail("event trace invalid bool field: " + key);
}

CheckerStatus ReadJsonStringField(const std::string &json, const std::string &key,
                                  std::string *value) {
    if (!FindJsonFieldValue(json, key, value)) {
        return CheckerStatus::Fail("event trace missing string field: " + key);
    }
    return CheckerStatus::Ok();
}

bool ParseEventKindName(const std::string &text, EventKind *kind) {
    if (text == "RANK_START") {
        *kind = EventKind::kRankStart;
    } else if (text == "RANK_END") {
        *kind = EventKind::kRankEnd;
    } else if (text == "COPY") {
        *kind = EventKind::kCopy;
    } else if (text == "READ") {
        *kind = EventKind::kRead;
    } else if (text == "WRITE") {
        *kind = EventKind::kWrite;
    } else if (text == "FLAG_STORE") {
        *kind = EventKind::kFlagStore;
    } else if (text == "FLAG_WAIT") {
        *kind = EventKind::kFlagWait;
    } else if (text == "PIPE_SET") {
        *kind = EventKind::kPipeSet;
    } else if (text == "PIPE_WAIT") {
        *kind = EventKind::kPipeWait;
    } else if (text == "PIPE_BARRIER") {
        *kind = EventKind::kPipeBarrier;
    } else if (text == "BARRIER") {
        *kind = EventKind::kBarrier;
    } else if (text == "DIAGNOSTIC") {
        *kind = EventKind::kDiagnostic;
    } else {
        return false;
    }
    return true;
}

bool ParseBufferRoleName(const std::string &text, BufferRole *role) {
    if (text == "UserInput") {
        *role = BufferRole::kUserInput;
    } else if (text == "UserOutput") {
        *role = BufferRole::kUserOutput;
    } else if (text == "CommFlag") {
        *role = BufferRole::kCommFlag;
    } else if (text == "CommData") {
        *role = BufferRole::kCommData;
    } else if (text == "LocalUb") {
        *role = BufferRole::kLocalUb;
    } else if (text == "Metadata") {
        *role = BufferRole::kMetadata;
    } else if (text == "RegisteredCommBuffer") {
        *role = BufferRole::kRegisteredCommBuffer;
    } else {
        return false;
    }
    return true;
}

CheckerStatus ParseEventJsonLine(const std::string &line, Event *event) {
    if (event == nullptr) {
        return CheckerStatus::Fail("event trace parser event pointer is null");
    }
    CheckerStatus status = ReadJsonUint64Field(line, "id", &event->id);
    if (!status.ok()) {
        return status;
    }
    std::string kind_name;
    status = ReadJsonStringField(line, "kind", &kind_name);
    if (!status.ok()) {
        return status;
    }
    if (!ParseEventKindName(kind_name, &event->kind)) {
        return CheckerStatus::Fail("event trace unsupported event kind: " + kind_name);
    }
    std::string role_name;
    status = ReadJsonStringField(line, "buffer_role", &role_name);
    if (!status.ok()) {
        return status;
    }
    if (!ParseBufferRoleName(role_name, &event->buffer_role)) {
        return CheckerStatus::Fail("event trace unsupported buffer role: " + role_name);
    }
    status = ReadJsonIntField(line, "rank", &event->rank);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonIntField(line, "peer_rank", &event->peer_rank);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonIntField(line, "server", &event->server);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonIntField(line, "peer_server", &event->peer_server);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonIntField(line, "core", &event->core);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonIntField(line, "pipe", &event->pipe);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonIntField(line, "event_id", &event->event_id);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonIntField(line, "slot", &event->slot);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonUint64Field(line, "magic", &event->magic);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonSizeField(line, "offset", &event->offset);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonSizeField(line, "bytes", &event->bytes);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonBoolField(line, "allow_future_producer",
                               &event->allow_future_producer);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonStringField(line, "source_file", &event->source_file);
    if (!status.ok()) {
        return status;
    }
    status = ReadJsonIntField(line, "source_line", &event->source_line);
    if (!status.ok()) {
        return status;
    }
    return ReadJsonStringField(line, "detail", &event->detail);
}

CheckerStatus ReadEventTraceJsonl(const std::string &path, EventLog *events) {
    if (events == nullptr) {
        return CheckerStatus::Fail("event trace output pointer is null");
    }
    std::ifstream input(path.c_str());
    if (!input.is_open()) {
        return CheckerStatus::Fail("failed to open event trace: " + path);
    }
    events->Clear();
    std::string line;
    int line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        Event event;
        CheckerStatus status = ParseEventJsonLine(line, &event);
        if (!status.ok()) {
            std::ostringstream message;
            message << status.message << " at " << path << ":" << line_no;
            return CheckerStatus::Fail(message.str());
        }
        events->Add(event);
    }
    if (events->events().empty()) {
        return CheckerStatus::Unsupported("event trace contains no events: " + path);
    }
    return CheckerStatus::Ok();
}

struct RequiredEventCoverage {
    EventKind kind = EventKind::kDiagnostic;
    std::string source_file;
    int source_line = 0;
};

std::string InstalledTraceManifestPath(const std::string &repo_root,
                                       const std::string &adapter_name) {
    return JoinPath(JoinPath(repo_root, "tools/checker/installed_traces"),
                    adapter_name + "_trace_manifest.json");
}

CheckerStatus ReadRequiredEventCoverage(const std::string &repo_root,
                                        const std::string &adapter_name,
                                        std::vector<RequiredEventCoverage> *coverage) {
    if (coverage == nullptr) {
        return CheckerStatus::Fail("event coverage output pointer is null");
    }
    coverage->clear();
    if (adapter_name.empty()) {
        return CheckerStatus::Ok();
    }
    const std::string manifest_path =
        InstalledTraceManifestPath(repo_root.empty() ? "." : repo_root,
                                   adapter_name);
    std::string manifest;
    if (!ReadTextFile(manifest_path, &manifest)) {
        return CheckerStatus::Unsupported(
            "installed trace manifest is missing for event trace validation: " +
            manifest_path);
    }

    size_t array_pos = 0;
    if (!FindJsonValueStart(manifest, "required_event_coverage", &array_pos)) {
        return CheckerStatus::Ok();
    }
    if (array_pos >= manifest.size() || manifest[array_pos] != '[') {
        return CheckerStatus::Unsupported(
            "installed trace manifest required_event_coverage must be an array: " +
            manifest_path);
    }
    const size_t array_end =
        FindMatchingJsonBracket(manifest, array_pos, '[', ']');
    if (array_end == std::string::npos) {
        return CheckerStatus::Unsupported(
            "installed trace manifest required_event_coverage is malformed: " +
            manifest_path);
    }

    size_t pos = array_pos + 1;
    int entry_index = 0;
    while (pos < array_end) {
        while (pos < array_end &&
               (manifest[pos] == ' ' || manifest[pos] == '\t' ||
                manifest[pos] == '\r' || manifest[pos] == '\n' ||
                manifest[pos] == ',')) {
            ++pos;
        }
        if (pos >= array_end) {
            break;
        }
        if (manifest[pos] != '{') {
            return CheckerStatus::Unsupported(
                "installed trace manifest required_event_coverage entry is malformed: " +
                manifest_path);
        }
        const size_t object_end =
            FindMatchingJsonBracket(manifest, pos, '{', '}');
        if (object_end == std::string::npos || object_end > array_end) {
            return CheckerStatus::Unsupported(
                "installed trace manifest required_event_coverage entry is malformed: " +
                manifest_path);
        }
        const std::string object = manifest.substr(pos, object_end - pos + 1);
        std::string kind_name;
        RequiredEventCoverage item;
        CheckerStatus status = ReadJsonStringField(object, "kind", &kind_name);
        if (!status.ok() || !ParseEventKindName(kind_name, &item.kind)) {
            return CheckerStatus::Unsupported(
                "installed trace manifest required_event_coverage[" +
                std::to_string(entry_index) + "].kind is invalid");
        }
        status = ReadJsonStringField(object, "source_file", &item.source_file);
        if (!status.ok()) {
            return CheckerStatus::Unsupported(
                "installed trace manifest required_event_coverage[" +
                std::to_string(entry_index) + "].source_file is missing");
        }
        status = ReadJsonIntField(object, "source_line", &item.source_line);
        if (!status.ok()) {
            return CheckerStatus::Unsupported(
                "installed trace manifest required_event_coverage[" +
                std::to_string(entry_index) + "].source_line is missing");
        }
        coverage->push_back(item);
        pos = object_end + 1;
        ++entry_index;
    }
    return CheckerStatus::Ok();
}

bool HasRequiredEventCoverage(const EventLog &events,
                              const RequiredEventCoverage &required) {
    const std::vector<Event> &items = events.events();
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].kind == required.kind &&
            items[i].source_file == required.source_file &&
            items[i].source_line == required.source_line) {
            return true;
        }
    }
    return false;
}

FindingSet CheckRequiredEventCoverage(
        const EventLog &events,
        const std::vector<RequiredEventCoverage> &coverage) {
    FindingSet findings;
    for (size_t i = 0; i < coverage.size(); ++i) {
        const RequiredEventCoverage &required = coverage[i];
        if (HasRequiredEventCoverage(events, required)) {
            continue;
        }
        Finding finding;
        finding.id = std::string("EVENT_COVERAGE_GAP:") +
                     EventKindName(required.kind) + ":" +
                     required.source_file + ":" +
                     std::to_string(required.source_line);
        finding.kind = FindingKind::kEventCoverageGap;
        finding.severity = Severity::kError;
        finding.message =
            "Event trace is missing a required source coverage event from the installed trace manifest.";
        finding.next_action =
            "Regenerate or fix the trace adapter so this source hook emits an abstract event, or mark the manifest action as an explicit unsupported gate.";
        finding.source_file = required.source_file;
        finding.source_line = required.source_line;
        finding.context = std::string("event_coverage:") +
                          EventKindName(required.kind) + ":" +
                          required.source_file + ":" +
                          std::to_string(required.source_line);
        finding.confidence = 0.99;
        findings.Add(finding);
    }
    return findings;
}

bool IsProductionOutputPath(const std::string &path) {
    return path.find("src/collectives/") == 0 ||
           path.find("/src/collectives/") != std::string::npos ||
           path.find("src/ep/") == 0 ||
           path.find("/src/ep/") != std::string::npos;
}

CheckerStatus RejectProductionOutputPath(const std::string &path,
                                         const std::string &kind) {
    if (IsProductionOutputPath(path)) {
        return CheckerStatus::Unsupported(
            kind + " output must be a checker output path, not a production source path");
    }
    return CheckerStatus::Ok();
}

bool SafeAdd(size_t lhs, size_t rhs, size_t *out) {
    if (lhs > static_cast<size_t>(-1) - rhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool EventCoversOutputOffset(const Event &event, int rank, size_t offset, size_t bytes) {
    if ((event.kind != EventKind::kWrite && event.kind != EventKind::kCopy) ||
        event.rank != rank || event.buffer_role != BufferRole::kUserOutput) {
        return false;
    }
    size_t event_end = 0;
    size_t target_end = 0;
    if (!SafeAdd(event.offset, event.bytes, &event_end) ||
        !SafeAdd(offset, bytes, &target_end)) {
        return false;
    }
    return event.offset <= offset && event_end >= target_end;
}

const Event *FindLatestOutputProducer(const EventLog &events, int rank, size_t offset,
                                      size_t bytes) {
    const std::vector<Event> &items = events.events();
    const Event *producer = nullptr;
    for (size_t i = 0; i < items.size(); ++i) {
        if (EventCoversOutputOffset(items[i], rank, offset, bytes)) {
            producer = &items[i];
        }
    }
    return producer;
}

FindingSet AttachOutputProducerLocations(const FindingSet &findings, const EventLog &events) {
    FindingSet enriched;
    const std::vector<Finding> &items = findings.findings();
    for (size_t i = 0; i < items.size(); ++i) {
        Finding finding = items[i];
        if (finding.kind == FindingKind::kOutputMismatch) {
            const Event *producer =
                FindLatestOutputProducer(events, finding.rank, finding.offset, finding.bytes);
            if (producer != nullptr) {
                finding.event_log_id = producer->id;
                finding.peer_rank = producer->peer_rank;
                finding.server = producer->server;
                finding.peer_server = producer->peer_server;
                finding.core = producer->core;
                finding.pipe = producer->pipe;
                finding.event_id = producer->event_id;
                finding.slot = producer->slot;
                finding.buffer_role = producer->buffer_role;
                finding.source_file = producer->source_file;
                finding.source_line = producer->source_line;
            }
        }
        enriched.Add(finding);
    }
    return enriched;
}

void RenderEventContextJson(const EventLog &events, uint64_t event_log_id, std::ostream *out) {
    if (out == nullptr) {
        return;
    }
    (*out) << "[";
    if (event_log_id == 0) {
        (*out) << "]";
        return;
    }

    const std::vector<Event> &items = events.events();
    size_t found = items.size();
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].id == event_log_id) {
            found = i;
            break;
        }
    }
    if (found == items.size()) {
        (*out) << "]";
        return;
    }

    const size_t begin = found > 2 ? found - 2 : 0;
    const size_t end = found + 3 < items.size() ? found + 3 : items.size();
    for (size_t i = begin; i < end; ++i) {
        const Event &event = items[i];
        if (i != begin) {
            (*out) << ",";
        }
        (*out) << "{"
               << "\"relative\":" << static_cast<long long>(i) - static_cast<long long>(found) << ","
               << "\"id\":" << event.id << ","
               << "\"kind\":\"" << EventKindName(event.kind) << "\","
               << "\"rank\":" << event.rank << ","
               << "\"peer_rank\":" << event.peer_rank << ","
               << "\"server\":" << event.server << ","
               << "\"peer_server\":" << event.peer_server << ","
               << "\"core\":" << event.core << ","
               << "\"pipe\":" << event.pipe << ","
               << "\"event_id\":" << event.event_id << ","
               << "\"buffer_role\":\"" << BufferReportName(event.buffer_role) << "\","
               << "\"offset\":" << event.offset << ","
               << "\"bytes\":" << event.bytes << ","
               << "\"source_file\":\"" << EscapeJson(event.source_file) << "\","
               << "\"source_line\":" << event.source_line << ","
               << "\"detail\":\"" << EscapeJson(event.detail) << "\""
               << "}";
    }
    (*out) << "]";
}

void RenderSourceExcerptJson(const std::string &source_file, int source_line,
                             const std::string &source_root,
                             std::ostream *out) {
    if (out == nullptr) {
        return;
    }
    (*out) << "[";
    if (source_file.empty() || source_line <= 0) {
        (*out) << "]";
        return;
    }
    std::ifstream input(source_file.c_str());
    if (!input.is_open() && !source_root.empty() &&
        !source_file.empty() && source_file[0] != '/') {
        input.clear();
        input.open(JoinPath(source_root, source_file).c_str());
    }
    if (!input.is_open()) {
        (*out) << "]";
        return;
    }

    const int begin_line = source_line > 2 ? source_line - 2 : 1;
    const int end_line = source_line + 2;
    std::string line;
    int line_no = 0;
    size_t emitted = 0;
    while (std::getline(input, line)) {
        ++line_no;
        if (line_no < begin_line) {
            continue;
        }
        if (line_no > end_line) {
            break;
        }
        if (emitted != 0) {
            (*out) << ",";
        }
        (*out) << "{"
               << "\"line\":" << line_no << ","
               << "\"text\":\"" << EscapeJson(line) << "\","
               << "\"is_target\":" << (line_no == source_line ? "true" : "false")
               << "}";
        ++emitted;
    }
    (*out) << "]";
}

void RenderSourceExcerptJson(const std::string &source_file, int source_line,
                             std::ostream *out) {
    RenderSourceExcerptJson(source_file, source_line, std::string(), out);
}

bool IsFlagEvent(EventKind kind) {
    return kind == EventKind::kFlagStore || kind == EventKind::kFlagWait;
}

bool IsPipeEvent(EventKind kind) {
    return kind == EventKind::kPipeSet || kind == EventKind::kPipeWait ||
           kind == EventKind::kPipeBarrier;
}

EventKind EventKindFromTraceSourceEvent(const std::string &event) {
    if (event == "COPY") {
        return EventKind::kCopy;
    }
    if (event == "FLAG_STORE") {
        return EventKind::kFlagStore;
    }
    if (event == "FLAG_WAIT") {
        return EventKind::kFlagWait;
    }
    if (event == "PIPE_SET") {
        return EventKind::kPipeSet;
    }
    if (event == "PIPE_WAIT") {
        return EventKind::kPipeWait;
    }
    if (event == "PIPE_BARRIER") {
        return EventKind::kPipeBarrier;
    }
    return EventKind::kDiagnostic;
}

Event MakeTraceTemplateEvent(const TraceSourceObservationItem &item,
                             int line_index) {
    Event event;
    event.kind = EventKindFromTraceSourceEvent(item.event);
    event.rank = 0;
    event.peer_rank = IsFlagEvent(event.kind) ? 0 : -1;
    event.server = 0;
    event.peer_server = event.peer_rank >= 0 ? 0 : -1;
    event.core = IsPipeEvent(event.kind) ? 0 : -1;
    event.pipe = IsPipeEvent(event.kind) ? 0 : -1;
    event.event_id =
        event.kind == EventKind::kPipeSet || event.kind == EventKind::kPipeWait
            ? 0
            : -1;
    event.buffer_role = event.kind == EventKind::kCopy ? BufferRole::kCommData
                                                        : BufferRole::kMetadata;
    event.slot = IsFlagEvent(event.kind) ? 0 : -1;
    event.magic = IsFlagEvent(event.kind) ? 1 : 0;
    event.bytes = event.kind == EventKind::kCopy ? 1 : 0;
    if (line_index >= 0 &&
        static_cast<size_t>(line_index) < item.lines.size()) {
        event.source_line = item.lines[static_cast<size_t>(line_index)];
    }
    std::ostringstream detail;
    detail << "trace template " << item.symbol << " -> " << item.event
           << "; replace rank/slot/offset/bytes with instrumented values";
    event.detail = detail.str();
    return event;
}

EventLog BuildTraceEventTemplate(const TraceSourceAnalysis &analysis) {
    EventLog events;
    for (size_t i = 0; i < analysis.auto_hook_candidates.size(); ++i) {
        const TraceSourceObservationItem &item = analysis.auto_hook_candidates[i];
        for (size_t j = 0; j < item.lines.size(); ++j) {
            Event event = MakeTraceTemplateEvent(item, static_cast<int>(j));
            event.source_file = analysis.source_file;
            events.Add(event);
        }
    }
    return events;
}

bool SameFlagSlotAsFinding(const Event &event, const Finding &finding) {
    return IsFlagEvent(event.kind) && finding.buffer_role == BufferRole::kCommFlag &&
           finding.slot >= 0 && event.slot == finding.slot &&
           ((event.rank == finding.rank && event.peer_rank == finding.peer_rank) ||
            (event.rank == finding.peer_rank && event.peer_rank == finding.rank));
}

bool SamePipeAsFinding(const Event &event, const Finding &finding) {
    return IsPipeEvent(event.kind) && finding.pipe >= 0 &&
           event.rank == finding.rank && event.core == finding.core &&
           event.pipe == finding.pipe &&
           (finding.event_id < 0 || event.event_id == finding.event_id ||
            event.kind == EventKind::kPipeBarrier);
}

bool SameSourceAsFinding(const Event &event, const Finding &finding) {
    if (finding.source_file.empty() || event.source_file != finding.source_file) {
        return false;
    }
    const int delta = event.source_line - finding.source_line;
    return delta >= -2 && delta <= 2;
}

std::string TimelineRelation(const Event &event, const Finding &finding) {
    if (event.id == finding.event_log_id) {
        return "trigger";
    }
    if (SameFlagSlotAsFinding(event, finding)) {
        return "same_flag_slot";
    }
    if (SamePipeAsFinding(event, finding)) {
        return "same_pipe_event";
    }
    if (SameSourceAsFinding(event, finding)) {
        return "near_source";
    }
    return "near_event";
}

bool EventBelongsToTimeline(const Event &event, const Finding &finding) {
    if (event.id == finding.event_log_id) {
        return true;
    }
    if (SameFlagSlotAsFinding(event, finding) ||
        SamePipeAsFinding(event, finding) ||
        SameSourceAsFinding(event, finding)) {
        return true;
    }
    if (finding.event_log_id != 0) {
        const uint64_t min_id = finding.event_log_id > 2 ? finding.event_log_id - 2 : 1;
        const uint64_t max_id = finding.event_log_id + 2;
        return event.id >= min_id && event.id <= max_id;
    }
    return false;
}

void RenderTimelineEventJson(const Event &event, const Finding &finding,
                             std::ostream *out) {
    const uint64_t magic_epoch = event.magic >> 32;
    const uint32_t magic_value = static_cast<uint32_t>(event.magic & 0xFFFFFFFFULL);
    (*out) << "{"
           << "\"relation\":\"" << TimelineRelation(event, finding) << "\","
           << "\"id\":" << event.id << ","
           << "\"kind\":\"" << EventKindName(event.kind) << "\","
           << "\"rank\":" << event.rank << ","
           << "\"peer_rank\":" << event.peer_rank << ","
           << "\"server\":" << event.server << ","
           << "\"peer_server\":" << event.peer_server << ","
           << "\"core\":" << event.core << ","
           << "\"pipe\":" << event.pipe << ","
           << "\"event_id\":" << event.event_id << ","
           << "\"buffer_role\":\"" << BufferReportName(event.buffer_role) << "\","
           << "\"slot\":" << event.slot << ","
           << "\"magic\":" << event.magic << ","
           << "\"magic_epoch\":" << magic_epoch << ","
           << "\"magic_value\":" << magic_value << ","
           << "\"offset\":" << event.offset << ","
           << "\"bytes\":" << event.bytes << ","
           << "\"source_file\":\"" << EscapeJson(event.source_file) << "\","
           << "\"source_line\":" << event.source_line << ","
           << "\"detail\":\"" << EscapeJson(event.detail) << "\""
           << "}";
}

void RenderTimelineJson(const EventLog &events, const Finding &finding,
                        std::ostream *out) {
    if (out == nullptr) {
        return;
    }
    (*out) << "[";
    const std::vector<Event> &items = events.events();
    size_t emitted = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (!EventBelongsToTimeline(items[i], finding)) {
            continue;
        }
        if (emitted != 0) {
            (*out) << ",";
        }
        RenderTimelineEventJson(items[i], finding, out);
        ++emitted;
        if (emitted >= 16) {
            break;
        }
    }
    (*out) << "]";
}

std::string BuildTimelineSummary(const EventLog &events, const Finding *finding) {
    if (finding == nullptr) {
        return "";
    }
    const std::vector<Event> &items = events.events();
    size_t count = 0;
    int best_priority = 100;
    std::string best_relation;
    for (size_t i = 0; i < items.size(); ++i) {
        if (!EventBelongsToTimeline(items[i], *finding)) {
            continue;
        }
        const std::string relation = TimelineRelation(items[i], *finding);
        int priority = 5;
        if (relation == "same_flag_slot" || relation == "same_pipe_event") {
            priority = 0;
        } else if (relation == "near_source") {
            priority = 1;
        } else if (relation == "near_event") {
            priority = 2;
        } else if (relation == "trigger") {
            priority = 3;
        }
        if (priority < best_priority) {
            best_priority = priority;
            best_relation = relation;
        }
        ++count;
        if (count >= 16) {
            break;
        }
    }
    if (count == 0) {
        return "";
    }
    if (best_relation.empty()) {
        best_relation = "trigger";
    }
    std::ostringstream out;
    out << "timeline events: " << count << "\n"
        << "timeline first relation: " << best_relation << "\n"
        << "timeline: checker_report.json top_finding.timeline\n";
    return out.str();
}

CheckerStatus FinalizeCliStatus(const RunResult &result) {
    if (HasFindingKind(result.findings, FindingKind::kUnsupportedApi)) {
        return CheckerStatus::Unsupported("checker encountered unsupported case");
    }
    if (!result.findings.findings().empty() || !result.mismatches.empty()) {
        return CheckerStatus::Fail("checker detected findings or mismatches");
    }
    return result.status.ok() ? CheckerStatus::Ok() : result.status;
}

void InjectReadBeforeCopy(RankWorld *world) {
    Event event;
    event.kind = EventKind::kRead;
    event.rank = 1;
    event.peer_rank = 0;
    event.server = world == nullptr ? -1 : world->ServerOfRank(event.rank);
    event.peer_server = world == nullptr ? -1 : world->ServerOfRank(event.peer_rank);
    event.buffer_role = BufferRole::kCommData;
    event.slot = 99;
    event.offset = 0;
    event.bytes = sizeof(int32_t);
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = "cli injected read before copy";
    world->events().Add(event);
}

void InjectPeerUserRead(RankWorld *world) {
    Event event;
    event.kind = EventKind::kRead;
    event.rank = 0;
    event.peer_rank = 1;
    event.server = world == nullptr ? -1 : world->ServerOfRank(event.rank);
    event.peer_server = world == nullptr ? -1 : world->ServerOfRank(event.peer_rank);
    event.buffer_role = BufferRole::kUserInput;
    event.slot = 0;
    event.offset = 0;
    event.bytes = sizeof(int32_t);
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = "cli injected peer user read";
    world->events().Add(event);
}

void InjectPipeWait(RankWorld *world) {
    Event event;
    event.kind = EventKind::kPipeWait;
    event.rank = 0;
    event.peer_rank = -1;
    event.server = world == nullptr ? -1 : world->ServerOfRank(event.rank);
    event.peer_server = -1;
    event.core = 3;
    event.pipe = 1;
    event.event_id = 1;
    event.buffer_role = BufferRole::kMetadata;
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = "cli injected pipe wait without producer";
    world->events().Add(event);
}

void InjectEpWindowRead(RankWorld *world) {
    Event event;
    event.kind = EventKind::kRead;
    event.rank = 0;
    event.peer_rank = 1;
    event.server = world == nullptr ? -1 : world->ServerOfRank(event.rank);
    event.peer_server = world == nullptr ? -1 : world->ServerOfRank(event.peer_rank);
    event.buffer_role = BufferRole::kRegisteredCommBuffer;
    event.slot = 99;
    event.offset = 999999;
    event.bytes = 64;
    event.source_file = "src/ep/kernels/tilexr_ep_combine_kernel.cpp";
    event.source_line = 211;
    event.detail = "cli injected ep window read without producer";
    world->events().Add(event);
}

void PrintUsage(std::ostream *stream) {
    if (stream == nullptr) {
        return;
    }
    (*stream)
        << "Usage: tilexr_checker --op <allgather|allreduce> --rank-size <n> --count <n> "
           "--datatype int32 [--reduce-op sum] [--scheduler <serial|round_robin>] "
           "[--algorithm <default|allgather_hierarchy_double_ring|allreduce_big_data>] "
           "[--server-count <n>] "
           "[--output-dir <dir>] [--inject-read-before-copy] [--inject-peer-user-read] "
           "[--inject-pipe-wait] [--inject-ep-window-read]\n"
           "       tilexr_checker --op <ep_dispatch|ep_combine> --rank-size <n> --bs <n> --h <n> "
           "--top-k <n> --moe-expert-num <n> --datatype <fp16|bfp16> "
           "[--server-count <n>] [--output-dir <dir>] [--inject-ep-window-read]\n"
           "       tilexr_checker --generate-trace-adapter --adapter-name <name> "
           "--source-file <path> --target-header <include> --output <checker-shim.h> "
           "[--manifest-output <manifest.json>] "
           "[--runner-output <runner.cpp> --runner-function <name> --runner-materializer <name>] "
           "[--probe-output <probe.cpp>] "
           "[--onboarding-output <onboarding.md>] [--strict-source-observation]\n";
    (*stream)
        << "       tilexr_checker --scaffold-trace-bundle --source-file <path> "
           "--output-dir <generated-checker-dir> [--adapter-name <name>] "
           "[--target-header <include>] [--repo-root <repo>] "
           "[--verification-report <file>] [--strict-source-observation]\n";
    (*stream)
        << "       tilexr_checker --verify-trace-bundle --adapter-name <name> "
           "--output-dir <generated-checker-dir> [--repo-root <repo>] "
           "[--verification-report <file>]\n";
    (*stream)
        << "       tilexr_checker --verify-installed-trace --adapter-name <name> "
           "[--repo-root <repo>] [--verification-report <file>]\n";
    (*stream)
        << "       tilexr_checker --list-installed-traces [--repo-root <repo>]\n";
    (*stream)
        << "       tilexr_checker --verify-all-installed-traces [--repo-root <repo>] "
           "[--verification-report <file>]\n";
    (*stream)
        << "       tilexr_checker --list-capabilities [--repo-root <repo>] "
           "[--capability-report <file>]\n";
    (*stream)
        << "       tilexr_checker --analyze-trace-source --source-file <path> "
           "[--adapter-name <name>] [--target-header <include>] [--repo-root <repo>] "
           "[--trace-analysis-output <file>] "
           "[--event-trace-template-output <events.jsonl>]\n";
    (*stream)
        << "       tilexr_checker --validate-event-trace --event-trace <events.jsonl> "
           "--output-dir <dir> [--repo-root <repo>] [--adapter-name <name>]\n";
}

void RenderJsonStringArray(const std::vector<std::string> &items, std::ostream *out) {
    (*out) << "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
            (*out) << ",";
        }
        (*out) << "\"" << EscapeJson(items[i]) << "\"";
    }
    (*out) << "]";
}

const char *StatusCodeReportName(CheckerStatusCode code) {
    switch (code) {
        case CheckerStatusCode::kOk:
            return "OK";
        case CheckerStatusCode::kFail:
            return "FAIL";
        case CheckerStatusCode::kUnsupported:
            return "UNSUPPORTED";
        case CheckerStatusCode::kInconclusive:
            return "INCONCLUSIVE";
        case CheckerStatusCode::kInternalError:
            return "INTERNAL_ERROR";
    }
    return "UNKNOWN";
}

bool EndsWith(const std::string &text, const std::string &suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> ListInstalledTraceAdapters(const std::string &repo_root) {
    std::vector<std::string> adapters;
    const std::string dir = JoinPath(repo_root, "tools/checker/installed_traces");
    DIR *handle = opendir(dir.c_str());
    if (handle == nullptr) {
        return adapters;
    }
    const std::string suffix = "_trace_manifest.json";
    while (dirent *entry = readdir(handle)) {
        const std::string name = entry->d_name;
        if (EndsWith(name, suffix)) {
            adapters.push_back(name.substr(0, name.size() - suffix.size()));
        }
    }
    closedir(handle);
    std::sort(adapters.begin(), adapters.end());
    return adapters;
}

struct InstalledTraceBatchItem {
    std::string adapter_name;
    CheckerStatus status;
    TraceBundleVerificationResult verification;
};

std::string RenderInstalledTraceInventoryJson(
        const std::string &repo_root,
        const std::vector<InstalledTraceBatchItem> &items,
        const CheckerStatus &overall_status) {
    std::ostringstream out;
    out << "{"
        << "\"mode\":\"installed_trace_inventory\","
        << "\"repo_root\":\"" << EscapeJson(repo_root) << "\","
        << "\"overall_status\":\"" << StatusCodeReportName(overall_status.code) << "\","
        << "\"adapter_count\":" << items.size() << ","
        << "\"adapters\":[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "{"
            << "\"adapter_name\":\"" << EscapeJson(items[i].adapter_name) << "\","
            << "\"status_code\":\"" << StatusCodeReportName(items[i].status.code) << "\","
            << "\"status_message\":\""
            << EscapeJson(items[i].status.ok() ? "OK" : items[i].status.message) << "\","
            << "\"complete\":" << (items[i].verification.complete ? "true" : "false")
            << ",\"probe_compile\":\""
            << (items[i].verification.probe_compile_passed ? "PASS" :
                (items[i].verification.probe_compile_attempted ? "FAIL" : "SKIPPED"))
            << "\",\"runtime_smoke\":\""
            << (items[i].verification.runtime_smoke_passed ? "PASS" : "FAIL")
            << "\",\"runtime_smoke_cases\":"
            << items[i].verification.runtime_smoke_case_count
            << ",\"runtime_events\":" << items[i].verification.runtime_smoke_event_count
            << ",\"missing_artifacts\":";
        RenderJsonStringArray(items[i].verification.missing_artifacts, &out);
        out << ",\"missing_manifest_sections\":";
        RenderJsonStringArray(items[i].verification.missing_manifest_sections, &out);
        out << ",\"unchecked_items\":";
        RenderJsonStringArray(items[i].verification.unchecked_items, &out);
        out << ",\"summary\":\"" << EscapeJson(items[i].verification.summary) << "\""
            << "}";
    }
    out << "]}";
    return out.str();
}

CheckerStatus WriteInstalledTraceInventoryReportIfRequested(
        const std::string &path,
        const std::string &repo_root,
        const std::vector<InstalledTraceBatchItem> &items,
        const CheckerStatus &overall_status) {
    if (path.empty()) {
        return CheckerStatus::Ok();
    }
    CheckerStatus mkdir_status = EnsureDirectory(ParentDir(path));
    if (!mkdir_status.ok()) {
        return mkdir_status;
    }
    return WriteTextFile(path, RenderInstalledTraceInventoryJson(repo_root, items,
                                                                 overall_status));
}

struct CapabilityItem {
    std::string op;
    std::string name;
    std::string status;
    std::string mode;
    std::string source_file;
    std::string evidence;
};

std::vector<CapabilityItem> BuildCapabilityInventory(const std::string &repo_root) {
    std::vector<CapabilityItem> items;

    CapabilityItem allgather_default;
    allgather_default.op = "allgather";
    allgather_default.name = "default";
    allgather_default.status = "supported";
    allgather_default.mode = "generic_software_model";
    allgather_default.source_file = "checker generic model";
    allgather_default.evidence = "CollectiveExecutor default allgather oracle";
    items.push_back(allgather_default);

    CapabilityItem allreduce_default;
    allreduce_default.op = "allreduce";
    allreduce_default.name = "default";
    allreduce_default.status = "supported";
    allreduce_default.mode = "generic_software_model";
    allreduce_default.source_file = "checker generic model";
    allreduce_default.evidence = "CollectiveExecutor default allreduce sum oracle";
    items.push_back(allreduce_default);

    const std::vector<std::string> installed = ListInstalledTraceAdapters(repo_root);
    for (size_t i = 0; i < installed.size(); ++i) {
        CapabilityItem item;
        item.name = installed[i];
        item.status = "installed";
        item.evidence = "tools/checker/installed_traces/" + installed[i] +
                        "_trace_manifest.json";
        if (installed[i] == "allreduce_big_data") {
            item.op = "allreduce";
            item.mode = "production_header_trace";
            item.source_file = "src/collectives/kernels/allreduce_big_data.h";
        } else if (installed[i] == "allgather_hierarchy_double_ring") {
            item.op = "allgather";
            item.mode = "production_header_trace";
            item.source_file =
                "src/collectives/kernels/91093/allgather_hierarchy_double_ring.h";
        } else if (installed[i] == "ep_dispatch") {
            item.op = "ep_dispatch";
            item.mode = "source_aligned_cpu_oracle";
            item.source_file = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
        } else {
            item.op = "unknown";
            item.mode = "installed_manifest";
            item.source_file = "manifest";
        }
        items.push_back(item);
    }

    CapabilityItem ep_combine;
    ep_combine.op = "ep_combine";
    ep_combine.name = "ep_combine";
    ep_combine.status = "supported";
    ep_combine.mode = "source_aligned_cpu_oracle";
    ep_combine.source_file = "src/ep/kernels/tilexr_ep_combine_kernel.cpp";
    ep_combine.evidence =
        "checker-side inverse routing oracle consumes dispatch payload and assist metadata";
    items.push_back(ep_combine);
    return items;
}

std::string CapabilityDisplayName(const CapabilityItem &item) {
    if (item.op == "ep_dispatch" || item.op == "ep_combine") {
        return "default";
    }
    return item.name;
}

std::string RenderCapabilityInventoryJson(const std::string &repo_root,
                                          const std::vector<CapabilityItem> &items) {
    std::ostringstream out;
    out << "{"
        << "\"mode\":\"capability_inventory\","
        << "\"repo_root\":\"" << EscapeJson(repo_root) << "\","
        << "\"boundary\":{"
        << "\"requires_npu\":false,"
        << "\"instruction_level_execution\":false,"
        << "\"production_sources_modified\":false"
        << "},"
        << "\"capabilities\":[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "{"
            << "\"op\":\"" << EscapeJson(items[i].op) << "\","
            << "\"name\":\"" << EscapeJson(items[i].name) << "\","
            << "\"status\":\"" << EscapeJson(items[i].status) << "\","
            << "\"mode\":\"" << EscapeJson(items[i].mode) << "\","
            << "\"source_file\":\"" << EscapeJson(items[i].source_file) << "\","
            << "\"evidence\":\"" << EscapeJson(items[i].evidence) << "\""
            << "}";
    }
    out << "]}";
    return out.str();
}

CheckerStatus WriteCapabilityReportIfRequested(
        const std::string &path,
        const std::string &repo_root,
        const std::vector<CapabilityItem> &items) {
    if (path.empty()) {
        return CheckerStatus::Ok();
    }
    CheckerStatus mkdir_status = EnsureDirectory(ParentDir(path));
    if (!mkdir_status.ok()) {
        return mkdir_status;
    }
    return WriteTextFile(path, RenderCapabilityInventoryJson(repo_root, items));
}

std::string RenderVerificationReportJson(
        const std::string &mode,
        const std::string &adapter_name,
        const std::string &repo_root,
        const std::string &bundle_dir,
        const CheckerStatus &status,
        const TraceBundleVerificationResult &verification) {
    std::ostringstream out;
    out << "{"
        << "\"mode\":\"" << EscapeJson(mode) << "\","
        << "\"adapter_name\":\"" << EscapeJson(adapter_name) << "\","
        << "\"repo_root\":\"" << EscapeJson(repo_root) << "\","
        << "\"bundle_dir\":\"" << EscapeJson(bundle_dir) << "\","
        << "\"status_code\":\"" << StatusCodeReportName(status.code) << "\","
        << "\"status_message\":\"" << EscapeJson(status.ok() ? "OK" : status.message) << "\","
        << "\"complete\":" << (verification.complete ? "true" : "false") << ","
        << "\"probe_compile\":\""
        << (verification.probe_compile_passed ? "PASS" :
            (verification.probe_compile_attempted ? "FAIL" : "SKIPPED")) << "\","
        << "\"runtime_smoke\":\""
        << (verification.runtime_smoke_passed ? "PASS" : "FAIL") << "\","
        << "\"runtime_smoke_cases\":" << verification.runtime_smoke_case_count << ","
        << "\"runtime_events\":" << verification.runtime_smoke_event_count << ","
        << "\"missing_artifacts\":";
    RenderJsonStringArray(verification.missing_artifacts, &out);
    out << ",\"missing_manifest_sections\":";
    RenderJsonStringArray(verification.missing_manifest_sections, &out);
    out << ",\"unchecked_items\":";
    RenderJsonStringArray(verification.unchecked_items, &out);
    out << ",\"summary\":\"" << EscapeJson(verification.summary) << "\""
        << "}";
    return out.str();
}

CheckerStatus WriteVerificationReportIfRequested(
        const std::string &path,
        const std::string &mode,
        const std::string &adapter_name,
        const std::string &repo_root,
        const std::string &bundle_dir,
        const CheckerStatus &status,
        const TraceBundleVerificationResult &verification) {
    if (path.empty()) {
        return CheckerStatus::Ok();
    }
    CheckerStatus mkdir_status = EnsureDirectory(ParentDir(path));
    if (!mkdir_status.ok()) {
        return mkdir_status;
    }
    return WriteTextFile(path, RenderVerificationReportJson(mode, adapter_name,
                                                            repo_root, bundle_dir,
                                                            status,
                                                            verification));
}

CheckerStatus PrepareGeneratedTraceBundleOptions(CliOptions *parsed,
                                                 bool require_output_dir,
                                                 bool saw_output_dir) {
    if (parsed == nullptr) {
        return CheckerStatus::Fail("cli options pointer is null");
    }
    if (parsed->trace_adapter_spec.source_scan_file.empty() &&
        !parsed->repo_root.empty() &&
        !parsed->trace_adapter_spec.source_file.empty() &&
        parsed->trace_adapter_spec.source_file[0] != '/') {
        parsed->trace_adapter_spec.source_scan_file =
            JoinPath(parsed->repo_root, parsed->trace_adapter_spec.source_file);
    }
    CheckerStatus normalize_status =
        NormalizeTraceAdapterSpec(&parsed->trace_adapter_spec);
    if (!normalize_status.ok()) {
        return normalize_status;
    }
    if (parsed->trace_adapter_output.empty()) {
        if (!saw_output_dir) {
            return CheckerStatus::Unsupported(
                "required trace adapter argument: --output or --output-dir");
        }
        parsed->trace_adapter_output =
            JoinPath(parsed->output_dir,
                     parsed->trace_adapter_spec.adapter_name + "_trace_shim.h");
    }
    if (saw_output_dir && parsed->trace_adapter_manifest_output.empty()) {
        parsed->trace_adapter_manifest_output =
            JoinPath(parsed->output_dir,
                     parsed->trace_adapter_spec.adapter_name + "_trace_manifest.json");
    }
    if (saw_output_dir && parsed->trace_runner_output.empty()) {
        parsed->trace_runner_output =
            JoinPath(parsed->output_dir,
                     parsed->trace_adapter_spec.adapter_name + "_trace_runner.cpp");
    }
    if (saw_output_dir && parsed->trace_probe_output.empty()) {
        parsed->trace_probe_output =
            JoinPath(parsed->output_dir,
                     parsed->trace_adapter_spec.adapter_name + "_trace_probe.cpp");
    }
    if (saw_output_dir && parsed->trace_onboarding_output.empty()) {
        parsed->trace_onboarding_output =
            JoinPath(parsed->output_dir,
                     parsed->trace_adapter_spec.adapter_name + "_trace_onboarding.md");
    }
    if (require_output_dir) {
        if (!saw_output_dir) {
            return CheckerStatus::Unsupported(
                "required trace bundle scaffold argument: --output-dir");
        }
        if (parsed->trace_adapter_manifest_output.empty() ||
            parsed->trace_runner_output.empty() ||
            parsed->trace_probe_output.empty() ||
            parsed->trace_onboarding_output.empty()) {
            return CheckerStatus::Unsupported(
                "trace bundle scaffold requires shim, manifest, runner, probe, and onboarding outputs");
        }
        if (parsed->verification_report_output.empty()) {
            parsed->verification_report_output =
                JoinPath(parsed->output_dir, "verification.json");
        }
    }
    if (!parsed->trace_runner_output.empty()) {
        const std::string pascal = PascalName(parsed->trace_adapter_spec.adapter_name);
        if (parsed->trace_runner_spec.function_name.empty()) {
            parsed->trace_runner_spec.function_name = "Run" + pascal + "Trace";
        }
        if (parsed->trace_runner_spec.materializer_name.empty()) {
            parsed->trace_runner_spec.materializer_name = "Materialize" + pascal;
        }
    }
    if (!parsed->trace_runner_output.empty()) {
        if (parsed->trace_runner_spec.function_name.empty() ||
            parsed->trace_runner_spec.materializer_name.empty()) {
            return CheckerStatus::Unsupported(
                "required trace runner arguments: --runner-function, --runner-materializer");
        }
        parsed->trace_runner_spec.shim_include =
            "tilexr/checker/" + BaseName(parsed->trace_adapter_output);
    }
    if (!parsed->trace_probe_output.empty()) {
        parsed->trace_probe_spec.adapter_name = parsed->trace_adapter_spec.adapter_name;
        parsed->trace_probe_spec.shim_include = BaseName(parsed->trace_adapter_output);
        parsed->trace_probe_spec.expected_source_file =
            parsed->trace_adapter_spec.source_file;
        parsed->trace_probe_spec.expected_target_header =
            parsed->trace_adapter_spec.target_header;
    }

    std::string rendered;
    CheckerStatus render_status = RenderTraceAdapter(parsed->trace_adapter_spec, &rendered);
    if (!render_status.ok()) {
        return render_status;
    }
    return CheckerStatus::Ok();
}

}  // namespace

CheckerStatus NormalizeExecutorCliStatus(const RunResult &result) {
    if (result.status.code == CheckerStatusCode::kUnsupported ||
        result.status.code == CheckerStatusCode::kInconclusive ||
        result.status.code == CheckerStatusCode::kInternalError) {
        return result.status;
    }
    return FinalizeCliStatus(result);
}

CheckerStatus ParseCliArgs(int argc, const char *const *argv, CliOptions *options) {
    if (options == nullptr) {
        return CheckerStatus::Fail("cli options pointer is null");
    }

    CliOptions parsed;
    parsed.test_case.op = CollectiveOp::kAllGather;
    parsed.test_case.rank_size = 0;
    parsed.test_case.count = 0;
    parsed.test_case.bs = 0;
    parsed.test_case.h = 0;
    parsed.test_case.top_k = 0;
    parsed.test_case.moe_expert_num = 0;
    parsed.test_case.data_type = TileXR::TILEXR_DATA_TYPE_RESERVED;
    parsed.test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    parsed.test_case.scheduler = SchedulerMode::kSerial;
    parsed.test_case.algorithm = AlgorithmId::kDefault;
    parsed.test_case.server_count = 1;
    parsed.test_case.magic = 0xC001CA5EULL;
    parsed.output_dir = ".";
    parsed.repo_root = ".";

    bool saw_op = false;
    bool saw_rank_size = false;
    bool saw_count = false;
    bool saw_bs = false;
    bool saw_h = false;
    bool saw_top_k = false;
    bool saw_moe_expert_num = false;
    bool saw_datatype = false;
    bool saw_output_dir = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const char *value = nullptr;
        CheckerStatus status;
        if (arg == "--help" || arg == "-h") {
            parsed.show_help = true;
            *options = parsed;
            return CheckerStatus::Ok();
        } else if (arg == "--inject-read-before-copy") {
            parsed.inject_read_before_copy = true;
        } else if (arg == "--inject-peer-user-read") {
            parsed.inject_peer_user_read = true;
        } else if (arg == "--inject-pipe-wait") {
            parsed.inject_pipe_wait = true;
        } else if (arg == "--inject-ep-window-read") {
            parsed.inject_ep_window_read = true;
        } else if (arg == "--scaffold-trace-bundle") {
            parsed.scaffold_trace_bundle = true;
            parsed.generate_trace_adapter = true;
        } else if (arg == "--generate-trace-adapter") {
            parsed.generate_trace_adapter = true;
        } else if (arg == "--strict-source-observation") {
            parsed.trace_adapter_spec.strict_source_observation = true;
        } else if (arg == "--verify-trace-bundle") {
            parsed.verify_trace_bundle = true;
        } else if (arg == "--verify-installed-trace") {
            parsed.verify_installed_trace = true;
        } else if (arg == "--verify-all-installed-traces") {
            parsed.verify_all_installed_traces = true;
        } else if (arg == "--list-installed-traces") {
            parsed.list_installed_traces = true;
        } else if (arg == "--list-capabilities") {
            parsed.list_capabilities = true;
        } else if (arg == "--analyze-trace-source") {
            parsed.analyze_trace_source = true;
        } else if (arg == "--validate-event-trace") {
            parsed.validate_event_trace = true;
        } else if (arg == "--adapter-name") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_adapter_spec.adapter_name = value;
            ++i;
        } else if (arg == "--source-file") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_adapter_spec.source_file = value;
            ++i;
        } else if (arg == "--target-header") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_adapter_spec.target_header = value;
            ++i;
        } else if (arg == "--output") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_adapter_output = value;
            ++i;
        } else if (arg == "--manifest-output") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_adapter_manifest_output = value;
            ++i;
        } else if (arg == "--runner-output") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_runner_output = value;
            ++i;
        } else if (arg == "--probe-output") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_probe_output = value;
            ++i;
        } else if (arg == "--onboarding-output") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_onboarding_output = value;
            ++i;
        } else if (arg == "--runner-function") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_runner_spec.function_name = value;
            ++i;
        } else if (arg == "--runner-materializer") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_runner_spec.materializer_name = value;
            ++i;
        } else if (arg == "--op") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseOp(value, &parsed.test_case.op);
            if (!status.ok()) {
                return status;
            }
            saw_op = true;
            ++i;
        } else if (arg == "--rank-size") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt(value, &parsed.test_case.rank_size);
            if (!status.ok()) {
                return status;
            }
            saw_rank_size = true;
            ++i;
        } else if (arg == "--server-count") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt(value, &parsed.test_case.server_count);
            if (!status.ok()) {
                return status;
            }
            ++i;
        } else if (arg == "--count") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt64(value, &parsed.test_case.count);
            if (!status.ok()) {
                return status;
            }
            saw_count = true;
            ++i;
        } else if (arg == "--bs") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt64(value, &parsed.test_case.bs);
            if (!status.ok()) {
                return status;
            }
            saw_bs = true;
            ++i;
        } else if (arg == "--h") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt64(value, &parsed.test_case.h);
            if (!status.ok()) {
                return status;
            }
            saw_h = true;
            ++i;
        } else if (arg == "--top-k") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt64(value, &parsed.test_case.top_k);
            if (!status.ok()) {
                return status;
            }
            saw_top_k = true;
            ++i;
        } else if (arg == "--moe-expert-num") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseInt64(value, &parsed.test_case.moe_expert_num);
            if (!status.ok()) {
                return status;
            }
            saw_moe_expert_num = true;
            ++i;
        } else if (arg == "--datatype") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseDatatype(value, &parsed.test_case.data_type);
            if (!status.ok()) {
                return status;
            }
            saw_datatype = true;
            ++i;
        } else if (arg == "--reduce-op") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseReduceOp(value, &parsed.test_case.reduce_op);
            if (!status.ok()) {
                return status;
            }
            ++i;
        } else if (arg == "--scheduler") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseScheduler(value, &parsed.test_case.scheduler);
            if (!status.ok()) {
                return status;
            }
            ++i;
        } else if (arg == "--algorithm") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            status = ParseAlgorithm(value, &parsed.test_case.algorithm);
            if (!status.ok()) {
                return status;
            }
            ++i;
        } else if (arg == "--output-dir") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.output_dir = value;
            saw_output_dir = true;
            ++i;
        } else if (arg == "--repo-root") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.repo_root = value;
            ++i;
        } else if (arg == "--verification-report") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.verification_report_output = value;
            ++i;
        } else if (arg == "--capability-report") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.capability_report_output = value;
            ++i;
        } else if (arg == "--trace-analysis-output") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.trace_analysis_output = value;
            ++i;
        } else if (arg == "--event-trace-template-output") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.event_trace_template_output = value;
            ++i;
        } else if (arg == "--event-trace") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.event_trace_input = value;
            ++i;
        } else {
            return CheckerStatus::Unsupported("unknown argument: " + arg);
        }
    }

    if (parsed.validate_event_trace) {
        if (parsed.event_trace_input.empty()) {
            return CheckerStatus::Unsupported(
                "required event trace validation argument: --event-trace");
        }
        if (!saw_output_dir) {
            return CheckerStatus::Unsupported(
                "required event trace validation argument: --output-dir");
        }
        *options = parsed;
        return CheckerStatus::Ok();
    }

    if (parsed.analyze_trace_source) {
        if (parsed.trace_adapter_spec.source_file.empty()) {
            return CheckerStatus::Unsupported(
                "required trace source analysis argument: --source-file");
        }
        CheckerStatus normalize_status =
            NormalizeTraceAdapterSpec(&parsed.trace_adapter_spec);
        if (!normalize_status.ok()) {
            return normalize_status;
        }
        if (!parsed.repo_root.empty() &&
            !parsed.trace_adapter_spec.source_file.empty() &&
            parsed.trace_adapter_spec.source_file[0] != '/') {
            parsed.trace_adapter_spec.source_scan_file =
                JoinPath(parsed.repo_root, parsed.trace_adapter_spec.source_file);
        }
        if (parsed.trace_adapter_output.empty()) {
            parsed.trace_adapter_output =
                JoinPath("tools/checker/shim/tilexr/checker",
                         parsed.trace_adapter_spec.adapter_name + "_trace_shim.h");
        }
        *options = parsed;
        return CheckerStatus::Ok();
    }

    if (parsed.generate_trace_adapter) {
        CheckerStatus prepare_status =
            PrepareGeneratedTraceBundleOptions(&parsed, parsed.scaffold_trace_bundle,
                                               saw_output_dir);
        if (!prepare_status.ok()) {
            return prepare_status;
        }
        *options = parsed;
        return CheckerStatus::Ok();
    }

    if (parsed.verify_trace_bundle) {
        if (parsed.trace_adapter_spec.adapter_name.empty()) {
            return CheckerStatus::Unsupported(
                "required trace bundle verification argument: --adapter-name");
        }
        if (!saw_output_dir) {
            return CheckerStatus::Unsupported(
                "required trace bundle verification argument: --output-dir");
        }
        *options = parsed;
        return CheckerStatus::Ok();
    }

    if (parsed.verify_installed_trace) {
        if (parsed.trace_adapter_spec.adapter_name.empty()) {
            return CheckerStatus::Unsupported(
                "required installed trace verification argument: --adapter-name");
        }
        *options = parsed;
        return CheckerStatus::Ok();
    }

    if (parsed.verify_all_installed_traces || parsed.list_installed_traces ||
        parsed.list_capabilities) {
        *options = parsed;
        return CheckerStatus::Ok();
    }

    if (!saw_op || !saw_rank_size || !saw_datatype) {
        return CheckerStatus::Unsupported(
            "required arguments: --op, --rank-size, --datatype");
    }

    if (parsed.test_case.op == CollectiveOp::kEpDispatch ||
        parsed.test_case.op == CollectiveOp::kEpCombine) {
        if (!saw_bs || !saw_h || !saw_top_k || !saw_moe_expert_num) {
            return CheckerStatus::Unsupported(
                std::string("required ") + ToString(parsed.test_case.op) +
                " arguments: --bs, --h, --top-k, --moe-expert-num");
        }
    } else if (!saw_count) {
        return CheckerStatus::Unsupported(
            "required collective argument: --count");
    }

    CheckerStatus shape_status = ValidateCliRunnableShape(parsed.test_case);
    if (!shape_status.ok()) {
        return shape_status;
    }

    *options = parsed;
    return CheckerStatus::Ok();
}

int ExitCodeFromStatus(const CheckerStatus &status) {
    switch (status.code) {
        case CheckerStatusCode::kOk:
            return 0;
        case CheckerStatusCode::kFail:
            return 1;
        case CheckerStatusCode::kUnsupported:
            return 2;
        case CheckerStatusCode::kInconclusive:
        case CheckerStatusCode::kInternalError:
            return 3;
    }
    return 3;
}

int ReturnTraceGenerationError(const CheckerStatus &status,
                               std::ostream *stderr_stream) {
    if (stderr_stream != nullptr) {
        (*stderr_stream) << status.message << "\n";
    }
    return status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
}

int GenerateTraceBundleFiles(const CliOptions &options,
                             std::ostream *stdout_stream,
                             std::ostream *stderr_stream) {
    CheckerStatus status =
        RejectProductionOutputPath(options.trace_adapter_output, "trace adapter");
    if (!status.ok()) {
        return ReturnTraceGenerationError(status, stderr_stream);
    }
    if (!options.trace_adapter_manifest_output.empty()) {
        status = RejectProductionOutputPath(options.trace_adapter_manifest_output,
                                            "trace adapter manifest");
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    if (!options.trace_runner_output.empty()) {
        status = RejectProductionOutputPath(options.trace_runner_output,
                                            "trace runner skeleton");
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    if (!options.trace_probe_output.empty()) {
        status = RejectProductionOutputPath(options.trace_probe_output,
                                            "trace header probe");
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    if (!options.trace_onboarding_output.empty()) {
        status = RejectProductionOutputPath(options.trace_onboarding_output,
                                            "trace onboarding");
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }

    status = EnsureDirectory(ParentDir(options.trace_adapter_output));
    if (!status.ok()) {
        return ReturnTraceGenerationError(status, stderr_stream);
    }
    if (!options.trace_adapter_manifest_output.empty()) {
        status = EnsureDirectory(ParentDir(options.trace_adapter_manifest_output));
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    if (!options.trace_runner_output.empty()) {
        status = EnsureDirectory(ParentDir(options.trace_runner_output));
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    if (!options.trace_probe_output.empty()) {
        status = EnsureDirectory(ParentDir(options.trace_probe_output));
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    if (!options.trace_onboarding_output.empty()) {
        status = EnsureDirectory(ParentDir(options.trace_onboarding_output));
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }

    std::string adapter_content;
    status = RenderTraceAdapter(options.trace_adapter_spec, &adapter_content);
    if (!status.ok()) {
        return ReturnTraceGenerationError(status, stderr_stream);
    }
    std::string manifest_content;
    if (!options.trace_adapter_manifest_output.empty()) {
        status = RenderTraceAdapterManifest(options.trace_adapter_spec,
                                            options.trace_adapter_output,
                                            &manifest_content);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    std::string runner_content;
    if (!options.trace_runner_output.empty()) {
        status = RenderTraceRunnerSkeleton(options.trace_runner_spec, &runner_content);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    std::string probe_content;
    if (!options.trace_probe_output.empty()) {
        TraceHeaderProbeSpec probe_spec = options.trace_probe_spec;
        TraceAdapterSpec normalized = options.trace_adapter_spec;
        status = NormalizeTraceAdapterSpec(&normalized);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
        if (probe_spec.adapter_name.empty()) {
            probe_spec.adapter_name = normalized.adapter_name;
        }
        if (probe_spec.shim_include.empty()) {
            probe_spec.shim_include = BaseName(options.trace_adapter_output);
        }
        if (probe_spec.expected_source_file.empty()) {
            probe_spec.expected_source_file = normalized.source_file;
        }
        if (probe_spec.expected_target_header.empty()) {
            probe_spec.expected_target_header = normalized.target_header;
        }
        status = RenderTraceHeaderProbeSkeleton(probe_spec, &probe_content);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }
    std::string onboarding_content;
    if (!options.trace_onboarding_output.empty()) {
        status = RenderTraceOnboardingPlan(options.trace_adapter_spec,
                                           options.trace_runner_spec,
                                           options.trace_adapter_output,
                                           options.trace_adapter_manifest_output,
                                           options.trace_runner_output,
                                           &onboarding_content);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
    }

    status = WriteTextFile(options.trace_adapter_output, adapter_content);
    if (!status.ok()) {
        return ReturnTraceGenerationError(status, stderr_stream);
    }
    if (stdout_stream != nullptr) {
        (*stdout_stream) << "generated trace adapter: "
                         << options.trace_adapter_output << "\n";
    }
    if (!options.trace_adapter_manifest_output.empty()) {
        status = WriteTextFile(options.trace_adapter_manifest_output, manifest_content);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << "generated trace adapter manifest: "
                             << options.trace_adapter_manifest_output << "\n";
        }
    }
    if (!options.trace_runner_output.empty()) {
        status = WriteTextFile(options.trace_runner_output, runner_content);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << "generated trace runner skeleton: "
                             << options.trace_runner_output << "\n";
        }
    }
    if (!options.trace_probe_output.empty()) {
        status = WriteTextFile(options.trace_probe_output, probe_content);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << "generated trace header probe: "
                             << options.trace_probe_output << "\n";
        }
    }
    if (!options.trace_onboarding_output.empty()) {
        status = WriteTextFile(options.trace_onboarding_output, onboarding_content);
        if (!status.ok()) {
            return ReturnTraceGenerationError(status, stderr_stream);
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << "generated trace onboarding plan: "
                             << options.trace_onboarding_output << "\n";
        }
    }
    return 0;
}

CheckerStatus WriteReportFiles(const std::string &output_dir,
                               const CheckerCase &test_case,
                               const RunResult &result,
                               const EventLog &events,
                               ReportPaths *paths) {
    return WriteReportFiles(output_dir, test_case, result, events, std::string(), paths);
}

CheckerStatus WriteReportFiles(const std::string &output_dir,
                               const CheckerCase &test_case,
                               const RunResult &result,
                               const EventLog &events,
                               const std::string &source_root,
                               ReportPaths *paths) {
    return WriteReportFiles(output_dir, test_case, result, events, source_root,
                            "checker_run", std::string(), paths);
}

CheckerStatus WriteReportFiles(const std::string &output_dir,
                               const CheckerCase &test_case,
                               const RunResult &result,
                               const EventLog &events,
                               const std::string &source_root,
                               const std::string &mode,
                               const std::string &event_trace_input,
                               ReportPaths *paths) {
    if (paths == nullptr) {
        return CheckerStatus::Fail("report paths pointer is null");
    }

    CheckerStatus mkdir_status = EnsureDirectory(output_dir);
    if (!mkdir_status.ok()) {
        return mkdir_status;
    }

    paths->summary_txt = JoinPath(output_dir, "summary.txt");
    paths->findings_json = JoinPath(output_dir, "findings.json");
    paths->events_jsonl = JoinPath(output_dir, "events.jsonl");
    paths->checker_report_json = JoinPath(output_dir, "checker_report.json");

    FindingSet report_findings = result.findings;
    if (!result.mismatches.empty() &&
        !HasFindingKind(report_findings, FindingKind::kOutputMismatch) &&
        !HasFindingKind(report_findings, FindingKind::kUnsupportedApi)) {
        report_findings =
            MergeFindings(report_findings, CheckOutputMismatches(result.mismatches));
    }
    report_findings = AttachOutputProducerLocations(report_findings, events);

    const size_t event_count = events.events().size();
    const std::string timeline_summary =
        BuildTimelineSummary(events, report_findings.TopFinding());
    CheckerStatus status = WriteTextFile(paths->summary_txt,
                                         RenderSummary(test_case, result.status, report_findings,
                                                       result.mismatches.size(), event_count,
                                                       timeline_summary));
    if (!status.ok()) {
        return status;
    }
    status = WriteTextFile(paths->findings_json, RenderFindingsJson(report_findings));
    if (!status.ok()) {
        return status;
    }
    status = WriteTextFile(paths->events_jsonl, RenderEventsJsonl(events));
    if (!status.ok()) {
        return status;
    }

    std::ostringstream report;
    const AlgorithmSelectionExplanation selection =
        ExplainAlgorithmSelection(test_case);
    report << "{"
           << "\"mode\":\"" << EscapeJson(mode) << "\",";
    if (!event_trace_input.empty()) {
        report << "\"event_trace_input\":\"" << EscapeJson(event_trace_input) << "\",";
    }
    report
           << "\"case\":\"" << EscapeJson(DescribeCase(test_case)) << "\","
           << "\"algorithm\":\"" << EscapeJson(ToString(test_case.algorithm)) << "\","
           << "\"source_file\":\"" << EscapeJson(CaseSourceFile(test_case)) << "\","
           << "\"algorithm_selection\":{"
           << "\"algorithm\":\"" << EscapeJson(ToString(selection.algorithm)) << "\","
           << "\"eligible\":" << (selection.eligible ? "true" : "false") << ","
           << "\"reason\":\"" << EscapeJson(selection.reason) << "\","
           << "\"source_file\":\"" << EscapeJson(selection.source_file) << "\""
           << "},"
           << "\"status\":\"" << StatusLabel(result.status.code) << "\","
           << "\"topology\":{"
           << "\"rank_size\":" << test_case.rank_size << ","
           << "\"server_count\":" << test_case.server_count << ","
           << "\"local_rank_size\":" << test_case.rank_size / test_case.server_count
           << "},"
           << "\"artifacts\":{"
           << "\"summary_txt\":\"" << EscapeJson(BaseName(paths->summary_txt)) << "\","
           << "\"findings_json\":\"" << EscapeJson(BaseName(paths->findings_json)) << "\","
           << "\"events_jsonl\":\"" << EscapeJson(BaseName(paths->events_jsonl)) << "\","
           << "\"checker_report_json\":\"" << EscapeJson(BaseName(paths->checker_report_json))
           << "\"},"
           << "\"artifact_paths\":{"
           << "\"summary_txt\":\"" << EscapeJson(paths->summary_txt) << "\","
           << "\"findings_json\":\"" << EscapeJson(paths->findings_json) << "\","
           << "\"events_jsonl\":\"" << EscapeJson(paths->events_jsonl) << "\","
           << "\"checker_report_json\":\"" << EscapeJson(paths->checker_report_json)
           << "\"},"
           << "\"counts\":{"
           << "\"events\":" << event_count << ","
           << "\"findings\":" << report_findings.findings().size() << ","
           << "\"mismatches\":" << result.mismatches.size() << "},"
           << "\"ep_dispatch_layout\":";
    if (test_case.op == CollectiveOp::kEpDispatch) {
        EpDispatchWindowLayout window_layout;
        const CheckerStatus window_status =
            ComputeEpDispatchWindowLayout(test_case, &window_layout);
        report << "{"
               << "\"payload_bytes\":" << EpDispatchPayloadBytes(test_case) << ","
               << "\"assist\":{"
               << "\"offset\":" << EpDispatchMetadataOffset(
                      test_case, EpDispatchMetadataRole::kAssist)
               << ",\"bytes\":" << EpDispatchMetadataBytes(
                      test_case, EpDispatchMetadataRole::kAssist)
               << "},"
               << "\"recv_counts\":{"
               << "\"offset\":" << EpDispatchMetadataOffset(
                      test_case, EpDispatchMetadataRole::kRecvCounts)
               << ",\"bytes\":" << EpDispatchMetadataBytes(
                      test_case, EpDispatchMetadataRole::kRecvCounts)
               << "},"
               << "\"expert_token_nums\":{"
               << "\"offset\":" << EpDispatchMetadataOffset(
                      test_case, EpDispatchMetadataRole::kExpertTokenNums)
               << ",\"bytes\":" << EpDispatchMetadataBytes(
                      test_case, EpDispatchMetadataRole::kExpertTokenNums)
               << "},"
               << "\"window\":";
        if (window_status.ok()) {
            report << "{"
                   << "\"rank_size\":" << window_layout.rank_size << ","
                   << "\"local_expert_num\":" << window_layout.local_expert_num << ","
                   << "\"dtype_bytes\":" << window_layout.dtype_bytes << ","
                   << "\"max_routes_per_src\":" << window_layout.max_routes_per_src << ","
                   << "\"row_bytes\":" << window_layout.row_bytes << ","
                   << "\"payload_bytes_per_slot\":" << window_layout.payload_bytes_per_slot << ","
                   << "\"assist_bytes_per_slot\":" << window_layout.assist_bytes_per_slot << ","
                   << "\"slot_bytes\":" << window_layout.slot_bytes << ","
                   << "\"total_bytes\":" << window_layout.total_bytes
                   << "}";
        } else {
            report << "null";
        }
        report
               << "}";
    } else {
        report << "null";
    }
    report << ","
           << "\"top_finding\":";

    const Finding *top = report_findings.TopFinding();
    if (top == nullptr) {
        report << "null";
    } else {
        report << "{"
               << "\"kind\":\"" << ToString(top->kind) << "\","
               << "\"severity\":\"" << ToString(top->severity) << "\","
               << "\"message\":\"" << EscapeJson(top->message) << "\","
               << "\"next_action\":\"" << EscapeJson(top->next_action) << "\","
               << "\"event_log_id\":" << top->event_log_id << ","
               << "\"rank\":" << top->rank << ","
               << "\"peer_rank\":" << top->peer_rank << ","
               << "\"server\":" << top->server << ","
               << "\"peer_server\":" << top->peer_server << ","
               << "\"server_scope\":\"" << ServerScopeName(top->server, top->peer_server) << "\","
               << "\"core\":" << top->core << ","
               << "\"pipe\":" << top->pipe << ","
               << "\"event_id\":" << top->event_id << ","
               << "\"slot\":" << top->slot << ","
               << "\"buffer_role\":\"" << BufferReportName(top->buffer_role) << "\","
               << "\"offset\":" << top->offset << ","
               << "\"bytes\":" << top->bytes << ","
               << "\"source_file\":\"" << EscapeJson(top->source_file) << "\","
               << "\"source_line\":" << top->source_line << ","
               << "\"has_expected_actual\":" << (top->has_expected_actual ? "true" : "false") << ","
               << "\"expected\":" << top->expected << ","
               << "\"actual\":" << top->actual << ","
               << "\"context\":\"" << EscapeJson(top->context) << "\","
               << "\"source_excerpt\":";
        RenderSourceExcerptJson(top->source_file, top->source_line,
                                source_root, &report);
        report << ","
               << "\"event_context\":";
        RenderEventContextJson(events, top->event_log_id, &report);
        report << ",\"timeline\":";
        RenderTimelineJson(events, *top, &report);
        report << "}";
    }
    report << "}";

    return WriteTextFile(paths->checker_report_json, report.str());
}

int RunCheckerCli(const CliOptions &options, std::ostream *stdout_stream,
                  std::ostream *stderr_stream) {
    if (options.show_help) {
        PrintUsage(stdout_stream);
        return 0;
    }
    if (options.analyze_trace_source) {
        TraceSourceAnalysis analysis;
        CheckerStatus status =
            AnalyzeTraceSource(options.trace_adapter_spec,
                               options.trace_adapter_output, &analysis);
        if (!status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << status.message << "\n";
            }
            return status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        std::string text;
        status = RenderTraceSourceAnalysisText(analysis, &text);
        if (!status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << status.message << "\n";
            }
            return status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        if (!options.trace_analysis_output.empty()) {
            CheckerStatus path_status =
                RejectProductionOutputPath(options.trace_analysis_output,
                                           "trace source analysis");
            if (!path_status.ok()) {
                if (stderr_stream != nullptr) {
                    (*stderr_stream) << path_status.message << "\n";
                }
                return path_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
            }
            std::string json;
            status = RenderTraceSourceAnalysisJson(analysis, &json);
            if (!status.ok()) {
                if (stderr_stream != nullptr) {
                    (*stderr_stream) << status.message << "\n";
                }
                return status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
            }
            status = WriteTextFile(options.trace_analysis_output, json);
            if (!status.ok()) {
                if (stderr_stream != nullptr) {
                    (*stderr_stream) << status.message << "\n";
                }
                return status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
            }
        }
        if (!options.event_trace_template_output.empty()) {
            CheckerStatus path_status =
                RejectProductionOutputPath(options.event_trace_template_output,
                                           "event trace template");
            if (!path_status.ok()) {
                if (stderr_stream != nullptr) {
                    (*stderr_stream) << path_status.message << "\n";
                }
                return path_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
            }
            CheckerStatus dir_status =
                EnsureDirectory(ParentDir(options.event_trace_template_output));
            if (!dir_status.ok()) {
                if (stderr_stream != nullptr) {
                    (*stderr_stream) << dir_status.message << "\n";
                }
                return dir_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
            }
            const EventLog template_events = BuildTraceEventTemplate(analysis);
            status = WriteTextFile(options.event_trace_template_output,
                                   RenderEventsJsonl(template_events));
            if (!status.ok()) {
                if (stderr_stream != nullptr) {
                    (*stderr_stream) << status.message << "\n";
                }
                return status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
            }
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << text;
            if (!options.trace_analysis_output.empty()) {
                (*stdout_stream) << "analysis report: "
                                 << options.trace_analysis_output << "\n";
            }
            if (!options.event_trace_template_output.empty()) {
                (*stdout_stream) << "event trace template: "
                                 << options.event_trace_template_output << "\n";
            }
        }
        return 0;
    }
    if (options.generate_trace_adapter || options.scaffold_trace_bundle) {
        const int generate_exit =
            GenerateTraceBundleFiles(options, stdout_stream, stderr_stream);
        if (generate_exit != 0 || !options.scaffold_trace_bundle) {
            return generate_exit;
        }

        TraceBundleVerificationResult verification;
        const std::string repo_root = options.repo_root.empty() ? "." : options.repo_root;
        CheckerStatus status = VerifyTraceBundle(options.output_dir,
                                                 options.trace_adapter_spec.adapter_name,
                                                 repo_root,
                                                 &verification);
        CheckerStatus report_status = WriteVerificationReportIfRequested(
            options.verification_report_output, "trace_bundle_scaffold",
            options.trace_adapter_spec.adapter_name, repo_root, options.output_dir,
            status, verification);
        if (!report_status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << report_status.message << "\n";
            }
            return report_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << "scaffold verification: "
                             << (verification.complete ? "complete" : "incomplete")
                             << "\n"
                             << verification.summary << "\n"
                             << "verification report: "
                             << options.verification_report_output << "\n";
        }
        return 0;
    }

    if (options.verify_trace_bundle) {
        TraceBundleVerificationResult verification;
        const std::string repo_root = options.repo_root.empty() ? "." : options.repo_root;
        CheckerStatus status = VerifyTraceBundle(options.output_dir,
                                                 options.trace_adapter_spec.adapter_name,
                                                 repo_root,
                                                 &verification);
        CheckerStatus report_status = WriteVerificationReportIfRequested(
            options.verification_report_output, "trace_bundle",
            options.trace_adapter_spec.adapter_name, repo_root, options.output_dir,
            status, verification);
        if (!report_status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << report_status.message << "\n";
            }
            return report_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << verification.summary << "\n";
        }
        if (status.ok()) {
            return 0;
        }
        return status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
    }

    if (options.validate_event_trace) {
        EventLog events;
        CheckerStatus read_status =
            ReadEventTraceJsonl(options.event_trace_input, &events);
        if (!read_status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << read_status.message << "\n";
            }
            return read_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }

        RunResult result;
        result.findings = CheckOrdering(events);
        std::vector<RequiredEventCoverage> required_coverage;
        CheckerStatus coverage_status =
            ReadRequiredEventCoverage(options.repo_root.empty() ? "." : options.repo_root,
                                      options.trace_adapter_spec.adapter_name,
                                      &required_coverage);
        if (!coverage_status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << coverage_status.message << "\n";
            }
            return coverage_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        result.findings = MergeFindings(
            result.findings,
            CheckRequiredEventCoverage(events, required_coverage));
        result.event_count = events.events().size();
        result.status = FinalizeCliStatus(result);

        CheckerCase trace_case;
        trace_case.op = CollectiveOp::kEpDispatch;
        trace_case.rank_size = 1;
        trace_case.server_count = 1;
        trace_case.bs = 1;
        trace_case.h = 1;
        trace_case.top_k = 1;
        trace_case.moe_expert_num = 1;
        trace_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
        trace_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
        trace_case.scheduler = SchedulerMode::kSerial;
        trace_case.algorithm = AlgorithmId::kDefault;
        trace_case.magic = 0;

        ReportPaths paths;
        CheckerStatus write_status = WriteReportFiles(
            options.output_dir, trace_case, result, events,
            options.repo_root.empty() ? "." : options.repo_root,
            "event_trace_validation", options.event_trace_input, &paths);
        if (!write_status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << write_status.message << "\n";
            }
            return write_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << RenderSummary(trace_case, result.status, result.findings,
                                              result.mismatches.size(),
                                              events.events().size());
        }
        return ExitCodeFromStatus(result.status);
    }

    if (options.list_installed_traces) {
        const std::string repo_root = options.repo_root.empty() ? "." : options.repo_root;
        const std::vector<std::string> adapters = ListInstalledTraceAdapters(repo_root);
        if (adapters.empty()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << "no installed trace manifests found under "
                                 << JoinPath(repo_root, "tools/checker/installed_traces")
                                 << "\n";
            }
            return 2;
        }
        if (stdout_stream != nullptr) {
            for (size_t i = 0; i < adapters.size(); ++i) {
                (*stdout_stream) << adapters[i] << "\n";
            }
        }
        return 0;
    }

    if (options.verify_all_installed_traces) {
        const std::string repo_root = options.repo_root.empty() ? "." : options.repo_root;
        const std::vector<std::string> adapters = ListInstalledTraceAdapters(repo_root);
        std::vector<InstalledTraceBatchItem> items;
        CheckerStatus overall_status = CheckerStatus::Ok();
        if (adapters.empty()) {
            overall_status = CheckerStatus::Unsupported(
                "no installed trace manifests found under " +
                JoinPath(repo_root, "tools/checker/installed_traces"));
        }
        for (size_t i = 0; i < adapters.size(); ++i) {
            InstalledTraceBatchItem item;
            item.adapter_name = adapters[i];
            item.status = VerifyInstalledTraceAlgorithm(repo_root, adapters[i],
                                                        &item.verification);
            if (!item.status.ok() && overall_status.ok()) {
                overall_status = item.status;
            }
            items.push_back(item);
            if (stdout_stream != nullptr) {
                (*stdout_stream) << adapters[i] << ": "
                                 << (item.verification.complete ? "complete" : "incomplete")
                                 << "\n";
            }
        }
        CheckerStatus report_status = WriteInstalledTraceInventoryReportIfRequested(
            options.verification_report_output, repo_root, items, overall_status);
        if (!report_status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << report_status.message << "\n";
            }
            return report_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        if (overall_status.ok()) {
            return 0;
        }
        return overall_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
    }

    if (options.list_capabilities) {
        const std::string repo_root = options.repo_root.empty() ? "." : options.repo_root;
        const std::vector<CapabilityItem> items = BuildCapabilityInventory(repo_root);
        CheckerStatus report_status = WriteCapabilityReportIfRequested(
            options.capability_report_output, repo_root, items);
        if (!report_status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << report_status.message << "\n";
            }
            return report_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        if (stdout_stream != nullptr) {
            for (size_t i = 0; i < items.size(); ++i) {
                (*stdout_stream) << items[i].op << "/" << CapabilityDisplayName(items[i])
                                 << ": "
                                 << items[i].status << " (" << items[i].mode << ")"
                                 << "\n";
            }
        }
        return 0;
    }

    if (options.verify_installed_trace) {
        TraceBundleVerificationResult verification;
        const std::string repo_root = options.repo_root.empty() ? "." : options.repo_root;
        CheckerStatus status = VerifyInstalledTraceAlgorithm(
            repo_root, options.trace_adapter_spec.adapter_name, &verification);
        CheckerStatus report_status = WriteVerificationReportIfRequested(
            options.verification_report_output, "installed_trace",
            options.trace_adapter_spec.adapter_name, repo_root, std::string(),
            status, verification);
        if (!report_status.ok()) {
            if (stderr_stream != nullptr) {
                (*stderr_stream) << report_status.message << "\n";
            }
            return report_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
        }
        if (stdout_stream != nullptr) {
            (*stdout_stream) << verification.summary << "\n";
        }
        if (status.ok()) {
            return 0;
        }
        return status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
    }

    size_t input_bytes = 0;
    size_t output_bytes = 0;
    size_t comm_data_bytes = 0;
    if (options.test_case.op == CollectiveOp::kEpDispatch) {
        const size_t element_size = ElementSize(options.test_case.data_type);
        input_bytes = static_cast<size_t>(options.test_case.bs * options.test_case.h) *
                      element_size;
        output_bytes = EpDispatchOutputBytes(options.test_case);
        comm_data_bytes = output_bytes;
    } else if (options.test_case.op == CollectiveOp::kEpCombine) {
        input_bytes = EpCombineInputBytes(options.test_case);
        output_bytes = EpCombineOutputBytes(options.test_case);
        comm_data_bytes = input_bytes;
    } else {
        input_bytes = static_cast<size_t>(options.test_case.count) *
                      ElementSize(options.test_case.data_type);
        const size_t output_elements =
            options.test_case.op == CollectiveOp::kAllGather
                ? static_cast<size_t>(options.test_case.rank_size) *
                      static_cast<size_t>(options.test_case.count)
                : static_cast<size_t>(options.test_case.count);
        output_bytes = output_elements * ElementSize(options.test_case.data_type);
        comm_data_bytes = input_bytes;
    }

    RankWorld world = RankWorld::Create(options.test_case.rank_size, input_bytes, output_bytes,
                                        comm_data_bytes);
    world.ConfigureServers(options.test_case.server_count);
    CollectiveExecutor executor;
    RunResult result = executor.Run(&world, options.test_case);
    result.status = NormalizeExecutorCliStatus(result);
    if (result.status.code == CheckerStatusCode::kInconclusive ||
        result.status.code == CheckerStatusCode::kInternalError) {
        if (stderr_stream != nullptr) {
            (*stderr_stream) << result.status.message << "\n";
        }
        return 3;
    }

    if (options.inject_read_before_copy) {
        InjectReadBeforeCopy(&world);
    }
    if (options.inject_peer_user_read) {
        InjectPeerUserRead(&world);
    }
    if (options.inject_pipe_wait) {
        InjectPipeWait(&world);
    }
    if (options.inject_ep_window_read) {
        InjectEpWindowRead(&world);
    }
    if (options.inject_read_before_copy || options.inject_peer_user_read ||
        options.inject_pipe_wait || options.inject_ep_window_read) {
        result.findings = MergeFindings(result.findings, CheckOrdering(world.events()));
        result.event_count = world.events().events().size();
        result.status = FinalizeCliStatus(result);
    }

    ReportPaths paths;
    CheckerStatus write_status =
        WriteReportFiles(options.output_dir, options.test_case, result, world.events(),
                         options.repo_root.empty() ? "." : options.repo_root, &paths);
    if (!write_status.ok()) {
        if (stderr_stream != nullptr) {
            (*stderr_stream) << write_status.message << "\n";
        }
        return write_status.code == CheckerStatusCode::kUnsupported ? 2 : 3;
    }

    if (stdout_stream != nullptr) {
        (*stdout_stream) << RenderSummary(options.test_case, result.status, result.findings,
                                          result.mismatches.size(),
                                          world.events().events().size());
    }

    return ExitCodeFromStatus(result.status);
}

}  // namespace checker
}  // namespace tilexr
