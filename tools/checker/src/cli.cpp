#include "tilexr/checker/cli.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include "tilexr/checker/diagnostics.h"
#include "tilexr/checker/oracle.h"
#include "tilexr/checker/report.h"
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
    return CheckerStatus::Unsupported("unsupported datatype: " + text);
}

CheckerStatus ParseReduceOp(const std::string &text, TileXR::TileXRReduceOp *reduce_op) {
    if (text == "sum") {
        *reduce_op = TileXR::TILEXR_REDUCE_SUM;
        return CheckerStatus::Ok();
    }
    return CheckerStatus::Unsupported("unsupported reduce-op: " + text);
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

CheckerStatus FinalizeCliStatus(const RunResult &result) {
    if (HasFindingKind(result.findings, FindingKind::kUnsupportedApi)) {
        return CheckerStatus::Unsupported("checker encountered unsupported case");
    }
    if (!result.findings.findings().empty() || !result.mismatches.empty()) {
        return CheckerStatus::Fail("checker detected findings or mismatches");
    }
    return result.status.ok() ? CheckerStatus::Ok() : result.status;
}

void InjectReadBeforeCopy(EventLog *events) {
    Event event;
    event.kind = EventKind::kRead;
    event.rank = 1;
    event.peer_rank = 0;
    event.buffer_role = BufferRole::kCommData;
    event.slot = 99;
    event.offset = 0;
    event.bytes = sizeof(int32_t);
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = "cli injected read before copy";
    events->Add(event);
}

void InjectPeerUserRead(EventLog *events) {
    Event event;
    event.kind = EventKind::kRead;
    event.rank = 0;
    event.peer_rank = 1;
    event.buffer_role = BufferRole::kUserInput;
    event.slot = 0;
    event.offset = 0;
    event.bytes = sizeof(int32_t);
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = "cli injected peer user read";
    events->Add(event);
}

void PrintUsage(std::ostream *stream) {
    if (stream == nullptr) {
        return;
    }
    (*stream)
        << "Usage: tilexr_checker --op <allgather|allreduce> --rank-size <n> --count <n> "
           "--datatype int32 [--reduce-op sum] [--scheduler <serial|round_robin>] "
           "[--output-dir <dir>] [--inject-read-before-copy] [--inject-peer-user-read]\n";
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
    parsed.test_case.data_type = TileXR::TILEXR_DATA_TYPE_RESERVED;
    parsed.test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    parsed.test_case.scheduler = SchedulerMode::kSerial;
    parsed.test_case.magic = 0xC001CA5EULL;
    parsed.output_dir = ".";

    bool saw_op = false;
    bool saw_rank_size = false;
    bool saw_count = false;
    bool saw_datatype = false;

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
        } else if (arg == "--output-dir") {
            status = RequireValue(argc, argv, i, &value);
            if (!status.ok()) {
                return status;
            }
            parsed.output_dir = value;
            ++i;
        } else {
            return CheckerStatus::Unsupported("unknown argument: " + arg);
        }
    }

    if (!saw_op || !saw_rank_size || !saw_count || !saw_datatype) {
        return CheckerStatus::Unsupported(
            "required arguments: --op, --rank-size, --count, --datatype");
    }

    CheckerStatus case_status = ValidateCase(parsed.test_case);
    if (!case_status.ok()) {
        return case_status;
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

CheckerStatus WriteReportFiles(const std::string &output_dir,
                               const CheckerCase &test_case,
                               const RunResult &result,
                               const EventLog &events,
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

    const size_t event_count = events.events().size();
    CheckerStatus status = WriteTextFile(paths->summary_txt,
                                         RenderSummary(test_case, result.status, result.findings,
                                                       result.mismatches.size(), event_count));
    if (!status.ok()) {
        return status;
    }
    status = WriteTextFile(paths->findings_json, RenderFindingsJson(result.findings));
    if (!status.ok()) {
        return status;
    }
    status = WriteTextFile(paths->events_jsonl, RenderEventsJsonl(events));
    if (!status.ok()) {
        return status;
    }

    std::ostringstream report;
    report << "{"
           << "\"case\":\"" << EscapeJson(DescribeCase(test_case)) << "\","
           << "\"status\":\"" << StatusLabel(result.status.code) << "\","
           << "\"artifacts\":{"
           << "\"summary_txt\":\"" << EscapeJson(BaseName(paths->summary_txt)) << "\","
           << "\"findings_json\":\"" << EscapeJson(BaseName(paths->findings_json)) << "\","
           << "\"events_jsonl\":\"" << EscapeJson(BaseName(paths->events_jsonl)) << "\","
           << "\"checker_report_json\":\"" << EscapeJson(BaseName(paths->checker_report_json))
           << "\"},"
           << "\"counts\":{"
           << "\"events\":" << event_count << ","
           << "\"findings\":" << result.findings.findings().size() << ","
           << "\"mismatches\":" << result.mismatches.size() << "},"
           << "\"top_finding\":";

    const Finding *top = result.findings.TopFinding();
    if (top == nullptr) {
        report << "null";
    } else {
        report << "{"
               << "\"kind\":\"" << ToString(top->kind) << "\","
               << "\"severity\":\"" << ToString(top->severity) << "\","
               << "\"message\":\"" << EscapeJson(top->message) << "\","
               << "\"next_action\":\"" << EscapeJson(top->next_action) << "\","
               << "\"rank\":" << top->rank << ","
               << "\"peer_rank\":" << top->peer_rank << ","
               << "\"buffer_role\":\"" << ToString(top->buffer_role) << "\","
               << "\"offset\":" << top->offset << ","
               << "\"bytes\":" << top->bytes
               << "}";
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

    const size_t input_bytes = static_cast<size_t>(options.test_case.count) *
                               ElementSize(options.test_case.data_type);
    const size_t output_elements =
        options.test_case.op == CollectiveOp::kAllGather
            ? static_cast<size_t>(options.test_case.rank_size) *
                  static_cast<size_t>(options.test_case.count)
            : static_cast<size_t>(options.test_case.count);
    const size_t output_bytes = output_elements * ElementSize(options.test_case.data_type);

    RankWorld world = RankWorld::Create(options.test_case.rank_size, input_bytes, output_bytes,
                                        input_bytes);
    CollectiveExecutor executor;
    RunResult result = executor.Run(&world, options.test_case);
    result.status = NormalizeExecutorCliStatus(result);
    if (result.status.code == CheckerStatusCode::kUnsupported) {
        if (stderr_stream != nullptr) {
            (*stderr_stream) << result.status.message << "\n";
        }
        return 2;
    }
    if (result.status.code == CheckerStatusCode::kInconclusive ||
        result.status.code == CheckerStatusCode::kInternalError) {
        if (stderr_stream != nullptr) {
            (*stderr_stream) << result.status.message << "\n";
        }
        return 3;
    }

    if (options.inject_read_before_copy) {
        InjectReadBeforeCopy(&world.events());
    }
    if (options.inject_peer_user_read) {
        InjectPeerUserRead(&world.events());
    }
    if (options.inject_read_before_copy || options.inject_peer_user_read) {
        result.findings = MergeFindings(result.findings, CheckOrdering(world.events()));
        result.event_count = world.events().events().size();
        result.status = FinalizeCliStatus(result);
    }

    ReportPaths paths;
    CheckerStatus write_status =
        WriteReportFiles(options.output_dir, options.test_case, result, world.events(), &paths);
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
