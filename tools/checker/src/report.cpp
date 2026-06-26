#include "tilexr/checker/report.h"

#include <sstream>
#include <string>

#include "tilexr/checker/types.h"

namespace tilexr {
namespace checker {

namespace {

std::string BufferSummaryName(BufferRole role) {
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

const char *EventKindString(EventKind kind) {
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

const char *SummaryStatusLabel(const CheckerStatus &status, bool has_problems) {
    if (status.code == CheckerStatusCode::kUnsupported) {
        return "UNSUPPORTED";
    }
    if (status.code == CheckerStatusCode::kInconclusive) {
        return "INCONCLUSIVE";
    }
    if (status.code == CheckerStatusCode::kInternalError) {
        return "ERROR";
    }
    return has_problems ? "FAIL" : "PASS";
}

const char *ServerScopeName(int server, int peer_server) {
    if (server < 0 || peer_server < 0) {
        return "unknown";
    }
    return server == peer_server ? "same_server" : "cross_server";
}

}  // namespace

std::string RenderSummary(const CheckerCase &test_case,
                          const CheckerStatus &status,
                          const FindingSet &findings,
                          size_t mismatch_count,
                          size_t event_count,
                          const std::string &timeline_summary) {
    const bool has_problems =
        !status.ok() || mismatch_count != 0 || !findings.findings().empty();

    std::ostringstream out;
    out << "checker: " << SummaryStatusLabel(status, has_problems) << "\n";
    out << "case: " << DescribeCase(test_case) << "\n";
    out << "algorithm: " << ToString(test_case.algorithm) << "\n";
    const std::string source_file = CaseSourceFile(test_case);
    if (!source_file.empty()) {
        out << "source: " << source_file << "\n";
    }
    const Finding *top = findings.TopFinding();
    if (top != nullptr) {
        out << "top finding: " << ToString(top->kind) << "\n";
        out << "rank: " << top->rank << "\n";
        out << "peer rank: " << top->peer_rank << "\n";
        out << "server: " << top->server << "\n";
        out << "peer server: " << top->peer_server << "\n";
        out << "server scope: "
            << ServerScopeName(top->server, top->peer_server) << "\n";
        out << "event id: " << top->event_log_id << "\n";
        out << "source: "
            << (top->source_file.empty() ? "<unknown>" : top->source_file)
            << ":" << top->source_line << "\n";
        out << "buffer: " << BufferSummaryName(top->buffer_role) << "\n";
        out << "core: " << top->core << "\n";
        out << "slot: " << top->slot << "\n";
        out << "offset: " << top->offset << "\n";
        out << "bytes: " << top->bytes << "\n";
        if (top->has_expected_actual) {
            out << "expected: " << top->expected << "\n";
            out << "actual: " << top->actual << "\n";
            out << "context: " << top->context << "\n";
        }
        if (!timeline_summary.empty()) {
            out << timeline_summary;
        }
        out << "next action: " << top->next_action << "\n";
    } else {
        out << "top finding: NONE\n";
        out << "rank: -1\n";
        out << "peer rank: -1\n";
        out << "server: -1\n";
        out << "peer server: -1\n";
        out << "server scope: unknown\n";
        out << "event id: 0\n";
        out << "source: <unknown>:0\n";
        out << "buffer: Metadata\n";
        out << "core: -1\n";
        out << "slot: -1\n";
        out << "offset: 0\n";
        out << "bytes: 0\n";
        out << "expected: 0\n";
        out << "actual: 0\n";
        out << "context: \n";
        out << "next action: "
            << (mismatch_count != 0 ? "Inspect findings.json for output mismatch details."
                                    : "No action required.")
            << "\n";
    }
    out << "mismatches: " << mismatch_count << "\n";
    out << "events: " << event_count << "\n";
    return out.str();
}

std::string RenderSummary(const CheckerCase &test_case,
                          const FindingSet &findings,
                          size_t event_count) {
    return RenderSummary(test_case, findings.findings().empty() ? CheckerStatus::Ok()
                                                                : CheckerStatus::Fail("findings"),
                         findings, 0, event_count);
}

std::string RenderFindingsJson(const FindingSet &findings) {
    std::ostringstream out;
    out << "[";
    const std::vector<Finding> &items = findings.findings();
    for (size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const Finding &finding = items[i];
        out << "{"
            << "\"id\":\"" << EscapeJson(finding.id) << "\","
            << "\"kind\":\"" << ToString(finding.kind) << "\","
            << "\"severity\":\"" << ToString(finding.severity) << "\","
            << "\"message\":\"" << EscapeJson(finding.message) << "\","
            << "\"next_action\":\"" << EscapeJson(finding.next_action) << "\","
            << "\"event_log_id\":" << finding.event_log_id << ","
            << "\"rank\":" << finding.rank << ","
            << "\"peer_rank\":" << finding.peer_rank << ","
            << "\"server\":" << finding.server << ","
            << "\"peer_server\":" << finding.peer_server << ","
            << "\"server_scope\":\""
            << ServerScopeName(finding.server, finding.peer_server) << "\","
            << "\"core\":" << finding.core << ","
            << "\"pipe\":" << finding.pipe << ","
            << "\"event_id\":" << finding.event_id << ","
            << "\"slot\":" << finding.slot << ","
            << "\"buffer_role\":\"" << BufferSummaryName(finding.buffer_role) << "\","
            << "\"offset\":" << finding.offset << ","
            << "\"bytes\":" << finding.bytes << ","
            << "\"source_file\":\"" << EscapeJson(finding.source_file) << "\","
            << "\"source_line\":" << finding.source_line << ","
            << "\"has_expected_actual\":" << (finding.has_expected_actual ? "true" : "false") << ","
            << "\"expected\":" << finding.expected << ","
            << "\"actual\":" << finding.actual << ","
            << "\"context\":\"" << EscapeJson(finding.context) << "\","
            << "\"confidence\":" << finding.confidence
            << "}";
    }
    out << "]";
    return out.str();
}

std::string RenderEventsJsonl(const EventLog &events) {
    std::ostringstream out;
    const std::vector<Event> &items = events.events();
    for (size_t i = 0; i < items.size(); ++i) {
        const Event &event = items[i];
        out << "{"
            << "\"id\":" << event.id << ","
            << "\"kind\":\"" << EventKindString(event.kind) << "\","
            << "\"rank\":" << event.rank << ","
            << "\"peer_rank\":" << event.peer_rank << ","
            << "\"server\":" << event.server << ","
            << "\"peer_server\":" << event.peer_server << ","
            << "\"core\":" << event.core << ","
            << "\"pipe\":" << event.pipe << ","
            << "\"event_id\":" << event.event_id << ","
            << "\"buffer_role\":\"" << BufferSummaryName(event.buffer_role) << "\","
            << "\"slot\":" << event.slot << ","
            << "\"magic\":" << event.magic << ","
            << "\"offset\":" << event.offset << ","
            << "\"bytes\":" << event.bytes << ","
            << "\"allow_future_producer\":" << (event.allow_future_producer ? "true" : "false") << ","
            << "\"source_file\":\"" << EscapeJson(event.source_file) << "\","
            << "\"source_line\":" << event.source_line << ","
            << "\"detail\":\"" << EscapeJson(event.detail) << "\""
            << "}";
        if (i + 1 != items.size()) {
            out << "\n";
        }
    }
    return out.str();
}

}  // namespace checker
}  // namespace tilexr
