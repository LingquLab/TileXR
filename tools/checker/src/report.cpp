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
        case EventKind::kBarrier:
            return "BARRIER";
        case EventKind::kDiagnostic:
            return "DIAGNOSTIC";
    }
    return "UNKNOWN";
}

}  // namespace

std::string RenderSummary(const CheckerCase &test_case,
                          const FindingSet &findings,
                          size_t event_count) {
    (void)test_case;
    std::ostringstream out;
    out << "checker: " << (findings.HasErrors() ? "FAIL" : "PASS") << "\n";
    const Finding *top = findings.TopFinding();
    if (top != nullptr) {
        out << "top finding: " << ToString(top->kind) << "\n";
        out << "rank: " << top->rank << "\n";
        out << "peer rank: " << top->peer_rank << "\n";
        out << "buffer: " << BufferSummaryName(top->buffer_role) << "\n";
        out << "next action: " << top->next_action << "\n";
    } else {
        out << "top finding: NONE\n";
        out << "rank: -1\n";
        out << "peer rank: -1\n";
        out << "buffer: Metadata\n";
        out << "next action: No action required.\n";
    }
    out << "events: " << event_count << "\n";
    return out.str();
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
            << "\"rank\":" << finding.rank << ","
            << "\"peer_rank\":" << finding.peer_rank << ","
            << "\"core\":" << finding.core << ","
            << "\"buffer_role\":\"" << BufferSummaryName(finding.buffer_role) << "\","
            << "\"offset\":" << finding.offset << ","
            << "\"bytes\":" << finding.bytes << ","
            << "\"source_file\":\"" << EscapeJson(finding.source_file) << "\","
            << "\"source_line\":" << finding.source_line << ","
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
            << "\"core\":" << event.core << ","
            << "\"buffer_role\":\"" << BufferSummaryName(event.buffer_role) << "\","
            << "\"slot\":" << event.slot << ","
            << "\"magic\":" << event.magic << ","
            << "\"offset\":" << event.offset << ","
            << "\"bytes\":" << event.bytes << ","
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
