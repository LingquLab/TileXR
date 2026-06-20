#include "tilexr/checker/diagnostics.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

namespace tilexr {
namespace checker {

namespace {

int FindingPriority(FindingKind kind) {
    switch (kind) {
        case FindingKind::kDeadlock:
            return 0;
        case FindingKind::kFlagStaleMagic:
            return 1;
        case FindingKind::kFlagNoProducer:
            return 2;
        case FindingKind::kReadBeforeCopy:
            return 3;
        case FindingKind::kReadBeforeWrite:
            return 4;
        case FindingKind::kDirectPeerUserBuffer:
            return 5;
        case FindingKind::kOutputMismatch:
            return 6;
        case FindingKind::kUnsupportedApi:
            return 7;
        case FindingKind::kBufferLifetime:
            return 8;
        case FindingKind::kAmbiguousSource:
            return 9;
    }
    return 10;
}

bool SeverityIsError(Severity severity) {
    return severity == Severity::kError || severity == Severity::kFatal;
}

std::string MakeFindingId(FindingKind kind, const Event &event) {
    std::ostringstream out;
    out << ToString(kind) << ":" << event.id;
    return out.str();
}

std::string MakeMismatchId(FindingKind kind, const OutputMismatch &mismatch, size_t index) {
    std::ostringstream out;
    out << ToString(kind) << ":" << mismatch.rank << ":" << mismatch.element_index << ":" << index;
    return out.str();
}

bool ContainsInsensitive(const std::string &text, const std::string &needle) {
    std::string lower_text = text;
    std::string lower_needle = needle;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::transform(lower_needle.begin(), lower_needle.end(), lower_needle.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return lower_text.find(lower_needle) != std::string::npos;
}

Finding MakeFindingFromEvent(FindingKind kind, Severity severity, const Event &event,
                             const std::string &message,
                             const std::string &next_action,
                             double confidence) {
    Finding finding;
    finding.id = MakeFindingId(kind, event);
    finding.kind = kind;
    finding.severity = severity;
    finding.message = message;
    finding.next_action = next_action;
    finding.rank = event.rank;
    finding.peer_rank = event.peer_rank;
    finding.core = event.core;
    finding.buffer_role = event.buffer_role;
    finding.offset = event.offset;
    finding.bytes = event.bytes;
    finding.source_file = event.source_file;
    finding.source_line = event.source_line;
    finding.confidence = confidence;
    return finding;
}

bool SameCommDataRange(const Event &lhs, const Event &rhs) {
    return lhs.rank == rhs.rank && lhs.peer_rank == rhs.peer_rank &&
           lhs.slot == rhs.slot && lhs.buffer_role == rhs.buffer_role &&
           lhs.offset == rhs.offset && lhs.bytes == rhs.bytes;
}

bool SafeAdd(size_t lhs, size_t rhs, size_t *out) {
    if (lhs > static_cast<size_t>(-1) - rhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool IsReadSatisfiedByProducer(const Event &read, const Event &producer) {
    if (producer.id >= read.id) {
        return false;
    }
    if (producer.kind != EventKind::kCopy && producer.kind != EventKind::kWrite) {
        return false;
    }
    if (producer.buffer_role != BufferRole::kCommData) {
        return false;
    }
    if (read.buffer_role != BufferRole::kCommData) {
        return false;
    }

    size_t producer_end = 0;
    size_t read_end = 0;
    if (!SafeAdd(producer.offset, producer.bytes, &producer_end) ||
        !SafeAdd(read.offset, read.bytes, &read_end)) {
        return false;
    }

    return producer.peer_rank == read.peer_rank &&
           producer.slot == read.slot &&
           producer.offset <= read.offset &&
           producer_end >= read_end;
}

bool IsMatchingStore(const Event &wait, const Event &store) {
    return store.kind == EventKind::kFlagStore &&
           store.id < wait.id &&
           wait.rank == store.peer_rank &&
           wait.peer_rank == store.rank &&
           wait.slot == store.slot &&
           wait.magic == store.magic;
}

bool IsNewerStoreSameWaitSlot(const Event &wait, const Event &store) {
    return store.kind == EventKind::kFlagStore &&
           store.id < wait.id &&
           wait.rank == store.peer_rank &&
           wait.peer_rank == store.rank &&
           wait.slot == store.slot &&
           store.magic > wait.magic;
}

void SortFindings(std::vector<Finding> *findings) {
    std::sort(findings->begin(), findings->end(),
              [](const Finding &lhs, const Finding &rhs) {
                  const int lhs_priority = FindingPriority(lhs.kind);
                  const int rhs_priority = FindingPriority(rhs.kind);
                  if (lhs_priority != rhs_priority) {
                      return lhs_priority < rhs_priority;
                  }
                  if (lhs.rank != rhs.rank) {
                      return lhs.rank < rhs.rank;
                  }
                  if (lhs.peer_rank != rhs.peer_rank) {
                      return lhs.peer_rank < rhs.peer_rank;
                  }
                  return lhs.id < rhs.id;
              });
}

}  // namespace

void FindingSet::Add(Finding finding) {
    findings_.push_back(finding);
    SortFindings(&findings_);
}

const std::vector<Finding> &FindingSet::findings() const {
    return findings_;
}

const Finding *FindingSet::TopFinding() const {
    if (findings_.empty()) {
        return nullptr;
    }
    return &findings_.front();
}

bool FindingSet::HasErrors() const {
    for (size_t i = 0; i < findings_.size(); ++i) {
        if (SeverityIsError(findings_[i].severity)) {
            return true;
        }
    }
    return false;
}

const char *ToString(FindingKind kind) {
    switch (kind) {
        case FindingKind::kOutputMismatch:
            return "OUTPUT_MISMATCH";
        case FindingKind::kUnsupportedApi:
            return "UNSUPPORTED_API";
        case FindingKind::kReadBeforeWrite:
            return "READ_BEFORE_WRITE";
        case FindingKind::kReadBeforeCopy:
            return "READ_BEFORE_COPY";
        case FindingKind::kBufferLifetime:
            return "BUFFER_LIFETIME";
        case FindingKind::kFlagNoProducer:
            return "FLAG_NO_PRODUCER";
        case FindingKind::kFlagStaleMagic:
            return "FLAG_STALE_MAGIC";
        case FindingKind::kDeadlock:
            return "DEADLOCK";
        case FindingKind::kDirectPeerUserBuffer:
            return "DIRECT_PEER_USER_BUFFER";
        case FindingKind::kAmbiguousSource:
            return "AMBIGUOUS_SOURCE";
    }
    return "UNKNOWN";
}

const char *ToString(Severity severity) {
    switch (severity) {
        case Severity::kInfo:
            return "INFO";
        case Severity::kWarning:
            return "WARNING";
        case Severity::kError:
            return "ERROR";
        case Severity::kFatal:
            return "FATAL";
    }
    return "UNKNOWN";
}

FindingSet CheckOrdering(const EventLog &events) {
    FindingSet findings;
    const std::vector<Event> &entries = events.events();
    for (size_t i = 0; i < entries.size(); ++i) {
        const Event &event = entries[i];

        if (event.kind == EventKind::kFlagWait) {
            bool has_match = false;
            bool has_newer_store = false;
            for (size_t j = 0; j < entries.size(); ++j) {
                const Event &candidate = entries[j];
                if (IsMatchingStore(event, candidate)) {
                    has_match = true;
                }
                if (IsNewerStoreSameWaitSlot(event, candidate)) {
                    has_newer_store = true;
                }
            }
            if (has_newer_store) {
                findings.Add(MakeFindingFromEvent(
                    FindingKind::kFlagStaleMagic, Severity::kError, event,
                    "WaitFlag observed an older magic after a newer producer store on the same slot.",
                    "Verify that the consumer and producer reuse the same magic for each flag round.",
                    0.95));
            } else if (!has_match) {
                findings.Add(MakeFindingFromEvent(
                    FindingKind::kFlagNoProducer, Severity::kError, event,
                    "WaitFlag does not have an earlier matching StoreFlag producer.",
                    "Check whether the producer path skipped the StoreFlag or wrote the wrong peer, slot, or magic.",
                    0.98));
            }
        }

        if (event.kind == EventKind::kRead &&
            (event.buffer_role == BufferRole::kUserInput ||
             event.buffer_role == BufferRole::kUserOutput) &&
            event.peer_rank >= 0 && event.peer_rank != event.rank) {
            findings.Add(MakeFindingFromEvent(
                FindingKind::kDirectPeerUserBuffer, Severity::kError, event,
                "Read directly accesses a peer user buffer instead of staged comm data.",
                "Route peer reads through comm data or another checker-supported staging buffer.",
                0.99));
        }

        if (event.kind == EventKind::kRead &&
            event.buffer_role == BufferRole::kCommData) {
            bool has_copy = false;
            for (size_t j = 0; j < entries.size(); ++j) {
                if (IsReadSatisfiedByProducer(event, entries[j])) {
                    has_copy = true;
                    break;
                }
            }
            if (!has_copy) {
                findings.Add(MakeFindingFromEvent(
                    FindingKind::kReadBeforeCopy, Severity::kError, event,
                    "Comm data was read before any earlier copy or write prepared that region.",
                    "Check whether the producer copy is skipped, delayed, or guarded by the wrong flag.",
                    0.97));
            }
        }
    }
    return findings;
}

FindingSet CheckOutputMismatches(const std::vector<OutputMismatch> &mismatches) {
    FindingSet findings;
    for (size_t i = 0; i < mismatches.size(); ++i) {
        const OutputMismatch &mismatch = mismatches[i];
        const bool unsupported = mismatch.rank == -1 &&
                                 (ContainsInsensitive(mismatch.context, "validation") ||
                                  ContainsInsensitive(mismatch.context, "unsupported"));

        Finding finding;
        finding.id = MakeMismatchId(
            unsupported ? FindingKind::kUnsupportedApi : FindingKind::kOutputMismatch,
            mismatch, i);
        finding.kind = unsupported ? FindingKind::kUnsupportedApi
                                   : FindingKind::kOutputMismatch;
        finding.severity = unsupported ? Severity::kWarning : Severity::kError;
        finding.message = mismatch.context;
        finding.next_action = unsupported
                                  ? "Adjust the checker case to a supported API shape before trusting output validation."
                                  : "Inspect the expected output model and the producing rank's write path.";
        finding.rank = mismatch.rank;
        finding.peer_rank = -1;
        finding.core = -1;
        finding.buffer_role = BufferRole::kUserOutput;
        finding.offset = mismatch.element_index >= 0
                             ? static_cast<size_t>(mismatch.element_index) * sizeof(int32_t)
                             : 0;
        finding.bytes = sizeof(int32_t);
        finding.confidence = unsupported ? 0.85 : 0.92;
        findings.Add(finding);
    }
    return findings;
}

FindingSet MergeFindings(const FindingSet &a, const FindingSet &b) {
    FindingSet merged;
    const std::vector<Finding> &left = a.findings();
    for (size_t i = 0; i < left.size(); ++i) {
        merged.Add(left[i]);
    }
    const std::vector<Finding> &right = b.findings();
    for (size_t i = 0; i < right.size(); ++i) {
        merged.Add(right[i]);
    }
    return merged;
}

}  // namespace checker
}  // namespace tilexr
