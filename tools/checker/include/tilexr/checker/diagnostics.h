#ifndef TILEXR_CHECKER_DIAGNOSTICS_H
#define TILEXR_CHECKER_DIAGNOSTICS_H

#include <cstddef>
#include <string>
#include <vector>

#include "tilexr/checker/event.h"
#include "tilexr/checker/oracle.h"
#include "tilexr/checker/types.h"

namespace tilexr {
namespace checker {

enum class FindingKind {
    kOutputMismatch,
    kUnsupportedApi,
    kReadBeforeWrite,
    kReadBeforeCopy,
    kBufferLifetime,
    kFlagNoProducer,
    kFlagStaleMagic,
    kDeadlock,
    kDirectPeerUserBuffer,
    kAmbiguousSource
};

enum class Severity { kInfo, kWarning, kError, kFatal };

struct Finding {
    std::string id;
    FindingKind kind = FindingKind::kAmbiguousSource;
    Severity severity = Severity::kInfo;
    std::string message;
    std::string next_action;
    int rank = -1;
    int peer_rank = -1;
    int core = -1;
    BufferRole buffer_role = BufferRole::kMetadata;
    size_t offset = 0;
    size_t bytes = 0;
    std::string source_file;
    int source_line = 0;
    double confidence = 0.0;
};

class FindingSet {
public:
    void Add(Finding finding);
    const std::vector<Finding> &findings() const;
    const Finding *TopFinding() const;
    bool HasErrors() const;

private:
    std::vector<Finding> findings_;
};

const char *ToString(FindingKind kind);
const char *ToString(Severity severity);

FindingSet CheckOrdering(const EventLog &events);
FindingSet CheckOutputMismatches(const std::vector<OutputMismatch> &mismatches);
FindingSet MergeFindings(const FindingSet &a, const FindingSet &b);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_DIAGNOSTICS_H
