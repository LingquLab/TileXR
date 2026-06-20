#ifndef TILEXR_CHECKER_EVENT_H
#define TILEXR_CHECKER_EVENT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tilexr/checker/types.h"

namespace tilexr {
namespace checker {

enum class EventKind {
    kRankStart,
    kRankEnd,
    kCopy,
    kRead,
    kWrite,
    kFlagStore,
    kFlagWait,
    kBarrier,
    kDiagnostic
};

struct Event {
    uint64_t id = 0;
    EventKind kind = EventKind::kDiagnostic;
    int rank = -1;
    int peer_rank = -1;
    int core = -1;
    BufferRole buffer_role = BufferRole::kMetadata;
    int slot = -1;
    uint64_t magic = 0;
    size_t offset = 0;
    size_t bytes = 0;
    std::string source_file;
    int source_line = 0;
    std::string detail;
};

class EventLog {
public:
    Event &Add(Event event);
    const std::vector<Event> &events() const;
    void Clear();

private:
    uint64_t next_id_ = 1;
    std::vector<Event> events_;
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_EVENT_H
