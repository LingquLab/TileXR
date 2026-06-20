#include "tilexr/checker/event.h"

namespace tilexr {
namespace checker {

Event &EventLog::Add(Event event) {
    event.id = next_id_++;
    events_.push_back(event);
    return events_.back();
}

const std::vector<Event> &EventLog::events() const {
    return events_;
}

void EventLog::Clear() {
    events_.clear();
    next_id_ = 1;
}

}  // namespace checker
}  // namespace tilexr
