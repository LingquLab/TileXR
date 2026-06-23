#ifndef TILEXR_CHECKER_REPORT_H
#define TILEXR_CHECKER_REPORT_H

#include <cstddef>
#include <string>

#include "tilexr/checker/case.h"
#include "tilexr/checker/diagnostics.h"
#include "tilexr/checker/event.h"
#include "tilexr/checker/status.h"

namespace tilexr {
namespace checker {

std::string RenderSummary(const CheckerCase &test_case,
                          const CheckerStatus &status,
                          const FindingSet &findings,
                          size_t mismatch_count,
                          size_t event_count,
                          const std::string &timeline_summary = "");

std::string RenderSummary(const CheckerCase &test_case,
                          const FindingSet &findings,
                          size_t event_count);

std::string RenderFindingsJson(const FindingSet &findings);
std::string RenderEventsJsonl(const EventLog &events);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_REPORT_H
