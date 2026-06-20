#ifndef TILEXR_CHECKER_EXECUTOR_H
#define TILEXR_CHECKER_EXECUTOR_H

#include <cstddef>
#include <functional>
#include <vector>

#include "tilexr/checker/case.h"
#include "tilexr/checker/diagnostics.h"
#include "tilexr/checker/oracle.h"
#include "tilexr/checker/status.h"
#include "tilexr/checker/world.h"

namespace tilexr {
namespace checker {

struct RunResult {
    CheckerStatus status;
    FindingSet findings;
    std::vector<OutputMismatch> mismatches;
    size_t event_count = 0;
};

class CollectiveExecutor {
public:
    typedef std::function<void(RankWorld *)> TraceHook;

    RunResult Run(RankWorld *world, const CheckerCase &test_case);
    void SetPostTraceHookForTest(TraceHook hook);

private:
    RunResult RunAllGatherInt32(RankWorld *world, const CheckerCase &test_case);
    RunResult RunAllReduceSumInt32(RankWorld *world, const CheckerCase &test_case);

    TraceHook post_trace_hook_;
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_EXECUTOR_H
