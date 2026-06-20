#ifndef TILEXR_CHECKER_EXECUTOR_H
#define TILEXR_CHECKER_EXECUTOR_H

#include <cstddef>
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

CheckerStatus FinalizeExecutorStatus(const FindingSet &findings,
                                     const std::vector<OutputMismatch> &mismatches);

class CollectiveExecutor {
public:
    RunResult Run(RankWorld *world, const CheckerCase &test_case);

private:
    RunResult RunAllGatherInt32(RankWorld *world, const CheckerCase &test_case);
    RunResult RunAllReduceSumInt32(RankWorld *world, const CheckerCase &test_case);
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_EXECUTOR_H
