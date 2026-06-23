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

class CollectiveExecutor {
public:
    RunResult Run(RankWorld *world, const CheckerCase &test_case);

private:
    RunResult RunAllGatherInt32(RankWorld *world, const CheckerCase &test_case);
    RunResult RunAllGatherHierarchyDoubleRingInt32(RankWorld *world,
                                                   const CheckerCase &test_case);
    RunResult RunAllReduceSumInt32(RankWorld *world, const CheckerCase &test_case);
    RunResult RunAllReduceBigDataInt32(RankWorld *world, const CheckerCase &test_case);
    RunResult RunEpDispatch(RankWorld *world, const CheckerCase &test_case);
    RunResult RunEpCombine(RankWorld *world, const CheckerCase &test_case);
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_EXECUTOR_H
