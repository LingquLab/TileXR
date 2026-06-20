#ifndef TILEXR_CHECKER_ORACLE_H
#define TILEXR_CHECKER_ORACLE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tilexr/checker/case.h"
#include "tilexr/checker/status.h"
#include "tilexr/checker/world.h"

namespace tilexr {
namespace checker {

int32_t RankPatternValue(int rank, int64_t index);
CheckerStatus FillRankIndexInt32Inputs(RankWorld *world, const CheckerCase &test_case);

std::vector<int32_t> ExpectedAllGatherInt32(const RankWorld &world,
                                            const CheckerCase &test_case,
                                            int output_rank);

std::vector<int32_t> ExpectedAllReduceSumInt32(const RankWorld &world,
                                               const CheckerCase &test_case,
                                               int output_rank);

struct OutputMismatch {
    int rank;
    int64_t element_index;
    int32_t expected;
    int32_t actual;
    std::string context;
};

std::vector<OutputMismatch> CompareInt32Output(const RankWorld &world,
                                               const CheckerCase &test_case,
                                               size_t max_mismatches);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_ORACLE_H
