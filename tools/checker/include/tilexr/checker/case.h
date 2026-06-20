#ifndef TILEXR_CHECKER_CASE_H
#define TILEXR_CHECKER_CASE_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "tilexr/checker/status.h"
#include "tilexr/checker/types.h"
#include "tilexr_types.h"

namespace tilexr {
namespace checker {

struct CheckerCase {
    CollectiveOp op;
    int rank_size;
    int64_t count;
    TileXR::TileXRDataType data_type;
    TileXR::TileXRReduceOp reduce_op;
    SchedulerMode scheduler;
    uint64_t magic;
};

CheckerStatus ValidateCase(const CheckerCase &test_case);
size_t ElementSize(TileXR::TileXRDataType data_type);
std::string DescribeCase(const CheckerCase &test_case);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_CASE_H
