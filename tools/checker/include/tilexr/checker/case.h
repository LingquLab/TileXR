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
    CollectiveOp op = CollectiveOp::kAllGather;
    int rank_size = 0;
    int server_count = 1;
    int64_t count = 0;
    int64_t bs = 0;
    int64_t h = 0;
    int64_t top_k = 0;
    int64_t moe_expert_num = 0;
    TileXR::TileXRDataType data_type = TileXR::TILEXR_DATA_TYPE_RESERVED;
    TileXR::TileXRReduceOp reduce_op = TileXR::TILEXR_REDUCE_SUM;
    SchedulerMode scheduler = SchedulerMode::kSerial;
    AlgorithmId algorithm = AlgorithmId::kDefault;
    uint64_t magic = 0;
};

struct AlgorithmSelectionExplanation {
    AlgorithmId algorithm = AlgorithmId::kDefault;
    bool eligible = false;
    std::string reason;
    std::string source_file;
};

CheckerStatus ValidateCase(const CheckerCase &test_case);
AlgorithmSelectionExplanation ExplainAlgorithmSelection(const CheckerCase &test_case);
size_t ElementSize(TileXR::TileXRDataType data_type);
std::string DescribeCase(const CheckerCase &test_case);
std::string AlgorithmSourceFile(AlgorithmId algorithm);
std::string CaseSourceFile(const CheckerCase &test_case);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_CASE_H
