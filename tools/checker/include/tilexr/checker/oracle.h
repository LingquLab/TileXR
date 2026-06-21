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

struct EpDispatchRoute {
    int src_rank;
    int token_id;
    int top_k_id;
    int expert_id;
};

enum class EpDispatchMetadataRole {
    kAssist,
    kRecvCounts,
    kExpertTokenNums,
};

struct EpDispatchWindowLayout {
    int64_t rank_size = 0;
    int64_t bs = 0;
    int64_t h = 0;
    int64_t top_k = 0;
    int64_t moe_expert_num = 0;
    int64_t local_expert_num = 0;
    int64_t dtype_bytes = 0;
    int64_t max_routes_per_src = 0;
    int64_t row_bytes = 0;
    int64_t payload_bytes_per_slot = 0;
    int64_t assist_bytes_per_slot = 0;
    int64_t slot_bytes = 0;
    int64_t total_bytes = 0;
};

uint16_t EpDispatchInputValue(int rank, int64_t token, int64_t h_index);
std::vector<int32_t> DefaultEpDispatchExpertIds(const CheckerCase &test_case);
std::vector<EpDispatchRoute> ExpectedEpDispatchRoutes(const CheckerCase &test_case,
                                                       const std::vector<int32_t> &expert_ids,
                                                       int dst_rank);
CheckerStatus FillEpDispatchInputs(RankWorld *world, const CheckerCase &test_case);
CheckerStatus FillEpCombineInputs(RankWorld *world, const CheckerCase &test_case,
                                  const std::vector<int32_t> &expert_ids);
CheckerStatus ComputeEpDispatchWindowLayout(const CheckerCase &test_case,
                                            EpDispatchWindowLayout *layout);
size_t EpDispatchPayloadBytes(const CheckerCase &test_case);
size_t EpDispatchMetadataBytes(const CheckerCase &test_case, EpDispatchMetadataRole role);
size_t EpDispatchMetadataOffset(const CheckerCase &test_case, EpDispatchMetadataRole role);
size_t EpDispatchOutputBytes(const CheckerCase &test_case);
size_t EpCombineInputBytes(const CheckerCase &test_case);
size_t EpCombineOutputBytes(const CheckerCase &test_case);

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

std::vector<OutputMismatch> CompareEpDispatchOutput(const RankWorld &world,
                                                    const CheckerCase &test_case,
                                                    const std::vector<int32_t> &expert_ids,
                                                    size_t max_mismatches);

std::vector<OutputMismatch> CompareEpCombineOutput(const RankWorld &world,
                                                   const CheckerCase &test_case,
                                                   const std::vector<int32_t> &expert_ids,
                                                   size_t max_mismatches);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_ORACLE_H
