#include "tilexr/checker/oracle.h"

#include <sstream>

namespace tilexr {
namespace checker {

namespace {

CheckerStatus RequireValidInt32Case(const CheckerCase &test_case) {
    CheckerStatus status = ValidateCase(test_case);
    if (!status.ok()) {
        return status;
    }
    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_INT32) {
        return CheckerStatus::Unsupported("only int32 oracle cases are supported");
    }
    return CheckerStatus::Ok();
}

std::vector<int32_t> ExpectedInt32Output(const RankWorld &world,
                                         const CheckerCase &test_case,
                                         int output_rank) {
    if (test_case.op == CollectiveOp::kAllGather) {
        return ExpectedAllGatherInt32(world, test_case, output_rank);
    }
    return ExpectedAllReduceSumInt32(world, test_case, output_rank);
}

std::string MismatchContext(const CheckerCase &test_case, int rank, int64_t element_index) {
    std::ostringstream stream;
    stream << DescribeCase(test_case)
           << " rank=" << rank
           << " element=" << element_index;
    return stream.str();
}

}  // namespace

CheckerStatus ValidateCase(const CheckerCase &test_case) {
    if (test_case.count <= 0) {
        return CheckerStatus::Unsupported("count must be positive");
    }

    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_INT32) {
        return CheckerStatus::Unsupported("only int32 checker cases are supported");
    }

    switch (test_case.op) {
        case CollectiveOp::kAllGather:
            if (test_case.rank_size != 2 && test_case.rank_size != 4) {
                return CheckerStatus::Unsupported(
                    "allgather rank_size must be 2 or 4");
            }
            return CheckerStatus::Ok();
        case CollectiveOp::kAllReduce:
            if (test_case.rank_size != 2) {
                return CheckerStatus::Unsupported(
                    "allreduce rank_size must be 2");
            }
            if (test_case.reduce_op != TileXR::TILEXR_REDUCE_SUM) {
                return CheckerStatus::Unsupported(
                    "allreduce only supports reduce sum");
            }
            return CheckerStatus::Ok();
    }

    return CheckerStatus::Unsupported("unsupported collective op");
}

size_t ElementSize(TileXR::TileXRDataType data_type) {
    switch (data_type) {
        case TileXR::TILEXR_DATA_TYPE_INT8:
        case TileXR::TILEXR_DATA_TYPE_UINT8:
            return 1;
        case TileXR::TILEXR_DATA_TYPE_INT16:
        case TileXR::TILEXR_DATA_TYPE_UINT16:
        case TileXR::TILEXR_DATA_TYPE_FP16:
        case TileXR::TILEXR_DATA_TYPE_BFP16:
            return 2;
        case TileXR::TILEXR_DATA_TYPE_INT32:
        case TileXR::TILEXR_DATA_TYPE_UINT32:
        case TileXR::TILEXR_DATA_TYPE_FP32:
            return 4;
        case TileXR::TILEXR_DATA_TYPE_INT64:
        case TileXR::TILEXR_DATA_TYPE_UINT64:
        case TileXR::TILEXR_DATA_TYPE_FP64:
            return 8;
        case TileXR::TILEXR_DATA_TYPE_INT128:
            return 16;
        default:
            return 0;
    }
}

std::string DescribeCase(const CheckerCase &test_case) {
    std::ostringstream stream;
    stream << ToString(test_case.op)
           << "(ranks=" << test_case.rank_size
           << ",count=" << test_case.count
           << ",dtype=" << static_cast<int>(test_case.data_type)
           << ",reduce=" << static_cast<int>(test_case.reduce_op)
           << ",scheduler=" << ToString(test_case.scheduler)
           << ",magic=" << test_case.magic
           << ")";
    return stream.str();
}

int32_t RankPatternValue(int rank, int64_t index) {
    return static_cast<int32_t>(rank * 100000 + index);
}

CheckerStatus FillRankIndexInt32Inputs(RankWorld *world, const CheckerCase &test_case) {
    if (world == nullptr) {
        return CheckerStatus::Fail("rank world is null");
    }
    CheckerStatus status = RequireValidInt32Case(test_case);
    if (!status.ok()) {
        return status;
    }
    if (world->rank_size() != test_case.rank_size) {
        return CheckerStatus::Fail("rank world size does not match checker case");
    }

    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        for (int64_t i = 0; i < test_case.count; ++i) {
            status = world->UserInput(rank).WriteInt32(
                static_cast<size_t>(i), RankPatternValue(rank, i));
            if (!status.ok()) {
                return status;
            }
        }
    }
    return CheckerStatus::Ok();
}

std::vector<int32_t> ExpectedAllGatherInt32(const RankWorld &world,
                                            const CheckerCase &test_case,
                                            int output_rank) {
    (void)world;
    (void)output_rank;
    std::vector<int32_t> expected(
        static_cast<size_t>(test_case.rank_size * test_case.count), 0);
    for (int src_rank = 0; src_rank < test_case.rank_size; ++src_rank) {
        for (int64_t element_index = 0; element_index < test_case.count; ++element_index) {
            const size_t expected_index = static_cast<size_t>(
                src_rank * test_case.count + element_index);
            expected[expected_index] = RankPatternValue(src_rank, element_index);
        }
    }
    return expected;
}

std::vector<int32_t> ExpectedAllReduceSumInt32(const RankWorld &world,
                                               const CheckerCase &test_case,
                                               int output_rank) {
    (void)world;
    (void)output_rank;
    std::vector<int32_t> expected(static_cast<size_t>(test_case.count), 0);
    for (int64_t element_index = 0; element_index < test_case.count; ++element_index) {
        int32_t sum = 0;
        for (int rank = 0; rank < test_case.rank_size; ++rank) {
            sum += RankPatternValue(rank, element_index);
        }
        expected[static_cast<size_t>(element_index)] = sum;
    }
    return expected;
}

std::vector<OutputMismatch> CompareInt32Output(const RankWorld &world,
                                               const CheckerCase &test_case,
                                               size_t max_mismatches) {
    std::vector<OutputMismatch> mismatches;
    CheckerStatus status = RequireValidInt32Case(test_case);
    if (!status.ok() || max_mismatches == 0) {
        return mismatches;
    }

    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        const std::vector<int32_t> expected = ExpectedInt32Output(world, test_case, rank);
        for (size_t i = 0; i < expected.size(); ++i) {
            int32_t actual = 0;
            CheckerStatus read_status = world.UserOutput(rank).ReadInt32(i, &actual);
            if (!read_status.ok()) {
                return mismatches;
            }
            if (actual != expected[i]) {
                OutputMismatch mismatch;
                mismatch.rank = rank;
                mismatch.element_index = static_cast<int64_t>(i);
                mismatch.expected = expected[i];
                mismatch.actual = actual;
                mismatch.context = MismatchContext(test_case, rank, mismatch.element_index);
                mismatches.push_back(mismatch);
                if (mismatches.size() >= max_mismatches) {
                    return mismatches;
                }
            }
        }
    }
    return mismatches;
}

}  // namespace checker
}  // namespace tilexr
