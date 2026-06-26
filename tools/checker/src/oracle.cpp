#include "tilexr/checker/oracle.h"

#include <cstddef>
#include <cstdint>
#include <sstream>

namespace tilexr {
namespace checker {

namespace {

const int64_t kAllGatherHdbSmallDataSize = 32LL * 1024 * 1024;
const int kAllGatherHdbSmallRankSize = 8;
const int64_t kAllReduceBigDataMinBytes = 2LL * 1024 * 1024;
const char *kEpDispatchSourceFile = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
const char *kEpCombineSourceFile = "src/ep/kernels/tilexr_ep_combine_kernel.cpp";
const int64_t kEpWindowAlignmentBytes = 32;
const int64_t kEpWindowHeaderBytes = 64;
const int64_t kEpSrcSlotHeaderBytes = 64;
const int64_t kEpAssistTupleInts = 4;

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

bool IsEpDispatchDataType(TileXR::TileXRDataType data_type) {
    return data_type == TileXR::TILEXR_DATA_TYPE_FP16 ||
           data_type == TileXR::TILEXR_DATA_TYPE_BFP16;
}

bool IsEpOp(CollectiveOp op) {
    return op == CollectiveOp::kEpDispatch || op == CollectiveOp::kEpCombine;
}

CheckerStatus ValidateEpShape(const CheckerCase &test_case) {
    const char *op_name = ToString(test_case.op);
    if (test_case.algorithm != AlgorithmId::kDefault) {
        return CheckerStatus::Unsupported(
            std::string(op_name) + " only supports the default checker model");
    }
    if (!IsEpDispatchDataType(test_case.data_type)) {
        return CheckerStatus::Unsupported(std::string(op_name) +
                                          " supports fp16 or bfp16 only");
    }
    if (test_case.bs <= 0 || test_case.h <= 0 || test_case.top_k <= 0 ||
        test_case.moe_expert_num <= 0) {
        return CheckerStatus::Unsupported(
            std::string(op_name) +
            " requires positive bs, h, top_k, and moe_expert_num");
    }
    if (test_case.moe_expert_num % test_case.rank_size != 0) {
        return CheckerStatus::Unsupported(
            std::string(op_name) + " requires moe_expert_num divisible by rank_size");
    }
    return CheckerStatus::Ok();
}

bool MulInt64(int64_t lhs, int64_t rhs, int64_t *out) {
    if (out == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > INT64_MAX / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool AddInt64(int64_t lhs, int64_t rhs, int64_t *out) {
    if (out == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (rhs > INT64_MAX - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool AlignUpInt64(int64_t value, int64_t alignment, int64_t *out) {
    if (out == nullptr || value < 0 || alignment <= 0) {
        return false;
    }
    const int64_t remainder = value % alignment;
    if (remainder == 0) {
        *out = value;
        return true;
    }
    return AddInt64(value, alignment - remainder, out);
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

std::string EpMismatchContext(const CheckerCase &test_case, int rank, const std::string &field,
                              int64_t element_index) {
    std::ostringstream stream;
    stream << DescribeCase(test_case)
           << " rank=" << rank
           << " " << field << "=" << element_index;
    return stream.str();
}

OutputMismatch MakeFailureMismatch(int rank, int64_t element_index, int32_t expected,
                                   const std::string &context) {
    OutputMismatch mismatch;
    mismatch.rank = rank;
    mismatch.element_index = element_index;
    mismatch.expected = expected;
    mismatch.actual = 0;
    mismatch.context = context;
    return mismatch;
}

}  // namespace

CheckerStatus ValidateCase(const CheckerCase &test_case) {
    if (test_case.rank_size <= 0) {
        return CheckerStatus::Unsupported("rank_size must be positive");
    }

    if (test_case.server_count <= 0) {
        return CheckerStatus::Unsupported("server_count must be positive");
    }

    if (test_case.server_count > test_case.rank_size) {
        return CheckerStatus::Unsupported("server_count must not exceed rank_size");
    }

    if (test_case.rank_size % test_case.server_count != 0) {
        return CheckerStatus::Unsupported("rank_size must be divisible by server_count");
    }

    switch (test_case.op) {
        case CollectiveOp::kAllGather:
            if (test_case.count <= 0) {
                return CheckerStatus::Unsupported("count must be positive");
            }
            if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_INT32) {
                return CheckerStatus::Unsupported("only int32 checker cases are supported");
            }
            if (test_case.algorithm == AlgorithmId::kAllReduceBigData) {
                return CheckerStatus::Unsupported(
                    "allreduce_big_data can only be used with allreduce");
            }
            if (test_case.algorithm == AlgorithmId::kAllGatherHierarchyDoubleRing) {
                if (test_case.rank_size <= 2 || test_case.rank_size % 2 != 0) {
                    return CheckerStatus::Unsupported(
                        "allgather_hierarchy_double_ring requires even rank_size greater than 2");
                }
                const int64_t data_size = test_case.count * static_cast<int64_t>(sizeof(int32_t));
                if (data_size <= kAllGatherHdbSmallDataSize &&
                    test_case.rank_size <= kAllGatherHdbSmallRankSize) {
                    return CheckerStatus::Unsupported(
                        "allgather_hierarchy_double_ring requires data size > 32MiB or rank_size > 8");
                }
                return CheckerStatus::Ok();
            }
            if (test_case.rank_size != 2 && test_case.rank_size != 4) {
                return CheckerStatus::Unsupported(
                    "allgather rank_size must be 2 or 4");
            }
            return CheckerStatus::Ok();
        case CollectiveOp::kAllReduce:
            if (test_case.count <= 0) {
                return CheckerStatus::Unsupported("count must be positive");
            }
            if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_INT32) {
                return CheckerStatus::Unsupported("only int32 checker cases are supported");
            }
            if (test_case.algorithm == AlgorithmId::kAllGatherHierarchyDoubleRing) {
                return CheckerStatus::Unsupported(
                    "allgather_hierarchy_double_ring can only be used with allgather");
            }
            if (test_case.algorithm == AlgorithmId::kDefault && test_case.rank_size != 2) {
                return CheckerStatus::Unsupported(
                    "allreduce rank_size must be 2");
            }
            if (test_case.algorithm == AlgorithmId::kAllReduceBigData) {
                if (test_case.rank_size <= 2) {
                    return CheckerStatus::Unsupported(
                        "allreduce_big_data requires rank_size greater than 2");
                }
                const int64_t data_size = test_case.count * static_cast<int64_t>(sizeof(int32_t));
                if (data_size < kAllReduceBigDataMinBytes) {
                    return CheckerStatus::Unsupported(
                        "allreduce_big_data requires data size >= 2MiB");
                }
                if (test_case.reduce_op != TileXR::TILEXR_REDUCE_SUM) {
                    return CheckerStatus::Unsupported(
                        "allreduce only supports reduce sum");
                }
                return CheckerStatus::Ok();
            }
            if (test_case.rank_size != 2) {
                return CheckerStatus::Unsupported(
                    "allreduce rank_size must be 2");
            }
            if (test_case.reduce_op != TileXR::TILEXR_REDUCE_SUM) {
                return CheckerStatus::Unsupported(
                    "allreduce only supports reduce sum");
            }
            return CheckerStatus::Ok();
        case CollectiveOp::kEpDispatch:
            return ValidateEpShape(test_case);
        case CollectiveOp::kEpCombine:
            return ValidateEpShape(test_case);
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
           << ",servers=" << test_case.server_count;
    if (IsEpOp(test_case.op)) {
        stream << ",bs=" << test_case.bs
               << ",h=" << test_case.h
               << ",top_k=" << test_case.top_k
               << ",experts=" << test_case.moe_expert_num;
    } else {
        stream << ",count=" << test_case.count
               << ",dtype=" << static_cast<int>(test_case.data_type)
               << ",reduce=" << static_cast<int>(test_case.reduce_op);
    }
    if (IsEpOp(test_case.op)) {
        stream << ",dtype=" << static_cast<int>(test_case.data_type);
    }
    stream << ",scheduler=" << ToString(test_case.scheduler)
           << ",algorithm=" << ToString(test_case.algorithm)
           << ",magic=" << test_case.magic
           << ")";
    return stream.str();
}

std::string AlgorithmSourceFile(AlgorithmId algorithm) {
    switch (algorithm) {
        case AlgorithmId::kDefault:
            return "";
        case AlgorithmId::kAllGatherHierarchyDoubleRing:
            return "src/collectives/kernels/91093/allgather_hierarchy_double_ring.h";
        case AlgorithmId::kAllReduceBigData:
            return "src/collectives/kernels/allreduce_big_data.h";
    }
    return "";
}

std::string CaseSourceFile(const CheckerCase &test_case) {
    if (test_case.op == CollectiveOp::kEpDispatch) {
        return kEpDispatchSourceFile;
    }
    if (test_case.op == CollectiveOp::kEpCombine) {
        return kEpCombineSourceFile;
    }
    return AlgorithmSourceFile(test_case.algorithm);
}

AlgorithmSelectionExplanation ExplainAlgorithmSelection(const CheckerCase &test_case) {
    AlgorithmSelectionExplanation explanation;
    explanation.algorithm = test_case.algorithm;
    explanation.source_file = CaseSourceFile(test_case);

    if (test_case.op == CollectiveOp::kEpDispatch) {
        explanation.eligible = test_case.algorithm == AlgorithmId::kDefault;
        explanation.reason = explanation.eligible
            ? "ep_dispatch uses the source-aligned CPU oracle for dispatch payload, metadata, and window events"
            : "ep_dispatch only supports the default checker model";
        return explanation;
    }

    if (test_case.op == CollectiveOp::kEpCombine) {
        const CheckerStatus status = ValidateCase(test_case);
        explanation.eligible = status.ok();
        explanation.reason = explanation.eligible
            ? "ep_combine uses the source-aligned CPU oracle for dispatch metadata inverse routing and output reconstruction"
            : status.message;
        return explanation;
    }

    if (test_case.algorithm == AlgorithmId::kDefault) {
        explanation.eligible = ValidateCase(test_case).ok();
        explanation.reason = explanation.eligible
            ? "default checker software model selected"
            : ValidateCase(test_case).message;
        return explanation;
    }

    if (test_case.algorithm == AlgorithmId::kAllGatherHierarchyDoubleRing) {
        explanation.source_file = AlgorithmSourceFile(test_case.algorithm);
        if (test_case.op != CollectiveOp::kAllGather) {
            explanation.reason = "allgather_hierarchy_double_ring can only be used with allgather";
            return explanation;
        }
        if (test_case.rank_size <= 2 || test_case.rank_size % 2 != 0) {
            explanation.reason =
                "allgather_hierarchy_double_ring requires even rank_size greater than 2";
            return explanation;
        }
        const int64_t data_size = test_case.count * static_cast<int64_t>(sizeof(int32_t));
        if (data_size <= kAllGatherHdbSmallDataSize &&
            test_case.rank_size <= kAllGatherHdbSmallRankSize) {
            explanation.reason =
                "data size <= 32MiB and rank_size <= 8, so the production selector condition is not met";
            return explanation;
        }
        explanation.eligible = true;
        explanation.reason =
            test_case.rank_size > kAllGatherHdbSmallRankSize
                ? "allgather_hierarchy_double_ring selected because rank_size > 8"
                : "allgather_hierarchy_double_ring selected because data size > 32MiB";
        return explanation;
    }

    if (test_case.algorithm == AlgorithmId::kAllReduceBigData) {
        explanation.source_file = AlgorithmSourceFile(test_case.algorithm);
        if (test_case.op != CollectiveOp::kAllReduce) {
            explanation.reason = "allreduce_big_data can only be used with allreduce";
            return explanation;
        }
        if (test_case.rank_size <= 2) {
            explanation.reason = "allreduce_big_data requires rank_size greater than 2";
            return explanation;
        }
        const int64_t data_size = test_case.count * static_cast<int64_t>(sizeof(int32_t));
        if (data_size < kAllReduceBigDataMinBytes) {
            explanation.reason = "data size < 2MiB, so allreduce_big_data is not eligible";
            return explanation;
        }
        if (test_case.reduce_op != TileXR::TILEXR_REDUCE_SUM) {
            explanation.reason = "allreduce_big_data only supports reduce sum";
            return explanation;
        }
        explanation.eligible = true;
        explanation.reason = "allreduce_big_data selected because rank_size > 2 and data size >= 2MiB";
        return explanation;
    }

    explanation.reason = "unknown algorithm";
    return explanation;
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

uint16_t EpDispatchInputValue(int rank, int64_t token, int64_t h_index) {
    return static_cast<uint16_t>(0x1000 + rank * 0x100 + token * 0x10 + h_index);
}

std::vector<int32_t> DefaultEpDispatchExpertIds(const CheckerCase &test_case) {
    std::vector<int32_t> expert_ids;
    if (test_case.bs <= 0 || test_case.top_k <= 0 || test_case.rank_size <= 0 ||
        test_case.moe_expert_num <= 0 ||
        test_case.moe_expert_num % test_case.rank_size != 0) {
        return expert_ids;
    }

    expert_ids.reserve(static_cast<size_t>(test_case.bs * test_case.top_k));
    const int64_t local_expert_num = test_case.moe_expert_num / test_case.rank_size;
    for (int64_t token = 0; token < test_case.bs; ++token) {
        for (int64_t top_k_id = 0; top_k_id < test_case.top_k; ++top_k_id) {
            const int64_t rank_group =
                top_k_id % 2 == 0 ? token % test_case.rank_size
                                  : test_case.rank_size - 1 - (token % test_case.rank_size);
            int64_t local_expert = top_k_id % 2 == 0 ? 0 : local_expert_num - 1;
            local_expert = (local_expert + top_k_id / 2) % local_expert_num;
            const int64_t expert = (rank_group * local_expert_num + local_expert) %
                                   test_case.moe_expert_num;
            expert_ids.push_back(static_cast<int32_t>(expert));
        }
    }
    return expert_ids;
}

std::vector<EpDispatchRoute> ExpectedEpDispatchRoutes(const CheckerCase &test_case,
                                                       const std::vector<int32_t> &expert_ids,
                                                       int dst_rank) {
    std::vector<EpDispatchRoute> routes;
    if (dst_rank < 0 || dst_rank >= test_case.rank_size ||
        test_case.moe_expert_num % test_case.rank_size != 0) {
        return routes;
    }

    const int64_t route_count = test_case.bs * test_case.top_k;
    if (route_count < 0 || expert_ids.size() < static_cast<size_t>(route_count)) {
        return routes;
    }

    const int64_t local_expert_num = test_case.moe_expert_num / test_case.rank_size;
    for (int src_rank = 0; src_rank < test_case.rank_size; ++src_rank) {
        for (int64_t token = 0; token < test_case.bs; ++token) {
            for (int64_t top_k_id = 0; top_k_id < test_case.top_k; ++top_k_id) {
                const int64_t route_index = token * test_case.top_k + top_k_id;
                const int32_t expert_id = expert_ids[static_cast<size_t>(route_index)];
                if (expert_id < 0 ||
                    static_cast<int64_t>(expert_id) >= test_case.moe_expert_num) {
                    continue;
                }
                if (static_cast<int64_t>(expert_id) / local_expert_num != dst_rank) {
                    continue;
                }
                EpDispatchRoute route;
                route.src_rank = src_rank;
                route.token_id = static_cast<int>(token);
                route.top_k_id = static_cast<int>(top_k_id);
                route.expert_id = expert_id;
                routes.push_back(route);
            }
        }
    }
    return routes;
}

CheckerStatus FillEpDispatchInputs(RankWorld *world, const CheckerCase &test_case) {
    if (world == nullptr) {
        return CheckerStatus::Fail("rank world is null");
    }
    CheckerStatus status = ValidateCase(test_case);
    if (!status.ok()) {
        return status;
    }
    if (world->rank_size() != test_case.rank_size) {
        return CheckerStatus::Fail("rank world size does not match checker case");
    }

    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        for (int64_t token = 0; token < test_case.bs; ++token) {
            for (int64_t h_index = 0; h_index < test_case.h; ++h_index) {
                const uint16_t value = EpDispatchInputValue(rank, token, h_index);
                const size_t offset = static_cast<size_t>(
                    (token * test_case.h + h_index) * sizeof(uint16_t));
                status = world->UserInput(rank).WriteBytes(offset, &value, sizeof(value));
                if (!status.ok()) {
                    return status;
                }
            }
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus FillEpCombineInputs(RankWorld *world, const CheckerCase &test_case,
                                  const std::vector<int32_t> &expert_ids) {
    if (world == nullptr) {
        return CheckerStatus::Fail("rank world is null");
    }
    CheckerStatus status = ValidateCase(test_case);
    if (!status.ok()) {
        return status;
    }
    if (world->rank_size() != test_case.rank_size) {
        return CheckerStatus::Fail("rank world size does not match checker case");
    }

    const size_t dtype_bytes = ElementSize(test_case.data_type);
    const size_t row_bytes = static_cast<size_t>(test_case.h) * dtype_bytes;
    const int64_t local_expert_num = test_case.moe_expert_num / test_case.rank_size;
    for (int dst_rank = 0; dst_rank < test_case.rank_size; ++dst_rank) {
        const std::vector<EpDispatchRoute> routes =
            ExpectedEpDispatchRoutes(test_case, expert_ids, dst_rank);
        std::vector<int32_t> recv_counts(static_cast<size_t>(test_case.rank_size), 0);
        std::vector<int64_t> expert_counts(static_cast<size_t>(local_expert_num), 0);

        for (size_t row = 0; row < routes.size(); ++row) {
            const EpDispatchRoute &route = routes[row];
            for (int64_t h_index = 0; h_index < test_case.h; ++h_index) {
                const uint16_t value =
                    EpDispatchInputValue(route.src_rank, route.token_id, h_index);
                const size_t offset =
                    row * row_bytes + static_cast<size_t>(h_index) * sizeof(uint16_t);
                status = world->UserInput(dst_rank).WriteBytes(offset, &value, sizeof(value));
                if (!status.ok()) {
                    return status;
                }
            }

            const int32_t tuple[4] = {
                route.src_rank,
                route.token_id,
                route.top_k_id,
                route.expert_id,
            };
            const size_t tuple_offset =
                EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kAssist) +
                row * kEpAssistTupleInts * sizeof(int32_t);
            status = world->UserInput(dst_rank).WriteBytes(tuple_offset, tuple, sizeof(tuple));
            if (!status.ok()) {
                return status;
            }

            ++recv_counts[static_cast<size_t>(route.src_rank)];
            const int64_t local_expert =
                static_cast<int64_t>(route.expert_id) % local_expert_num;
            if (local_expert >= 0 && local_expert < local_expert_num) {
                ++expert_counts[static_cast<size_t>(local_expert)];
            }
        }

        const size_t recv_offset =
            EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kRecvCounts);
        for (int src_rank = 0; src_rank < test_case.rank_size; ++src_rank) {
            status = world->UserInput(dst_rank).WriteBytes(
                recv_offset + static_cast<size_t>(src_rank) * sizeof(int32_t),
                &recv_counts[static_cast<size_t>(src_rank)], sizeof(int32_t));
            if (!status.ok()) {
                return status;
            }
        }

        const size_t expert_offset =
            EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kExpertTokenNums);
        for (int64_t local_expert = 0; local_expert < local_expert_num; ++local_expert) {
            status = world->UserInput(dst_rank).WriteBytes(
                expert_offset + static_cast<size_t>(local_expert) * sizeof(int64_t),
                &expert_counts[static_cast<size_t>(local_expert)], sizeof(int64_t));
            if (!status.ok()) {
                return status;
            }
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus ComputeEpDispatchWindowLayout(const CheckerCase &test_case,
                                            EpDispatchWindowLayout *layout) {
    if (layout == nullptr) {
        return CheckerStatus::Fail("ep dispatch window layout pointer is null");
    }
    CheckerStatus status = ValidateCase(test_case);
    if (!status.ok()) {
        return status;
    }

    EpDispatchWindowLayout next;
    next.rank_size = test_case.rank_size;
    next.bs = test_case.bs;
    next.h = test_case.h;
    next.top_k = test_case.top_k;
    next.moe_expert_num = test_case.moe_expert_num;
    next.local_expert_num = test_case.moe_expert_num / test_case.rank_size;
    next.dtype_bytes = static_cast<int64_t>(ElementSize(test_case.data_type));
    if (next.dtype_bytes <= 0) {
        return CheckerStatus::Unsupported("ep dispatch unsupported dtype bytes");
    }

    if (!MulInt64(test_case.bs, test_case.top_k, &next.max_routes_per_src) ||
        !MulInt64(test_case.h, next.dtype_bytes, &next.row_bytes)) {
        return CheckerStatus::Unsupported("ep dispatch window layout overflow");
    }

    int64_t payload_bytes = 0;
    if (!MulInt64(next.max_routes_per_src, next.row_bytes, &payload_bytes) ||
        !AlignUpInt64(payload_bytes, kEpWindowAlignmentBytes,
                      &next.payload_bytes_per_slot)) {
        return CheckerStatus::Unsupported("ep dispatch payload layout overflow");
    }

    int64_t assist_tuple_bytes = 0;
    int64_t assist_bytes = 0;
    if (!MulInt64(kEpAssistTupleInts, static_cast<int64_t>(sizeof(int32_t)),
                  &assist_tuple_bytes) ||
        !MulInt64(next.max_routes_per_src, assist_tuple_bytes, &assist_bytes) ||
        !AlignUpInt64(assist_bytes, kEpWindowAlignmentBytes,
                      &next.assist_bytes_per_slot)) {
        return CheckerStatus::Unsupported("ep dispatch assist layout overflow");
    }

    int64_t slot_payload_end = 0;
    int64_t raw_slot_bytes = 0;
    if (!AddInt64(kEpSrcSlotHeaderBytes, next.payload_bytes_per_slot,
                  &slot_payload_end) ||
        !AddInt64(slot_payload_end, next.assist_bytes_per_slot,
                  &raw_slot_bytes) ||
        !AlignUpInt64(raw_slot_bytes, kEpWindowAlignmentBytes,
                      &next.slot_bytes)) {
        return CheckerStatus::Unsupported("ep dispatch slot layout overflow");
    }

    int64_t slots_bytes = 0;
    if (!MulInt64(test_case.rank_size, next.slot_bytes, &slots_bytes) ||
        !AddInt64(kEpWindowHeaderBytes, slots_bytes, &next.total_bytes)) {
        return CheckerStatus::Unsupported("ep dispatch total layout overflow");
    }

    *layout = next;
    return CheckerStatus::Ok();
}

size_t EpDispatchPayloadBytes(const CheckerCase &test_case) {
    return static_cast<size_t>(test_case.rank_size *
                               test_case.bs *
                               test_case.top_k *
                               test_case.h) *
           ElementSize(test_case.data_type);
}

size_t EpDispatchMetadataBytes(const CheckerCase &test_case, EpDispatchMetadataRole role) {
    switch (role) {
        case EpDispatchMetadataRole::kAssist:
            return static_cast<size_t>(test_case.bs * test_case.top_k *
                                       kEpAssistTupleInts * sizeof(int32_t));
        case EpDispatchMetadataRole::kRecvCounts:
            return static_cast<size_t>(test_case.rank_size) * sizeof(int32_t);
        case EpDispatchMetadataRole::kExpertTokenNums:
            if (test_case.rank_size <= 0 ||
                test_case.moe_expert_num % test_case.rank_size != 0) {
                return 0;
            }
            return static_cast<size_t>(test_case.moe_expert_num / test_case.rank_size) *
                   sizeof(int64_t);
    }
    return 0;
}

size_t EpDispatchMetadataOffset(const CheckerCase &test_case, EpDispatchMetadataRole role) {
    const size_t payload_bytes = EpDispatchPayloadBytes(test_case);
    switch (role) {
        case EpDispatchMetadataRole::kAssist:
            return payload_bytes;
        case EpDispatchMetadataRole::kRecvCounts:
            return payload_bytes +
                   EpDispatchMetadataBytes(test_case, EpDispatchMetadataRole::kAssist);
        case EpDispatchMetadataRole::kExpertTokenNums:
            return payload_bytes +
                   EpDispatchMetadataBytes(test_case, EpDispatchMetadataRole::kAssist) +
                   EpDispatchMetadataBytes(test_case, EpDispatchMetadataRole::kRecvCounts);
    }
    return payload_bytes;
}

size_t EpDispatchOutputBytes(const CheckerCase &test_case) {
    return EpDispatchPayloadBytes(test_case) +
           EpDispatchMetadataBytes(test_case, EpDispatchMetadataRole::kAssist) +
           EpDispatchMetadataBytes(test_case, EpDispatchMetadataRole::kRecvCounts) +
           EpDispatchMetadataBytes(test_case, EpDispatchMetadataRole::kExpertTokenNums);
}

size_t EpCombineInputBytes(const CheckerCase &test_case) {
    return EpDispatchOutputBytes(test_case);
}

size_t EpCombineOutputBytes(const CheckerCase &test_case) {
    return static_cast<size_t>(test_case.bs * test_case.top_k * test_case.h) *
           ElementSize(test_case.data_type);
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
    if (max_mismatches == 0) {
        return mismatches;
    }
    CheckerStatus status = RequireValidInt32Case(test_case);
    if (!status.ok()) {
        mismatches.push_back(
            MakeFailureMismatch(-1, -1, 0, "compare validation failed: " + status.message));
        return mismatches;
    }

    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        const std::vector<int32_t> expected = ExpectedInt32Output(world, test_case, rank);
        for (size_t i = 0; i < expected.size(); ++i) {
            int32_t actual = 0;
            CheckerStatus read_status = world.UserOutput(rank).ReadInt32(i, &actual);
            if (!read_status.ok()) {
                mismatches.push_back(MakeFailureMismatch(
                    rank, static_cast<int64_t>(i), expected[i],
                    MismatchContext(test_case, rank, static_cast<int64_t>(i)) +
                        " read failure: " + read_status.message));
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

std::vector<OutputMismatch> CompareEpDispatchOutput(const RankWorld &world,
                                                    const CheckerCase &test_case,
                                                    const std::vector<int32_t> &expert_ids,
                                                    size_t max_mismatches) {
    std::vector<OutputMismatch> mismatches;
    if (max_mismatches == 0) {
        return mismatches;
    }
    CheckerStatus status = ValidateCase(test_case);
    if (!status.ok()) {
        mismatches.push_back(
            MakeFailureMismatch(-1, -1, 0, "ep compare validation failed: " + status.message));
        return mismatches;
    }

    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        const std::vector<EpDispatchRoute> routes =
            ExpectedEpDispatchRoutes(test_case, expert_ids, rank);
        for (size_t row = 0; row < routes.size(); ++row) {
            const EpDispatchRoute &route = routes[row];
            for (int64_t h_index = 0; h_index < test_case.h; ++h_index) {
                const size_t element_index =
                    row * static_cast<size_t>(test_case.h) + static_cast<size_t>(h_index);
                const size_t offset = element_index * sizeof(uint16_t);
                uint16_t actual = 0;
                CheckerStatus read_status =
                    world.UserOutput(rank).ReadBytes(offset, &actual, sizeof(actual));
                const uint16_t expected =
                    EpDispatchInputValue(route.src_rank, route.token_id, h_index);
                if (!read_status.ok()) {
                    mismatches.push_back(MakeFailureMismatch(
                        rank, static_cast<int64_t>(element_index), expected,
                        MismatchContext(test_case, rank, static_cast<int64_t>(element_index)) +
                            " read failure: " + read_status.message));
                    return mismatches;
                }
                if (actual != expected) {
                    OutputMismatch mismatch;
                    mismatch.rank = rank;
                    mismatch.element_index = static_cast<int64_t>(element_index);
                    mismatch.expected = expected;
                    mismatch.actual = actual;
                    mismatch.context =
                        MismatchContext(test_case, rank, mismatch.element_index);
                    mismatches.push_back(mismatch);
                    if (mismatches.size() >= max_mismatches) {
                        return mismatches;
                    }
                }
            }
        }

        const size_t assist_offset =
            EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kAssist);
        const size_t recv_offset =
            EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kRecvCounts);
        const size_t expert_offset =
            EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kExpertTokenNums);
        std::vector<int32_t> expected_recv_counts(static_cast<size_t>(test_case.rank_size), 0);
        std::vector<int64_t> expected_expert_counts(
            static_cast<size_t>(test_case.moe_expert_num / test_case.rank_size), 0);
        for (size_t row = 0; row < routes.size(); ++row) {
            const EpDispatchRoute &route = routes[row];
            const int32_t expected_tuple[4] = {
                route.src_rank,
                route.token_id,
                route.top_k_id,
                route.expert_id,
            };
            ++expected_recv_counts[static_cast<size_t>(route.src_rank)];
            const int64_t local_expert =
                static_cast<int64_t>(route.expert_id) %
                (test_case.moe_expert_num / test_case.rank_size);
            if (local_expert >= 0 &&
                local_expert < static_cast<int64_t>(expected_expert_counts.size())) {
                ++expected_expert_counts[static_cast<size_t>(local_expert)];
            }
            for (int tuple_index = 0; tuple_index < 4; ++tuple_index) {
                int32_t actual = 0;
                const size_t tuple_offset =
                    assist_offset + (row * kEpAssistTupleInts +
                                     static_cast<size_t>(tuple_index)) * sizeof(int32_t);
                CheckerStatus read_status =
                    world.UserOutput(rank).ReadBytes(tuple_offset, &actual, sizeof(actual));
                if (!read_status.ok()) {
                    mismatches.push_back(MakeFailureMismatch(
                        rank, static_cast<int64_t>(row * kEpAssistTupleInts + tuple_index),
                        expected_tuple[tuple_index],
                        EpMismatchContext(test_case, rank, "assist",
                                          static_cast<int64_t>(row)) +
                            " read failure: " + read_status.message));
                    return mismatches;
                }
                if (actual != expected_tuple[tuple_index]) {
                    OutputMismatch mismatch;
                    mismatch.rank = rank;
                    mismatch.element_index =
                        static_cast<int64_t>(row * kEpAssistTupleInts + tuple_index);
                    mismatch.expected = expected_tuple[tuple_index];
                    mismatch.actual = actual;
                    mismatch.context =
                        EpMismatchContext(test_case, rank, "assist",
                                          static_cast<int64_t>(row));
                    mismatches.push_back(mismatch);
                    if (mismatches.size() >= max_mismatches) {
                        return mismatches;
                    }
                }
            }
        }

        for (int src_rank = 0; src_rank < test_case.rank_size; ++src_rank) {
            int32_t actual = 0;
            const size_t offset =
                recv_offset + static_cast<size_t>(src_rank) * sizeof(int32_t);
            CheckerStatus read_status =
                world.UserOutput(rank).ReadBytes(offset, &actual, sizeof(actual));
            if (!read_status.ok()) {
                mismatches.push_back(MakeFailureMismatch(
                    rank, src_rank, expected_recv_counts[static_cast<size_t>(src_rank)],
                    EpMismatchContext(test_case, rank, "recv_count", src_rank) +
                        " read failure: " + read_status.message));
                return mismatches;
            }
            if (actual != expected_recv_counts[static_cast<size_t>(src_rank)]) {
                OutputMismatch mismatch;
                mismatch.rank = rank;
                mismatch.element_index = src_rank;
                mismatch.expected = expected_recv_counts[static_cast<size_t>(src_rank)];
                mismatch.actual = actual;
                mismatch.context = EpMismatchContext(test_case, rank, "recv_count", src_rank);
                mismatches.push_back(mismatch);
                if (mismatches.size() >= max_mismatches) {
                    return mismatches;
                }
            }
        }

        for (size_t local_expert = 0; local_expert < expected_expert_counts.size();
             ++local_expert) {
            int64_t actual64 = 0;
            const size_t offset = expert_offset + local_expert * sizeof(int64_t);
            CheckerStatus read_status =
                world.UserOutput(rank).ReadBytes(offset, &actual64, sizeof(actual64));
            if (!read_status.ok()) {
                mismatches.push_back(MakeFailureMismatch(
                    rank, static_cast<int64_t>(local_expert),
                    static_cast<int32_t>(expected_expert_counts[local_expert]),
                    EpMismatchContext(test_case, rank, "expert_token_count",
                                      static_cast<int64_t>(local_expert)) +
                        " read failure: " + read_status.message));
                return mismatches;
            }
            if (actual64 != expected_expert_counts[local_expert]) {
                OutputMismatch mismatch;
                mismatch.rank = rank;
                mismatch.element_index = static_cast<int64_t>(local_expert);
                mismatch.expected = static_cast<int32_t>(expected_expert_counts[local_expert]);
                mismatch.actual = static_cast<int32_t>(actual64);
                mismatch.context =
                    EpMismatchContext(test_case, rank, "expert_token_count",
                                      static_cast<int64_t>(local_expert));
                mismatches.push_back(mismatch);
                if (mismatches.size() >= max_mismatches) {
                    return mismatches;
                }
            }
        }
    }
    return mismatches;
}

std::vector<OutputMismatch> CompareEpCombineOutput(const RankWorld &world,
                                                   const CheckerCase &test_case,
                                                   const std::vector<int32_t> &expert_ids,
                                                   size_t max_mismatches) {
    std::vector<OutputMismatch> mismatches;
    if (max_mismatches == 0) {
        return mismatches;
    }
    CheckerStatus status = ValidateCase(test_case);
    if (!status.ok()) {
        mismatches.push_back(
            MakeFailureMismatch(-1, -1, 0, "ep combine validation failed: " + status.message));
        return mismatches;
    }

    for (int dst_rank = 0; dst_rank < test_case.rank_size; ++dst_rank) {
        const std::vector<EpDispatchRoute> routes =
            ExpectedEpDispatchRoutes(test_case, expert_ids, dst_rank);
        for (size_t row = 0; row < routes.size(); ++row) {
            const EpDispatchRoute &route = routes[row];
            for (int64_t h_index = 0; h_index < test_case.h; ++h_index) {
                const int64_t output_element =
                    (static_cast<int64_t>(route.token_id) * test_case.top_k +
                     route.top_k_id) *
                        test_case.h +
                    h_index;
                const size_t offset =
                    static_cast<size_t>(output_element) * sizeof(uint16_t);
                uint16_t actual = 0;
                CheckerStatus read_status =
                    world.UserOutput(route.src_rank).ReadBytes(offset, &actual,
                                                               sizeof(actual));
                const uint16_t expected =
                    EpDispatchInputValue(route.src_rank, route.token_id, h_index);
                if (!read_status.ok()) {
                    mismatches.push_back(MakeFailureMismatch(
                        route.src_rank, output_element, expected,
                        EpMismatchContext(test_case, route.src_rank, "combine_payload",
                                          output_element) +
                            " read failure: " + read_status.message));
                    return mismatches;
                }
                if (actual != expected) {
                    OutputMismatch mismatch;
                    mismatch.rank = route.src_rank;
                    mismatch.element_index = output_element;
                    mismatch.expected = expected;
                    mismatch.actual = actual;
                    mismatch.context =
                        EpMismatchContext(test_case, route.src_rank, "combine_payload",
                                          output_element);
                    mismatches.push_back(mismatch);
                    if (mismatches.size() >= max_mismatches) {
                        return mismatches;
                    }
                }
            }
        }
    }
    return mismatches;
}

}  // namespace checker
}  // namespace tilexr
