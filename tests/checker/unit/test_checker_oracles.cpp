#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "tilexr/checker/case.h"
#include "tilexr/checker/oracle.h"
#include "tilexr/checker/status.h"
#include "tilexr/checker/world.h"

namespace {
int g_failures = 0;

void ExpectTrue(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << "\n";
        ++g_failures;
    }
}

void ExpectEqInt(int actual, int expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

void ExpectEqI64(int64_t actual, int64_t expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

void ExpectEqI32(int32_t actual, int32_t expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

void ExpectEqSize(size_t actual, size_t expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        ++g_failures;
    }
}

void ExpectContains(const std::string &text, const std::string &needle, const char *message) {
    if (text.find(needle) == std::string::npos) {
        std::cerr << message << ": text=\"" << text << "\" needle=\"" << needle << "\"\n";
        ++g_failures;
    }
}

tilexr::checker::CheckerCase MakeAllGatherCase(int rank_size, int64_t count) {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllGather;
    test_case.rank_size = rank_size;
    test_case.count = count;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    test_case.magic = 17;
    return test_case;
}

tilexr::checker::CheckerCase MakeAllReduceCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.rank_size = 2;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    test_case.magic = 29;
    return test_case;
}

tilexr::checker::CheckerCase MakeEpDispatchCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kEpDispatch;
    test_case.rank_size = 2;
    test_case.server_count = 2;
    test_case.bs = 3;
    test_case.h = 4;
    test_case.top_k = 2;
    test_case.moe_expert_num = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
    test_case.scheduler = tilexr::checker::SchedulerMode::kSerial;
    test_case.magic = 31;
    return test_case;
}
}  // namespace

int main() {
    {
        tilexr::checker::CheckerCase test_case = MakeAllGatherCase(2, 3);
        tilexr::checker::RankWorld world =
            tilexr::checker::RankWorld::Create(2, 3 * sizeof(int32_t), 6 * sizeof(int32_t), 16);
        ExpectTrue(tilexr::checker::FillRankIndexInt32Inputs(&world, test_case).ok(),
                   "fill allgather rank2 inputs");
        std::vector<int32_t> expected =
            tilexr::checker::ExpectedAllGatherInt32(world, test_case, 1);
        ExpectEqSize(expected.size(), 6, "allgather rank2 expected size");
        ExpectEqI32(expected[0], tilexr::checker::RankPatternValue(0, 0),
                    "allgather rank2 element 0");
        ExpectEqI32(expected[2], tilexr::checker::RankPatternValue(0, 2),
                    "allgather rank2 last rank0 element");
        ExpectEqI32(expected[3], tilexr::checker::RankPatternValue(1, 0),
                    "allgather rank2 first rank1 element");
        ExpectEqI32(expected[5], tilexr::checker::RankPatternValue(1, 2),
                    "allgather rank2 last element");
    }

    {
        tilexr::checker::CheckerCase test_case = MakeAllGatherCase(4, 2);
        tilexr::checker::RankWorld world =
            tilexr::checker::RankWorld::Create(4, 2 * sizeof(int32_t), 8 * sizeof(int32_t), 16);
        ExpectTrue(tilexr::checker::FillRankIndexInt32Inputs(&world, test_case).ok(),
                   "fill allgather rank4 inputs");
        std::vector<int32_t> expected =
            tilexr::checker::ExpectedAllGatherInt32(world, test_case, 3);
        ExpectEqSize(expected.size(), 8, "allgather rank4 expected size");
        ExpectEqI32(expected[0], tilexr::checker::RankPatternValue(0, 0),
                    "allgather rank4 first element");
        ExpectEqI32(expected[3], tilexr::checker::RankPatternValue(1, 1),
                    "allgather rank4 second rank tail");
        ExpectEqI32(expected[6], tilexr::checker::RankPatternValue(3, 0),
                    "allgather rank4 last rank first element");
        ExpectEqI32(expected[7], tilexr::checker::RankPatternValue(3, 1),
                    "allgather rank4 last element");
    }

    {
        tilexr::checker::CheckerCase test_case = MakeAllReduceCase();
        tilexr::checker::RankWorld world =
            tilexr::checker::RankWorld::Create(2, 4 * sizeof(int32_t), 4 * sizeof(int32_t), 16);
        ExpectTrue(tilexr::checker::FillRankIndexInt32Inputs(&world, test_case).ok(),
                   "fill allreduce inputs");
        std::vector<int32_t> expected =
            tilexr::checker::ExpectedAllReduceSumInt32(world, test_case, 0);
        ExpectEqSize(expected.size(), 4, "allreduce expected size");
        for (int64_t i = 0; i < test_case.count; ++i) {
            ExpectEqI32(
                expected[static_cast<size_t>(i)],
                tilexr::checker::RankPatternValue(0, i) +
                    tilexr::checker::RankPatternValue(1, i),
                "allreduce sum expected element");
        }
    }

    {
        tilexr::checker::CheckerCase fp16_case = MakeAllGatherCase(2, 2);
        fp16_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;
        tilexr::checker::CheckerStatus fp16_status =
            tilexr::checker::ValidateCase(fp16_case);
        ExpectEqInt(static_cast<int>(fp16_status.code),
                    static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                    "fp16 returns unsupported");

        tilexr::checker::CheckerCase rank_case = MakeAllReduceCase();
        rank_case.rank_size = 4;
        tilexr::checker::CheckerStatus rank_status =
            tilexr::checker::ValidateCase(rank_case);
        ExpectEqInt(static_cast<int>(rank_status.code),
                    static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                    "unsupported rank returns unsupported");
    }

    {
        tilexr::checker::CheckerCase unsupported_case = MakeAllGatherCase(3, 2);
        tilexr::checker::RankWorld world =
            tilexr::checker::RankWorld::Create(3, 2 * sizeof(int32_t), 6 * sizeof(int32_t), 16);

        std::vector<tilexr::checker::OutputMismatch> mismatches =
            tilexr::checker::CompareInt32Output(world, unsupported_case, 1);
        ExpectEqSize(mismatches.size(), 1, "unsupported compare returns one failure record");
        if (mismatches.size() == 1) {
            ExpectEqInt(mismatches[0].rank, -1, "unsupported compare failure rank");
            ExpectEqI64(mismatches[0].element_index, -1, "unsupported compare failure index");
            ExpectEqI32(mismatches[0].expected, 0, "unsupported compare failure expected");
            ExpectEqI32(mismatches[0].actual, 0, "unsupported compare failure actual");
            ExpectTrue(!mismatches[0].context.empty(),
                       "unsupported compare failure context is non-empty");
            ExpectContains(mismatches[0].context, "allgather rank_size must be 2 or 4",
                           "unsupported compare failure context includes validation message");
        }

        mismatches = tilexr::checker::CompareInt32Output(world, unsupported_case, 0);
        ExpectEqSize(mismatches.size(), 0, "unsupported compare honors zero mismatch budget");
    }

    {
        tilexr::checker::CheckerCase hdb_small_case = MakeAllGatherCase(4, 16);
        hdb_small_case.algorithm = tilexr::checker::AlgorithmId::kAllGatherHierarchyDoubleRing;
        tilexr::checker::CheckerStatus hdb_small_status =
            tilexr::checker::ValidateCase(hdb_small_case);
        ExpectEqInt(static_cast<int>(hdb_small_status.code),
                    static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                    "hdb small production selection returns unsupported");
        ExpectContains(hdb_small_status.message, "requires data size > 32MiB or rank_size > 8",
                       "hdb small production selection message");
    }

    {
        tilexr::checker::CheckerCase hdb_case = MakeAllGatherCase(10, 16);
        hdb_case.algorithm = tilexr::checker::AlgorithmId::kAllGatherHierarchyDoubleRing;
        const tilexr::checker::AlgorithmSelectionExplanation explanation =
            tilexr::checker::ExplainAlgorithmSelection(hdb_case);
        ExpectEqInt(static_cast<int>(explanation.algorithm),
                    static_cast<int>(tilexr::checker::AlgorithmId::kAllGatherHierarchyDoubleRing),
                    "hdb selection explanation algorithm");
        ExpectTrue(explanation.eligible, "hdb selection explanation eligible");
        ExpectContains(explanation.reason, "rank_size > 8",
                       "hdb selection explanation reason");
        ExpectContains(explanation.source_file, "allgather_hierarchy_double_ring.h",
                       "hdb selection explanation source");
    }

    {
        tilexr::checker::CheckerCase arbd_case = MakeAllReduceCase();
        arbd_case.rank_size = 4;
        arbd_case.count = 16;
        arbd_case.algorithm = tilexr::checker::AlgorithmId::kAllReduceBigData;
        const tilexr::checker::AlgorithmSelectionExplanation explanation =
            tilexr::checker::ExplainAlgorithmSelection(arbd_case);
        ExpectEqInt(static_cast<int>(explanation.algorithm),
                    static_cast<int>(tilexr::checker::AlgorithmId::kAllReduceBigData),
                    "arbd selection explanation algorithm");
        ExpectTrue(!explanation.eligible, "arbd small selection explanation ineligible");
        ExpectContains(explanation.reason, "data size < 2MiB",
                       "arbd small selection explanation reason");
        ExpectContains(explanation.source_file, "allreduce_big_data.h",
                       "arbd selection explanation source");
    }

    {
        tilexr::checker::CheckerCase ep_case = MakeEpDispatchCase();
        const tilexr::checker::AlgorithmSelectionExplanation explanation =
            tilexr::checker::ExplainAlgorithmSelection(ep_case);
        ExpectEqInt(static_cast<int>(explanation.algorithm),
                    static_cast<int>(tilexr::checker::AlgorithmId::kDefault),
                    "ep selection explanation algorithm");
        ExpectTrue(explanation.eligible, "ep selection explanation eligible");
        ExpectContains(explanation.reason, "source-aligned CPU oracle",
                       "ep selection explanation reason");
        ExpectContains(explanation.source_file, "tilexr_ep_dispatch_kernel.cpp",
                       "ep selection explanation source");
    }

    {
        tilexr::checker::CheckerCase ep_case = MakeEpDispatchCase();
        ep_case.op = tilexr::checker::CollectiveOp::kEpCombine;
        tilexr::checker::CheckerStatus ep_status = tilexr::checker::ValidateCase(ep_case);
        ExpectEqInt(static_cast<int>(ep_status.code),
                    static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                    "ep combine valid case");

        const tilexr::checker::AlgorithmSelectionExplanation explanation =
            tilexr::checker::ExplainAlgorithmSelection(ep_case);
        ExpectEqInt(static_cast<int>(explanation.algorithm),
                    static_cast<int>(tilexr::checker::AlgorithmId::kDefault),
                    "ep combine selection explanation algorithm");
        ExpectTrue(explanation.eligible, "ep combine selection explanation eligible");
        ExpectContains(explanation.reason, "source-aligned CPU oracle",
                       "ep combine selection explanation reason");
        ExpectContains(explanation.source_file, "tilexr_ep_combine_kernel.cpp",
                       "ep combine selection explanation source");
    }

    {
        tilexr::checker::CheckerCase ep_case = MakeEpDispatchCase();
        tilexr::checker::CheckerStatus ep_status = tilexr::checker::ValidateCase(ep_case);
        ExpectEqInt(static_cast<int>(ep_status.code),
                    static_cast<int>(tilexr::checker::CheckerStatusCode::kOk),
                    "ep dispatch valid case");

        std::vector<int32_t> expert_ids =
            tilexr::checker::DefaultEpDispatchExpertIds(ep_case);
        ExpectEqSize(expert_ids.size(), 6, "ep dispatch default route count");
        ExpectEqI32(expert_ids[0], 0, "ep route token0 top0");
        ExpectEqI32(expert_ids[1], 3, "ep route token0 top1");
        ExpectEqI32(tilexr::checker::EpDispatchInputValue(1, 2, 3), 0x1123,
                    "ep dispatch input pattern");

        const std::vector<tilexr::checker::EpDispatchRoute> rank1_routes =
            tilexr::checker::ExpectedEpDispatchRoutes(ep_case, expert_ids, 1);
        ExpectEqSize(rank1_routes.size(), 6, "ep dispatch rank1 route count");
        if (!rank1_routes.empty()) {
            ExpectEqInt(rank1_routes[0].src_rank, 0, "ep dispatch route src");
            ExpectEqInt(rank1_routes[0].token_id, 0, "ep dispatch route token");
            ExpectEqInt(rank1_routes[0].top_k_id, 1, "ep dispatch route topk");
            ExpectEqInt(rank1_routes[0].expert_id, 3, "ep dispatch route expert");
        }

        ExpectEqSize(tilexr::checker::EpDispatchMetadataBytes(
                         ep_case, tilexr::checker::EpDispatchMetadataRole::kAssist),
                     static_cast<size_t>(ep_case.bs * ep_case.top_k * 4 * sizeof(int32_t)),
                     "ep dispatch assist metadata bytes");
        ExpectEqSize(tilexr::checker::EpDispatchMetadataBytes(
                         ep_case, tilexr::checker::EpDispatchMetadataRole::kRecvCounts),
                     static_cast<size_t>(ep_case.rank_size * sizeof(int32_t)),
                     "ep dispatch recv count metadata bytes");
        ExpectEqSize(tilexr::checker::EpDispatchMetadataBytes(
                         ep_case, tilexr::checker::EpDispatchMetadataRole::kExpertTokenNums),
                     static_cast<size_t>((ep_case.moe_expert_num / ep_case.rank_size) *
                                         sizeof(int64_t)),
                     "ep dispatch expert count metadata bytes");

        tilexr::checker::EpDispatchWindowLayout layout;
        tilexr::checker::CheckerStatus layout_status =
            tilexr::checker::ComputeEpDispatchWindowLayout(ep_case, &layout);
        ExpectTrue(layout_status.ok(), "ep dispatch window layout status");
        ExpectEqI64(layout.rank_size, 2, "ep window rank size");
        ExpectEqI64(layout.local_expert_num, 2, "ep window local expert num");
        ExpectEqI64(layout.dtype_bytes, 2, "ep window dtype bytes");
        ExpectEqI64(layout.max_routes_per_src, 6, "ep window max routes per src");
        ExpectEqI64(layout.row_bytes, 8, "ep window row bytes");
        ExpectEqI64(layout.payload_bytes_per_slot, 64, "ep window aligned payload bytes");
        ExpectEqI64(layout.assist_bytes_per_slot, 96, "ep window aligned assist bytes");
        ExpectEqI64(layout.slot_bytes, 224, "ep window slot bytes");
        ExpectEqI64(layout.total_bytes, 512, "ep window total bytes");
    }

    {
        tilexr::checker::CheckerCase test_case = MakeAllGatherCase(2, 4);
        tilexr::checker::RankWorld world =
            tilexr::checker::RankWorld::Create(2, 4 * sizeof(int32_t), 8 * sizeof(int32_t), 16);
        ExpectTrue(tilexr::checker::FillRankIndexInt32Inputs(&world, test_case).ok(),
                   "fill inputs before mismatch compare");

        std::vector<int32_t> expected =
            tilexr::checker::ExpectedAllGatherInt32(world, test_case, 0);
        for (size_t i = 0; i < expected.size(); ++i) {
            ExpectTrue(world.UserOutput(0).WriteInt32(i, expected[i]).ok(),
                       "seed user output with expected values");
        }
        ExpectTrue(world.UserOutput(0).WriteInt32(1, -1).ok(), "inject mismatch 1");
        ExpectTrue(world.UserOutput(0).WriteInt32(3, -2).ok(), "inject mismatch 2");
        ExpectTrue(world.UserOutput(0).WriteInt32(5, -3).ok(), "inject mismatch 3");

        std::vector<tilexr::checker::OutputMismatch> mismatches =
            tilexr::checker::CompareInt32Output(world, test_case, 2);
        ExpectEqSize(mismatches.size(), 2, "mismatch list truncates");
        ExpectEqInt(mismatches[0].rank, 0, "first mismatch rank");
        ExpectEqI64(mismatches[0].element_index, 1, "first mismatch index");
        ExpectEqI32(mismatches[0].actual, -1, "first mismatch actual");
        ExpectEqInt(mismatches[1].rank, 0, "second mismatch rank");
        ExpectEqI64(mismatches[1].element_index, 3, "second mismatch index");
        ExpectEqI32(mismatches[1].actual, -2, "second mismatch actual");
    }

    {
        tilexr::checker::CheckerCase test_case = MakeAllGatherCase(2, 4);
        tilexr::checker::RankWorld world =
            tilexr::checker::RankWorld::Create(2, 4 * sizeof(int32_t), 3 * sizeof(int32_t), 16);
        ExpectTrue(tilexr::checker::FillRankIndexInt32Inputs(&world, test_case).ok(),
                   "fill inputs before undersized output compare");

        std::vector<int32_t> expected =
            tilexr::checker::ExpectedAllGatherInt32(world, test_case, 0);
        for (size_t i = 0; i < 3; ++i) {
            ExpectTrue(world.UserOutput(0).WriteInt32(i, expected[i]).ok(),
                       "seed readable undersized output values");
        }

        std::vector<tilexr::checker::OutputMismatch> mismatches =
            tilexr::checker::CompareInt32Output(world, test_case, 1);
        ExpectEqSize(mismatches.size(), 1, "undersized output returns one failure record");
        if (mismatches.size() == 1) {
            ExpectEqInt(mismatches[0].rank, 0, "undersized output failure rank");
            ExpectEqI64(mismatches[0].element_index, 3,
                        "undersized output first unreadable index");
            ExpectEqI32(mismatches[0].expected, expected[3], "undersized output failure expected");
            ExpectEqI32(mismatches[0].actual, 0, "undersized output failure actual");
            ExpectContains(mismatches[0].context, "byte buffer access out of bounds",
                           "undersized output failure context includes read error");
        }

        mismatches = tilexr::checker::CompareInt32Output(world, test_case, 0);
        ExpectEqSize(mismatches.size(), 0, "undersized output honors zero mismatch budget");
    }

    return g_failures == 0 ? 0 : 1;
}
