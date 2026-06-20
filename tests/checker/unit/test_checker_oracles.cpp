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
