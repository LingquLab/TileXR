#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "demo/tilexr_udma_allreduce_layout.h"

#ifndef TILEXR_SOURCE_ROOT
#define TILEXR_SOURCE_ROOT "."
#endif

namespace {

int g_failures = 0;

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << lhsValue << " vs " << rhsValue << ")" << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_CONTAINS(text, needle) \
    do { \
        if ((text).find(needle) == std::string::npos) { \
            std::cerr << "CHECK_CONTAINS failed at line " << __LINE__ << ": " << needle << std::endl; \
            ++g_failures; \
        } \
    } while (0)

std::string ReadFile(const std::string& path)
{
    std::ifstream in(path.c_str());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void TestAllReduceInputPattern()
{
    constexpr int rank = 3;
    constexpr int32_t elementsPerRank = 4;
    std::vector<int32_t> input(elementsPerRank * 2, -1);

    TileXR::Demo::FillAllReduceInput(input, rank, elementsPerRank);

    for (int32_t i = 0; i < elementsPerRank; ++i) {
        CHECK_EQ(input[static_cast<size_t>(i)], TileXR::Demo::AllReduceValue(rank));
    }
    CHECK_EQ(input[static_cast<size_t>(elementsPerRank)], -1);
}

void TestAllReduceExpectedEightRankSum()
{
    CHECK_EQ(TileXR::Demo::AllReduceExpectedSum(8), 8028);
}

void TestAllReduceOutputValidation()
{
    constexpr int rankSize = 4;
    constexpr int32_t elementsPerRank = 3;
    std::vector<int32_t> output(elementsPerRank, TileXR::Demo::AllReduceExpectedSum(rankSize));

    CHECK_EQ(TileXR::Demo::ValidateAllReduceOutput(output, rankSize, elementsPerRank), true);
    output[1] = -1;
    CHECK_EQ(TileXR::Demo::ValidateAllReduceOutput(output, rankSize, elementsPerRank), false);
}

void TestBuildAllReduceOutput()
{
    constexpr int rankSize = 5;
    constexpr int32_t elementsPerRank = 2;
    std::vector<int32_t> allInputs(static_cast<size_t>(rankSize) * elementsPerRank, -1);

    for (int rank = 0; rank < rankSize; ++rank) {
        std::fill(allInputs.begin() + static_cast<size_t>(rank) * elementsPerRank,
                  allInputs.begin() + static_cast<size_t>(rank + 1) * elementsPerRank,
                  TileXR::Demo::AllReduceValue(rank));
    }

    std::vector<int32_t> output(elementsPerRank, -1);
    TileXR::Demo::BuildAllReduceOutputFromInputs(allInputs, rankSize, elementsPerRank, output);

    CHECK_EQ(TileXR::Demo::ValidateAllReduceOutput(output, rankSize, elementsPerRank), true);
}

void TestDemoAllReduceSourceHooks()
{
    const std::string demo =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    const std::string kernel =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo_kernel.cpp");
    const std::string script =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/run_tilexr_udma_demo.sh");

    CHECK_CONTAINS(demo, "testType == 3");
    CHECK_CONTAINS(demo, "ValidateAllReduceData");
    CHECK_CONTAINS(kernel, "tilexr_all_reduce_ipc_scatter_kernel");
    CHECK_CONTAINS(kernel, "tilexr_all_reduce_ipc_sum_kernel");
    CHECK_CONTAINS(script, "3=all-reduce");
}

} // namespace

int main()
{
    TestAllReduceInputPattern();
    TestAllReduceExpectedEightRankSum();
    TestAllReduceOutputValidation();
    TestBuildAllReduceOutput();
    TestDemoAllReduceSourceHooks();
    if (g_failures != 0) {
        std::cerr << g_failures << " all-reduce layout checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA all-reduce layout checks passed" << std::endl;
    return 0;
}
