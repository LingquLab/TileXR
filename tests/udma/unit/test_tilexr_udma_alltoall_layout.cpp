#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "demo/tilexr_udma_alltoall_layout.h"

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

void TestAllToAllInputPattern()
{
    constexpr int rank = 2;
    constexpr int rankSize = 4;
    constexpr int32_t elementsPerPeer = 3;
    std::vector<int32_t> input(static_cast<size_t>(rankSize) * elementsPerPeer, -1);

    TileXR::Demo::FillAllToAllInput(input, rank, rankSize, elementsPerPeer);

    for (int dstRank = 0; dstRank < rankSize; ++dstRank) {
        int32_t expected = TileXR::Demo::AllToAllValue(rank, dstRank);
        for (int32_t elem = 0; elem < elementsPerPeer; ++elem) {
            CHECK_EQ(input[static_cast<size_t>(dstRank) * elementsPerPeer + elem], expected);
        }
    }
}

void TestAllToAllOutputValidation()
{
    constexpr int rank = 1;
    constexpr int rankSize = 3;
    constexpr int32_t elementsPerPeer = 2;
    std::vector<int32_t> output(static_cast<size_t>(rankSize) * elementsPerPeer, -1);

    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        std::fill(output.begin() + static_cast<size_t>(srcRank) * elementsPerPeer,
                  output.begin() + static_cast<size_t>(srcRank + 1) * elementsPerPeer,
                  TileXR::Demo::AllToAllValue(srcRank, rank));
    }

    CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(output, rank, rankSize, elementsPerPeer), true);
    output[static_cast<size_t>(2) * elementsPerPeer + 1] = 123;
    CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(output, rank, rankSize, elementsPerPeer), false);
}

void TestBuildAllToAllOutput()
{
    constexpr int rankSize = 3;
    constexpr int32_t elementsPerPeer = 2;
    std::vector<int32_t> allInputs(static_cast<size_t>(rankSize) * rankSize * elementsPerPeer, -1);

    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        std::vector<int32_t> oneInput(static_cast<size_t>(rankSize) * elementsPerPeer, -1);
        TileXR::Demo::FillAllToAllInput(oneInput, srcRank, rankSize, elementsPerPeer);
        std::copy(oneInput.begin(), oneInput.end(),
                  allInputs.begin() + static_cast<size_t>(srcRank) * rankSize * elementsPerPeer);
    }

    std::vector<int32_t> output(static_cast<size_t>(rankSize) * elementsPerPeer, -1);
    TileXR::Demo::BuildAllToAllOutputFromInputs(allInputs, 2, rankSize, elementsPerPeer, output);

    CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(output, 2, rankSize, elementsPerPeer), true);
}

void TestAllToAllMaxRank256With64MiBPerRank()
{
    constexpr int rankSize = 256;
    constexpr size_t perRankBytes = 64ULL * 1024ULL * 1024ULL;
    constexpr int32_t elementsPerPeer =
        static_cast<int32_t>(perRankBytes / (sizeof(int32_t) * rankSize));
    CHECK_EQ(elementsPerPeer, 65536);

    std::vector<int32_t> buffer(static_cast<size_t>(rankSize) * elementsPerPeer, -1);
    const int sampleRanks[] = {0, 1, 127, 255};
    for (int rank : sampleRanks) {
        TileXR::Demo::FillAllToAllInput(buffer, rank, rankSize, elementsPerPeer);
        CHECK_EQ(buffer[0], TileXR::Demo::AllToAllValue(rank, 0));
        CHECK_EQ(buffer[static_cast<size_t>(rankSize - 1) * elementsPerPeer],
                 TileXR::Demo::AllToAllValue(rank, rankSize - 1));
        CHECK_EQ(buffer[static_cast<size_t>(rankSize) * elementsPerPeer - 1],
                 TileXR::Demo::AllToAllValue(rank, rankSize - 1));

        for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
            std::fill(buffer.begin() + static_cast<size_t>(srcRank) * elementsPerPeer,
                      buffer.begin() + static_cast<size_t>(srcRank + 1) * elementsPerPeer,
                      TileXR::Demo::AllToAllValue(srcRank, rank));
        }
        CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(buffer, rank, rankSize, elementsPerPeer), true);
        buffer[static_cast<size_t>(rankSize) * elementsPerPeer - 1] = -1;
        CHECK_EQ(TileXR::Demo::ValidateAllToAllOutput(buffer, rank, rankSize, elementsPerPeer), false);
    }
}

void TestDemoDebugLayoutSource()
{
    const std::string demo =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    const std::string kernel =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo_kernel.cpp");

    CHECK_CONTAINS(demo, "kDebugUdmaStatusBase + TileXR::TILEXR_MAX_RANK_SIZE");
    CHECK_CONTAINS(demo, "kDebugIpcGather + 1");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + TileXR::TILEXR_MAX_RANK_SIZE");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer");
}

void TestAllToAllDataAsFlagSource()
{
    const std::string demo =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    const std::string kernel =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo_kernel.cpp");

    CHECK_CONTAINS(demo, "useAllToAllDataAsFlagIpc");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_USE_UDMA");
    CHECK_CONTAINS(demo, "skip TileXRUDMARegister for alltoall data-as-flag IPC path");
    CHECK_CONTAINS(demo, "forceAllToAllIpcFallback");
    CHECK_CONTAINS(demo, "strictAllToAllUdma");
    CHECK_CONTAINS(demo, "ERROR: strict alltoall UDMA registration failed");
    CHECK_CONTAINS(demo, "ERROR: strict alltoall UDMA CQ incomplete");
    CHECK_CONTAINS(demo, "TileXRUDMARegister failed; use alltoall data-as-flag IPC fallback");
    CHECK_CONTAINS(demo, "skip all-to-all UDMA kernel; use data-as-flag IPC fallback");
    CHECK_CONTAINS(kernel, "#include \"tilexr_data_as_flag.h\"");
    CHECK_CONTAINS(kernel, "TILEXR_UDMA_DEMO_DATA_AS_FLAG_STAGING_OFFSET");
    CHECK_CONTAINS(kernel, "DataAsFlagBlockCountForPayloadBytes");
    CHECK_CONTAINS(kernel, "DataAsFlagInit");
    CHECK_CONTAINS(kernel, "DataAsFlagSend");
    CHECK_CONTAINS(kernel, "DataAsFlagCheckAndRecv");
}

} // namespace

int main()
{
    TestAllToAllInputPattern();
    TestAllToAllOutputValidation();
    TestBuildAllToAllOutput();
    TestAllToAllMaxRank256With64MiBPerRank();
    TestDemoDebugLayoutSource();
    TestAllToAllDataAsFlagSource();
    if (g_failures != 0) {
        std::cerr << g_failures << " all-to-all layout checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA all-to-all layout checks passed" << std::endl;
    return 0;
}
