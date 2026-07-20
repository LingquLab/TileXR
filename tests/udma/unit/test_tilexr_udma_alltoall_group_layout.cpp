#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include "demo/tilexr_udma_alltoall_group_layout.h"

namespace {

#ifndef TILEXR_SOURCE_ROOT
#define TILEXR_SOURCE_ROOT "."
#endif

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

#define CHECK_NOT_CONTAINS(text, needle) \
    do { \
        if ((text).find(needle) != std::string::npos) { \
            std::cerr << "CHECK_NOT_CONTAINS failed at line " << __LINE__ << ": " << needle << std::endl; \
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

void CheckSchedule(int rankSize)
{
    for (int rank = 0; rank < rankSize; ++rank) {
        std::set<int> peers;
        for (uint32_t group = 0; group < TileXR::Demo::AllToAllGroupCount(rankSize); ++group) {
            for (uint32_t lane = 0; lane < TileXR::Demo::kAllToAllGroupWidth; ++lane) {
                const int peer = TileXR::Demo::AllToAllGroupPeer(rank, rankSize, group, lane);
                if (peer < 0) {
                    continue;
                }
                CHECK_EQ(peer == rank, false);
                CHECK_EQ(peers.insert(peer).second, true);
                bool symmetric = false;
                for (uint32_t remoteLane = 0; remoteLane < TileXR::Demo::kAllToAllGroupWidth;
                     ++remoteLane) {
                    symmetric = symmetric ||
                        TileXR::Demo::AllToAllGroupPeer(peer, rankSize, group, remoteLane) == rank;
                }
                CHECK_EQ(symmetric, true);
            }
        }
        CHECK_EQ(peers.size(), static_cast<size_t>(rankSize - 1));
    }
}

void TestSchedules()
{
    for (int rankSize : {8, 16, 24, 32, 40, 64, 128}) {
        CheckSchedule(rankSize);
    }
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(8), 1U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(16), 1U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(24), 2U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(64), 4U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(128), 8U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(7), 0U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(18), 0U);
    CHECK_EQ(TileXR::Demo::AllToAllGroupCount(136), 0U);

    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 0), 1);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 7), 8);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 8), 63);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 0, 15), 56);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 3, 7), 32);
    CHECK_EQ(TileXR::Demo::AllToAllGroupPeer(0, 64, 3, 15), -1);
}

void TestPlan()
{
    constexpr int rankSize = 16;
    constexpr int32_t elementsPerPeer = 2 * 1024 * 1024;
    const auto plan = TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, elementsPerPeer);

    CHECK_EQ(plan.valid, true);
    CHECK_EQ(plan.groupCount, 1U);
    CHECK_EQ(plan.passCount, 1U);
    CHECK_EQ(plan.chunkElements, elementsPerPeer);
    CHECK_EQ(plan.bytesPerPeer, 8ULL * 1024ULL * 1024ULL);
    CHECK_EQ(plan.payloadPlaneBytes, 128ULL * 1024ULL * 1024ULL);
    CHECK_EQ(plan.payloadOffset[0], 0ULL);
    CHECK_EQ(plan.payloadOffset[1] >= plan.payloadOffset[0] + plan.payloadPlaneBytes, true);
    CHECK_EQ(plan.signalPlaneBytes, static_cast<size_t>(rankSize) * 128ULL);
    CHECK_EQ(plan.signalOffset[0] >= plan.payloadOffset[1] + plan.payloadPlaneBytes, true);
    CHECK_EQ(plan.signalOffset[1] >= plan.signalOffset[0] + plan.signalPlaneBytes, true);
    CHECK_EQ(plan.controlOffset >= plan.signalOffset[1] + plan.signalPlaneBytes, true);
    CHECK_EQ(plan.registeredBytes <= TileXR::Demo::kAllToAllGroupMaxRegisteredBytes, true);

    const auto chunked = TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, elementsPerPeer / 4);
    CHECK_EQ(chunked.valid, true);
    CHECK_EQ(chunked.passCount, 4U);
    CHECK_EQ(chunked.payloadPlaneBytes, plan.payloadPlaneBytes);

    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(7, 1024, 1024).valid, false);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(18, 1024, 1024).valid, false);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(136, 1024, 1024).valid, false);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(16, 0, 1024).valid, false);
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(16, elementsPerPeer, 0).valid, false);

    constexpr int32_t thirtyTwoMiBElements = 8 * 1024 * 1024;
    CHECK_EQ(TileXR::Demo::PlanAllToAllGroup(
        rankSize, thirtyTwoMiBElements, thirtyTwoMiBElements).valid, false);
}

void TestTokens()
{
    const uint64_t token48 = TileXR::Demo::AllToAllGroupToken(48U, 0U, 0U);
    const uint64_t token49 = TileXR::Demo::AllToAllGroupToken(49U, 0U, 0U);
    CHECK_EQ(token48 != 0U, true);
    CHECK_EQ(token49 > token48, true);
    CHECK_EQ((token48 >> 31U) & 1ULL, 0ULL);
    CHECK_EQ((token49 >> 31U) & 1ULL, 1ULL);
    CHECK_EQ(TileXR::Demo::AllToAllGroupToken(49U, 1U, 0U) > token49, true);
    CHECK_EQ(TileXR::Demo::AllToAllGroupToken(49U, 1U, 1U) >
        TileXR::Demo::AllToAllGroupToken(49U, 1U, 0U), true);
}

void TestKernelStructure()
{
    const std::string kernel = ReadFile(
        std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp");
    CHECK_CONTAINS(kernel, "tilexr_udma_all_to_all_group_kernel");
    CHECK_CONTAINS(kernel, "TILEXR_ALLTOALL_GROUP_SEND_CORES");
    CHECK_CONTAINS(kernel, "UDMAPutSignalNbiOnQp<int32_t>");
    CHECK_CONTAINS(kernel, "UDMAQuietStatusOnQp(args, peer, qpIdx)");
    CHECK_CONTAINS(kernel, "AllToAllGroupWaitTokenMte");
    CHECK_CONTAINS(kernel, "observed >= expectedToken");
    CHECK_CONTAINS(kernel, "launch_tilexr_udma_all_to_all_group");
    CHECK_NOT_CONTAINS(kernel, "UDMAPutSignalNbi<int32_t>");
    CHECK_NOT_CONTAINS(kernel, "SyncAll");
}

void TestHostStructure()
{
    const std::string demo = ReadFile(
        std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/demo/tilexr_udma_demo.cpp");
    CHECK_CONTAINS(demo, "#include \"tilexr_udma_alltoall_group_layout.h\"");
    CHECK_CONTAINS(demo, "testType == 8");
    CHECK_CONTAINS(demo, "RunGroupedAllToAll");
    CHECK_CONTAINS(demo, "PlanAllToAllGroup");
    CHECK_CONTAINS(demo, "launch_tilexr_udma_all_to_all_group");
    CHECK_CONTAINS(demo, "TILEXR_DEMO_ALLTOALL_GROUP_CHUNK_ELEMENTS");
    CHECK_CONTAINS(demo, "grouped alltoall registeredBytes=");
    CHECK_CONTAINS(demo, "grouped alltoall warmup=");
    const size_t begin = demo.find("bool RunGroupedAllToAll(");
    const size_t end = demo.find("void Cleanup(", begin);
    const std::string grouped = begin == std::string::npos ? std::string() :
        demo.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    CHECK_NOT_CONTAINS(grouped, "DemoBarrierAll");
}

} // namespace

int main()
{
    TestSchedules();
    TestPlan();
    TestTokens();
    TestKernelStructure();
    TestHostStructure();
    if (g_failures != 0) {
        std::cerr << "TileXR grouped all-to-all layout checks failed: " << g_failures << std::endl;
        return 1;
    }
    std::cout << "TileXR grouped all-to-all layout checks passed" << std::endl;
    return 0;
}
