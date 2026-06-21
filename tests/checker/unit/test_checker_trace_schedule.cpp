#include <iostream>
#include <string>

#include "tilexr/checker/collective_trace_runner.h"

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

void ExpectEqString(const std::string &actual, const std::string &expected,
                    const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=\"" << actual
                  << "\" expected=\"" << expected << "\"\n";
        ++g_failures;
    }
}

tilexr::checker::CheckerCase MakeAllReduceBigDataCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.algorithm = tilexr::checker::AlgorithmId::kAllReduceBigData;
    test_case.rank_size = 4;
    test_case.server_count = 2;
    test_case.count = 524288;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    return test_case;
}

tilexr::checker::CheckerCase MakeAllGatherHdbCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllGather;
    test_case.algorithm = tilexr::checker::AlgorithmId::kAllGatherHierarchyDoubleRing;
    test_case.rank_size = 10;
    test_case.server_count = 2;
    test_case.count = 16;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    return test_case;
}

void TestAllReduceBigDataScheduleSpec() {
    tilexr::checker::TraceScheduleSpec spec =
        tilexr::checker::GetAllReduceBigDataScheduleSpec(MakeAllReduceBigDataCase());

    ExpectEqString(spec.name, "allreduce_big_data_block_major",
                   "allreduce schedule name");
    ExpectEqInt(spec.block_num, 8, "allreduce schedule block num");
    ExpectEqInt(static_cast<int>(spec.loop_order),
                static_cast<int>(tilexr::checker::TraceScheduleLoopOrder::kBlockMajor),
                "allreduce schedule loop order");
    ExpectEqInt(static_cast<int>(spec.peer_window_mode),
                static_cast<int>(tilexr::checker::TracePeerWindowMode::kPeerSlotZeroBase),
                "allreduce schedule peer window mode");
    ExpectTrue(spec.visits == 32, "allreduce schedule visit count");
}

void TestAllGatherHdbScheduleSpec() {
    tilexr::checker::TraceScheduleSpec spec =
        tilexr::checker::GetAllGatherHierarchyDoubleRingScheduleSpec(MakeAllGatherHdbCase());

    ExpectEqString(spec.name, "allgather_hierarchy_double_ring_stage_major",
                   "hdb schedule name");
    ExpectEqInt(spec.block_num, 40, "hdb schedule block num");
    ExpectEqInt(static_cast<int>(spec.loop_order),
                static_cast<int>(tilexr::checker::TraceScheduleLoopOrder::kBlockMajor),
                "hdb schedule loop order");
    ExpectEqInt(static_cast<int>(spec.peer_window_mode),
                static_cast<int>(tilexr::checker::TracePeerWindowMode::kPeerSlotZeroBase),
                "hdb schedule peer window mode");
    ExpectTrue(spec.visits == 400, "hdb schedule visit count");
}

}  // namespace

int main() {
    TestAllReduceBigDataScheduleSpec();
    TestAllGatherHdbScheduleSpec();
    return g_failures == 0 ? 0 : 1;
}
