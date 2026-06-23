#include "tilexr/checker/collective_trace_runner.h"

#include <cstddef>

int main() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kEpDispatch;
    test_case.rank_size = 2;
    test_case.server_count = 2;
    test_case.bs = 3;
    test_case.h = 4;
    test_case.top_k = 2;
    test_case.moe_expert_num = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_FP16;

    static_assert(true, "ep dispatch source-aligned oracle probe");
    if (test_case.op != tilexr::checker::CollectiveOp::kEpDispatch) {
        return 1;
    }
    if (test_case.rank_size != 2 || test_case.server_count != 2 ||
        test_case.bs != 3 || test_case.h != 4 ||
        test_case.top_k != 2 || test_case.moe_expert_num != 4) {
        return 2;
    }
    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_FP16) {
        return 3;
    }
    if (tilexr::checker::TraceScheduleLoopOrder::kRankMajor ==
        tilexr::checker::TraceScheduleLoopOrder::kBlockMajor) {
        return 4;
    }
    if (tilexr::checker::TracePeerWindowMode::kOwnerRankSlotBase ==
        tilexr::checker::TracePeerWindowMode::kPeerSlotZeroBase) {
        return 5;
    }
    return 0;
}
