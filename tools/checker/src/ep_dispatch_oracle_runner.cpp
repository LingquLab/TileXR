#include "tilexr/checker/collective_trace_runner.h"

#include <cstddef>

namespace tilexr {
namespace checker {

TraceScheduleSpec GetEpDispatchOracleScheduleSpec(const CheckerCase &test_case) {
    TraceScheduleSpec spec;
    spec.name = "ep_dispatch_rank_major";
    spec.rank_size = test_case.rank_size;
    spec.block_num = test_case.rank_size;
    spec.visits = static_cast<size_t>(test_case.rank_size);
    spec.loop_order = TraceScheduleLoopOrder::kRankMajor;
    spec.peer_window_mode = TracePeerWindowMode::kOwnerRankSlotBase;
    return spec;
}

CheckerStatus RunEpDispatchTrace(RankWorld *world, const CheckerCase &test_case) {
    (void)world;
    (void)test_case;
    return CheckerStatus::Unsupported(
        "ep_dispatch_source_aligned_cpu_oracle is executed through CollectiveExecutor");
}

}  // namespace checker
}  // namespace tilexr
