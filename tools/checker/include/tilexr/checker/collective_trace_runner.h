#ifndef TILEXR_CHECKER_COLLECTIVE_TRACE_RUNNER_H
#define TILEXR_CHECKER_COLLECTIVE_TRACE_RUNNER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tilexr/checker/case.h"
#include "tilexr/checker/status.h"
#include "tilexr/checker/world.h"

namespace tilexr {
namespace checker {

class TraceRuntime;

enum class TraceScheduleLoopOrder {
    kBlockMajor,
    kRankMajor
};

enum class TracePeerWindowMode {
    kPeerSlotZeroBase,
    kOwnerRankSlotBase
};

struct TraceScheduleSpec {
    std::string name;
    int block_num = 0;
    int rank_size = 0;
    size_t visits = 0;
    TraceScheduleLoopOrder loop_order = TraceScheduleLoopOrder::kBlockMajor;
    TracePeerWindowMode peer_window_mode = TracePeerWindowMode::kPeerSlotZeroBase;
};

size_t TraceCommVirtualBytes();
uintptr_t TraceCommVirtualBase(int rank_size, int owner_rank, int slot);
TraceScheduleSpec GetAllReduceBigDataScheduleSpec(const CheckerCase &test_case);
TraceScheduleSpec GetAllGatherHierarchyDoubleRingScheduleSpec(const CheckerCase &test_case);
TraceScheduleSpec GetEpDispatchOracleScheduleSpec(const CheckerCase &test_case);
void RegisterCollectiveTraceRanges(RankWorld *world, const CheckerCase &test_case,
                                   TraceRuntime *runtime, size_t output_bytes);
std::vector<std::vector<GM_ADDR> > InstallCollectiveTracePeerMems(RankWorld *world,
                                                                  const CheckerCase &test_case);
void RestoreCollectiveTracePeerMems(RankWorld *world, const CheckerCase &test_case,
                                    const std::vector<std::vector<GM_ADDR> > &original);

CheckerStatus RunAllReduceBigDataTrace(RankWorld *world, const CheckerCase &test_case);
CheckerStatus RunAllGatherHierarchyDoubleRingTrace(RankWorld *world,
                                                   const CheckerCase &test_case);
CheckerStatus RunEpDispatchTrace(RankWorld *world, const CheckerCase &test_case);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_COLLECTIVE_TRACE_RUNNER_H
