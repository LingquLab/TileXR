#ifndef TILEXR_EP_PLANNER_COMMON_EP_PLAN_PEER_MAILBOX_H
#define TILEXR_EP_PLANNER_COMMON_EP_PLAN_PEER_MAILBOX_H

#include <cstdint>

#include "comm_args.h"
#include "ep_plan_types.h"

#ifndef TILEXR_PLAN_MAILBOX_FN
#define TILEXR_PLAN_MAILBOX_FN inline
#endif

namespace TileXREp {
namespace Plan {

// Match the proven IPC/MTE transfer unit used by planner barrier flags.
constexpr uint64_t kPlanPeerMailboxTransferBytes = 512;
constexpr uint64_t kPlanPeerMailboxRowBytes =
    static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE) / TileXR::TILEXR_MAX_RANK_SIZE;

static_assert(kPlanPeerMailboxRowBytes % kPlanPeerMailboxTransferBytes == 0,
    "peer mailbox rows must preserve the MTE transfer alignment");
static_assert(kPlanPeerMailboxRowBytes * TileXR::TILEXR_MAX_RANK_SIZE ==
    static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE),
    "fixed peer mailbox rows must exactly partition the IPC data window");

struct PlanPeerMailboxLayout {
    uint64_t header;
    uint64_t globalRankId;
    uint64_t status;
    uint64_t tpe;
    uint64_t inputBytes;
    uint64_t rowBytes;
    uint64_t totalBytes;
};

TILEXR_PLAN_MAILBOX_FN uint64_t AlignPlanPeerMailbox(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

TILEXR_PLAN_MAILBOX_FN PlanPeerMailboxLayout BuildPlanPeerMailboxLayout(int64_t rankSize, int64_t expertNum)
{
    PlanPeerMailboxLayout layout {};
    uint64_t cursor = 0;
    layout.header = AlignPlanPeerMailbox(cursor, kPlanWorkspaceAlignment);
    cursor = layout.header + kPlanHeaderStrideBytes;
    layout.globalRankId = AlignPlanPeerMailbox(cursor, kPlanWorkspaceAlignment);
    cursor = layout.globalRankId + sizeof(int32_t);
    layout.status = AlignPlanPeerMailbox(cursor, kPlanWorkspaceAlignment);
    cursor = layout.status + kPlanStatusStrideBytes;
    layout.tpe = AlignPlanPeerMailbox(cursor, kPlanWorkspaceAlignment);
    layout.rowBytes = kPlanPeerMailboxRowBytes;
    layout.totalBytes = rankSize > 0
        ? static_cast<uint64_t>(rankSize) * layout.rowBytes : UINT64_MAX;

    const uint64_t expertCapacity = (layout.rowBytes - layout.tpe) / sizeof(int32_t);
    layout.inputBytes = expertNum > 0 && static_cast<uint64_t>(expertNum) <= expertCapacity
        ? AlignPlanPeerMailbox(layout.tpe + static_cast<uint64_t>(expertNum) * sizeof(int32_t),
            kPlanPeerMailboxTransferBytes)
        : UINT64_MAX;
    return layout;
}

TILEXR_PLAN_MAILBOX_FN uint64_t PlanPeerMailboxRowOffset(
    const PlanPeerMailboxLayout &layout, int64_t sourceRank)
{
    return static_cast<uint64_t>(sourceRank) * layout.rowBytes;
}

} // namespace Plan
} // namespace TileXREp

#endif // TILEXR_EP_PLANNER_COMMON_EP_PLAN_PEER_MAILBOX_H
