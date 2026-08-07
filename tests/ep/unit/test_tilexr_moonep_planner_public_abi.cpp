#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "tilexr_moonep.h"
#include "tilexr_moonep_planner.h"
#include "tilexr_ep_plan.h"

using WorkspaceQueryV1 = int (*)(TileXRCommPtr, int64_t, int64_t, int64_t,
    uint64_t *, int64_t *);
using WorkspaceQueryV2 = int (*)(TileXRCommPtr, int64_t, int64_t, int64_t,
    uint64_t *, int64_t *);
using PlannerV2 = int (*)(const int32_t *, const int32_t *, TileXRCommPtr,
    int64_t, int64_t, int64_t, void *, uint64_t, int32_t *, int32_t *,
    int32_t *, int32_t *, int32_t *, uint64_t, aclrtStream);
using OptimizedWorkspaceQueryV2 = int (*)(int64_t, int64_t, int64_t, int64_t,
    const TileXRMoonEPPlanConfig *, uint64_t *, uint64_t *);
using OptimizedPlannerV2 = int (*)(const int32_t *, const int32_t *, const int32_t *,
    TileXRCommPtr, int64_t, int64_t, int64_t, const TileXRMoonEPPlanConfig *,
    TileXRMoonEPPlanDesc *, void *, uint64_t, void *, uint64_t, aclrtStream);
using OptimizedPlannerV2WithMetadata = int (*)(const int32_t *, const int32_t *,
    const int32_t *, TileXRCommPtr, int64_t, int64_t, int64_t,
    const TileXRMoonEPPlanConfig *, TileXRMoonEPPlanMetadataV2 *, void *,
    uint64_t, void *, uint64_t, aclrtStream);
using EncodeGlobalTokenId = TileXRMoonEPPlanStatus (*)(int32_t, int32_t, int32_t,
    int64_t, int64_t, int64_t, uint64_t *);
using DecodeGlobalTokenId = TileXRMoonEPPlanStatus (*)(uint64_t, int64_t, int64_t,
    int64_t, int32_t *, int32_t *, int32_t *);
using DecodeDst = TileXRMoonEPPlanStatus (*)(int32_t, int64_t, int64_t,
    TileXREp::Plan::MoonEPRouteTarget *);
using BuildRouteDescriptor = TileXRMoonEPPlanStatus (*)(int32_t, int32_t, int32_t,
    int32_t, int64_t, int64_t, int64_t, int64_t, TileXREp::Plan::MoonEPRouteDescriptor *);

static_assert(std::is_same<decltype(&TileXRMoonEpPlanningGetWorkspaceSizeV1),
    WorkspaceQueryV1>::value, "stage V1 workspace ABI changed");
static_assert(std::is_same<decltype(&TileXRMoonEpPlannerGetWorkspaceSizeV2),
    WorkspaceQueryV2>::value, "Planner workspace V2 ABI changed");
static_assert(std::is_same<decltype(&TileXRMoonEpPlannerV2),
    PlannerV2>::value, "Planner V2 ABI changed");
static_assert(TILEXR_MOONEP_PLANNER_STATUS_SUCCESS == 0,
    "Planner success status ABI changed");
static_assert(TILEXR_MOONEP_PLANNER_STATUS_TIMEOUT_BASE > 0,
    "Planner timeout status must encode a peer rank");
static_assert(std::is_same<decltype(&TileXRMoeEpPlanV2GetWorkspaceSize),
    OptimizedWorkspaceQueryV2>::value, "optimized Planner workspace ABI changed");
static_assert(std::is_same<decltype(&TileXRMoeEpPlanV2),
    OptimizedPlannerV2>::value, "optimized Planner V2 ABI changed");
static_assert(std::is_same<decltype(&TileXRMoeEpPlanV2WithMetadata),
    OptimizedPlannerV2WithMetadata>::value, "metadata Planner V2 ABI changed");
static_assert(std::is_same<decltype(&TileXREp::Plan::EncodeMoonEPGlobalTokenId),
    EncodeGlobalTokenId>::value, "global token encoder ABI changed");
static_assert(std::is_same<decltype(&TileXREp::Plan::DecodeMoonEPGlobalTokenId),
    DecodeGlobalTokenId>::value, "global token decoder ABI changed");
static_assert(std::is_same<decltype(&TileXREp::Plan::DecodeMoonEPDst),
    DecodeDst>::value, "dst decoder ABI changed");
static_assert(std::is_same<decltype(&TileXREp::Plan::BuildMoonEPRouteDescriptor),
    BuildRouteDescriptor>::value, "route descriptor ABI changed");

static_assert(std::is_standard_layout<TileXRMoonEpPlanV1>::value,
    "stage V1 plan must remain standard-layout");
static_assert(std::is_trivially_copyable<TileXRMoonEpPlanV1>::value,
    "stage V1 plan must remain trivially copyable");
static_assert(sizeof(TileXRMoonEpPlanV1) == 104, "stage V1 plan ABI changed");
static_assert(alignof(TileXRMoonEpPlanV1) == 8, "stage V1 plan alignment changed");

static_assert(std::is_standard_layout<TileXRMoonEPPlanConfig>::value,
    "Planner config must remain standard-layout");
static_assert(std::is_trivially_copyable<TileXRMoonEPPlanConfig>::value,
    "Planner config must remain trivially copyable");
static_assert(sizeof(TileXRMoonEPPlanConfig) == 56, "Planner config ABI changed");
static_assert(alignof(TileXRMoonEPPlanConfig) == 8, "Planner config alignment changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, prefetchSlots) == 0, "config prefetchSlots ABI changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, rankTokenCapacity) == 8, "config capacity ABI changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, nvS) == 16, "config nvS ABI changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, tokenPadding) == 24, "config padding ABI changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, tokenRouteLimitPerPair) == 32, "config route limit ABI changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, cardsPerServer) == 40, "config server topology ABI changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, cardsPerCabinet) == 44, "config cabinet topology ABI changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, crossCandidateCount) == 48, "config candidates ABI changed");
static_assert(offsetof(TileXRMoonEPPlanConfig, reserved) == 52, "config reserved ABI changed");

static_assert(std::is_standard_layout<TileXRMoonEPPlanDesc>::value,
    "Planner descriptor must remain standard-layout");
static_assert(std::is_trivially_copyable<TileXRMoonEPPlanDesc>::value,
    "Planner descriptor must remain trivially copyable");
static_assert(sizeof(TileXRMoonEPPlanDesc) == 136, "Planner descriptor ABI changed");
static_assert(alignof(TileXRMoonEPPlanDesc) == 8, "Planner descriptor alignment changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, dst) == 0, "descriptor dst ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, cuSeqlens) == 8, "descriptor cuSeqlens ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, expertsToCopy) == 16, "descriptor expertsToCopy ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, remoteStats) == 24, "descriptor remoteStats ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, dupGroups) == 32, "descriptor dupGroups ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, dupLoffs) == 40, "descriptor dupLoffs ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, dupCounts) == 48, "descriptor dupCounts ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, status) == 56, "descriptor status ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, s) == 64, "descriptor s ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, k) == 72, "descriptor k ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, r) == 80, "descriptor r ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, e) == 88, "descriptor e ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, b) == 96, "descriptor b ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, cap) == 104, "descriptor cap ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, nvS) == 112, "descriptor nvS ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, tokenPadding) == 120, "descriptor padding ABI changed");
static_assert(offsetof(TileXRMoonEPPlanDesc, epoch) == 128, "descriptor epoch ABI changed");

static_assert(std::is_standard_layout<TileXRMoonEPPlanMetadataV2>::value,
    "metadata V2 must remain standard-layout");
static_assert(std::is_trivially_copyable<TileXRMoonEPPlanMetadataV2>::value,
    "metadata V2 must remain trivially copyable");
static_assert(sizeof(TileXRMoonEPPlanMetadataV2) == 208, "metadata V2 size ABI changed");
static_assert(alignof(TileXRMoonEPPlanMetadataV2) == 8, "metadata V2 alignment changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, structSize) == 0, "metadata structSize ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, abiVersion) == 4, "metadata abiVersion ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, dst) == 8, "metadata dst ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, dstCount) == 16, "metadata dstCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, cuSeqlens) == 24, "metadata cuSeqlens ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, cuSeqlensCount) == 32, "metadata cuSeqlensCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, remoteExperts) == 40, "metadata remoteExperts ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, remoteExpertsCount) == 48, "metadata remoteExpertsCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, expertTargets) == 56, "metadata expertTargets ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, expertTargetsCount) == 64, "metadata expertTargetsCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, remoteStats) == 72, "metadata remoteStats ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, remoteStatsCount) == 80, "metadata remoteStatsCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, status) == 88, "metadata status ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, statusCount) == 96, "metadata statusCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, dupGroups) == 104, "metadata dupGroups ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, dupGroupsCount) == 112, "metadata dupGroupsCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, dupLoffs) == 120, "metadata dupLoffs ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, dupLoffsCount) == 128, "metadata dupLoffsCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, dupCounts) == 136, "metadata dupCounts ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, dupCountsCount) == 144, "metadata dupCountsCount ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, s) == 152, "metadata s ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, k) == 160, "metadata k ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, r) == 168, "metadata r ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, e) == 176, "metadata e ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, b) == 184, "metadata b ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, nvS) == 192, "metadata nvS ABI changed");
static_assert(offsetof(TileXRMoonEPPlanMetadataV2, epoch) == 200, "metadata epoch ABI changed");

static_assert(std::is_standard_layout<TileXREp::Plan::MoonEPRouteTarget>::value,
    "route target must remain standard-layout");
static_assert(std::is_trivially_copyable<TileXREp::Plan::MoonEPRouteTarget>::value,
    "route target must remain trivially copyable");
static_assert(sizeof(TileXREp::Plan::MoonEPRouteTarget) == 20, "route target ABI changed");
static_assert(alignof(TileXREp::Plan::MoonEPRouteTarget) == 4,
    "route target alignment changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteTarget, rawDst) == 0,
    "route target rawDst ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteTarget, dstRank) == 4,
    "route target dstRank ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteTarget, recvSlot) == 8,
    "route target recvSlot ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteTarget, sendHidden) == 12,
    "route target sendHidden ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteTarget, writeRouteWeight) == 16,
    "route target writeRouteWeight ABI changed");
static_assert(std::is_standard_layout<TileXREp::Plan::MoonEPRouteDescriptor>::value,
    "route descriptor must remain standard-layout");
static_assert(std::is_trivially_copyable<TileXREp::Plan::MoonEPRouteDescriptor>::value,
    "route descriptor must remain trivially copyable");
static_assert(sizeof(TileXREp::Plan::MoonEPRouteDescriptor) == 48, "route descriptor ABI changed");
static_assert(alignof(TileXREp::Plan::MoonEPRouteDescriptor) == 8,
    "route descriptor alignment changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, srcRank) == 0,
    "route descriptor srcRank ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, tokenId) == 4,
    "route descriptor tokenId ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, topKId) == 8,
    "route descriptor topKId ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, globalTokenId) == 16,
    "route descriptor globalTokenId ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, rawDst) == 24,
    "route descriptor rawDst ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, dstRank) == 28,
    "route descriptor dstRank ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, recvSlot) == 32,
    "route descriptor recvSlot ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, sendHidden) == 36,
    "route descriptor sendHidden ABI changed");
static_assert(offsetof(TileXREp::Plan::MoonEPRouteDescriptor, writeRouteWeight) == 40,
    "route descriptor writeRouteWeight ABI changed");

static_assert(PLAN_OK == 0, "PLAN_OK ABI changed");
static_assert(PLAN_PARTIAL_NO_FEASIBLE_PAIR == 1, "partial status ABI changed");
static_assert(PLAN_PARTIAL_PREFETCH_SLOT_EXHAUSTED == 2, "prefetch status ABI changed");
static_assert(PLAN_PARTIAL_TOKEN_SUPPLY == 3, "token supply status ABI changed");
static_assert(PLAN_ERROR_CONFIG_MISMATCH == 4, "config status ABI changed");
static_assert(PLAN_ERROR_TPE_MISMATCH == 5, "TPE status ABI changed");
static_assert(PLAN_ERROR_LAYOUT_EXCEEDS_NVS == 6, "layout status ABI changed");
static_assert(PLAN_ERROR_MOVE_RECORD_OVERFLOW == 7, "move status ABI changed");
static_assert(PLAN_ERROR_INTERNAL_INVARIANT == 8, "invariant status ABI changed");
static_assert(TILEXR_MOONEP_PLAN_METADATA_V2_ABI_VERSION == 2, "metadata ABI version changed");
static_assert(TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID == UINT64_MAX,
    "invalid global token sentinel changed");

int main()
{
    return 0;
}
