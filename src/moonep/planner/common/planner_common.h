#ifndef TILEXR_MOONEP_PLANNER_COMMON_H
#define TILEXR_MOONEP_PLANNER_COMMON_H

#include <cstdint>

namespace TileXRMoonEpV3 {

constexpr int64_t kPlannerAivBlockCount = 64;
constexpr uint64_t kPlannerWorkspaceAlignment = 32;
constexpr int32_t kPlannerReadyStep = 91;
constexpr int32_t kPlannerDstReadyStep = 92;
constexpr int32_t kPlannerReadyEventId = 4096;
constexpr int32_t kPlannerStatusSuccess = 0;
constexpr int32_t kPlannerStatusTimeoutBase = 1000;
constexpr uint32_t kPlannerStatusObservedFlagMarker = UINT32_C(0x80000000);
constexpr int64_t kPlannerMaxTopK = 32;
constexpr int64_t kPlannerMaxExpertCount = 4096;
constexpr uint32_t kPlannerSyncUbBytes = 4 * 1024;
constexpr uint32_t kPlannerWorkUbBytes = 192 * 1024;
constexpr int64_t kPlannerRouteTileInts = 16 * 1024;

#if defined(__CCE_AICORE__)
__aicore__
#endif
inline int32_t EncodePlannerObservedFlagStatus(
    int32_t peer, int32_t expectedStep, int64_t observed)
{
    const uint32_t observedMagic = static_cast<uint32_t>(
        static_cast<uint64_t>(observed) >> 32);
    const uint32_t observedStep = static_cast<uint32_t>(observed);
    const uint32_t expectedDstPhase = expectedStep == kPlannerDstReadyStep ? 1U : 0U;
    return static_cast<int32_t>(kPlannerStatusObservedFlagMarker |
        ((static_cast<uint32_t>(peer) & UINT32_C(0x7)) << 28) |
        (expectedDstPhase << 27) |
        ((observedMagic & UINT32_C(0x7FF)) << 16) |
        (observedStep & UINT32_C(0xFFFF)));
}

} // namespace TileXRMoonEpV3

#endif // TILEXR_MOONEP_PLANNER_COMMON_H
