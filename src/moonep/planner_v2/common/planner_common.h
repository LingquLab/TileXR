#ifndef TILEXR_MOONEP_PLANNER_COMMON_H
#define TILEXR_MOONEP_PLANNER_COMMON_H

#include <cstdint>

namespace TileXRMoonEp {

constexpr int64_t kPlannerAivBlockCount = 64;
constexpr uint64_t kPlannerWorkspaceAlignment = 32;
constexpr int32_t kPlannerReadyStep = 91;
constexpr int32_t kPlannerReadyEventId = 4096;
constexpr int32_t kPlannerStatusSuccess = 0;
constexpr int32_t kPlannerStatusTimeoutBase = 1000;
constexpr int64_t kPlannerMaxTopK = 32;
constexpr int64_t kPlannerMaxExpertCount = 4096;
constexpr uint32_t kPlannerSyncUbBytes = 4 * 1024;
constexpr uint32_t kPlannerWorkUbBytes = 192 * 1024;
constexpr int64_t kPlannerRouteTileInts = 16 * 1024;

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PLANNER_COMMON_H
