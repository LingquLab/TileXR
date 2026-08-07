#ifndef TILEXR_MOONEP_PEER_WINDOW_H
#define TILEXR_MOONEP_PEER_WINDOW_H

#include <cstdint>

namespace TileXRMoonEp {

constexpr int64_t kMoonEpStageAivBlockCount = 1;
constexpr uint64_t kMoonEpStageAlignment = 32;
constexpr uint32_t kMoonEpSyncUbBytes = 4 * 1024;
constexpr uint32_t kMoonEpWorkUbBytes = 192 * 1024;
constexpr uint64_t kMoonEpPeerWaitIterations = 100000000ULL;
constexpr uint64_t kMoonEpFlagBuildDedup = UINT64_C(1) << 0;
constexpr uint64_t kMoonEpFlagSkipInterRankSync = UINT64_C(1) << 1;

constexpr int64_t kMoonEpModeBfloat16Hidden = 1;
constexpr int64_t kMoonEpModeFloat32Weights = 2;

constexpr int32_t kMoonEpDispatchWindowClearedStep = 101;
constexpr int32_t kMoonEpDispatchDataReadyStep = 102;
constexpr int32_t kMoonEpDispatchWindowDrainedStep = 103;
constexpr int32_t kMoonEpDispatchFailedStep = 104;
constexpr int32_t kMoonEpCombineDataReadyStep = 111;
constexpr int32_t kMoonEpCombineWindowDrainedStep = 112;
constexpr int32_t kMoonEpCombineFailedStep = 113;
constexpr int32_t kMoonEpPrefetchWeightReadyStep = 401;
constexpr int32_t kMoonEpPrefetchWeightDrainedStep = 402;
constexpr int32_t kMoonEpPrefetchWeightFailedStep = 403;

constexpr int32_t kMoonEpDispatchStatusSuccess = 2000;
constexpr int32_t kMoonEpDispatchStatusInvalidRoute = 2001;
constexpr int32_t kMoonEpDispatchStatusRemoteFailureBase = 2100;
constexpr int32_t kMoonEpDispatchStatusTimeoutBase = 2200;
constexpr int32_t kMoonEpCombineStatusSuccess = 3000;
constexpr int32_t kMoonEpCombineStatusInvalidRoute = 3001;
constexpr int32_t kMoonEpCombineStatusRemoteFailureBase = 3100;
constexpr int32_t kMoonEpCombineStatusTimeoutBase = 3200;
constexpr int32_t kMoonEpPrefetchWeightStatusSuccess = 4000;
constexpr int32_t kMoonEpPrefetchWeightStatusInvalidPlan = 4001;
constexpr int32_t kMoonEpPrefetchWeightStatusRemoteFailureBase = 4100;
constexpr int32_t kMoonEpPrefetchWeightStatusTimeoutBase = 4200;
} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PEER_WINDOW_H
