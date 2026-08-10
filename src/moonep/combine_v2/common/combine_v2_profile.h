#ifndef TILEXR_MOONEP_COMBINE_V2_PROFILE_H
#define TILEXR_MOONEP_COMBINE_V2_PROFILE_H

#include <cstdint>

namespace TileXRMoonEp {

constexpr uint32_t kMoonEpCombineV2ProfileMarker = 0x54584750U; // TXGP
constexpr uint16_t kMoonEpCombineV2ProfileVersion = 1U;
constexpr uint32_t kMoonEpCombineV2ProfileCyclesPerUs = 1000U;
constexpr uint32_t kMoonEpCombineV2ProfileTimePointCapacity = 32U;
constexpr uint32_t kMoonEpCombineV2ProfileStepCount = 8U;
constexpr uint32_t kMoonEpCombineV2ProfileStepPointStride = 2U;

enum MoonEpCombineV2ProfileTimePointIndex : uint32_t {
    MOONEP_COMBINE_V2_TIME_INIT_BEGIN = 0U,
    MOONEP_COMBINE_V2_TIME_INIT_END = 1U,
    MOONEP_COMBINE_V2_TIME_PREPARE_END = 2U,
    MOONEP_COMBINE_V2_TIME_STEP0_GRANT_END = 3U,
    MOONEP_COMBINE_V2_TIME_STEP0_SEND_END = 4U,
    MOONEP_COMBINE_V2_TIME_FINAL_CQ_END =
        MOONEP_COMBINE_V2_TIME_STEP0_GRANT_END +
        kMoonEpCombineV2ProfileStepCount *
            kMoonEpCombineV2ProfileStepPointStride,
    MOONEP_COMBINE_V2_TIME_INBOUND_DONE_END =
        MOONEP_COMBINE_V2_TIME_FINAL_CQ_END + 1U,
    MOONEP_COMBINE_V2_TIME_FINAL_END =
        MOONEP_COMBINE_V2_TIME_INBOUND_DONE_END + 1U,
};

constexpr uint32_t kMoonEpCombineV2ProfileTimePointCount =
    MOONEP_COMBINE_V2_TIME_FINAL_END + 1U;

enum MoonEpCombineV2ProfileDiagnosticIndex : uint32_t {
    MOONEP_COMBINE_V2_DIAG_FAILURE_STATUS =
        kMoonEpCombineV2ProfileTimePointCount,
    MOONEP_COMBINE_V2_DIAG_FAILURE_STEP,
    MOONEP_COMBINE_V2_DIAG_FAILURE_PEER,
    MOONEP_COMBINE_V2_DIAG_FAILURE_LANE,
    MOONEP_COMBINE_V2_DIAG_FAILURE_QP,
    MOONEP_COMBINE_V2_DIAG_CQ_STATUS,
    MOONEP_COMBINE_V2_DIAG_EXPECTED,
    MOONEP_COMBINE_V2_DIAG_OBSERVED,
};

struct alignas(64) MoonEpCombineV2ProfileRecord {
    uint32_t marker;
    uint16_t version;
    uint16_t recordBytes;
    uint32_t rank;
    uint32_t core;
    uint32_t blockDim;
    uint32_t timePointCount;
    int64_t timePoint[kMoonEpCombineV2ProfileTimePointCapacity];
};

static_assert(kMoonEpCombineV2ProfileTimePointCount == 22U,
    "Combine V2 profile must contain 22 cumulative timestamps");
static_assert(MOONEP_COMBINE_V2_DIAG_OBSERVED <
        kMoonEpCombineV2ProfileTimePointCapacity,
    "Combine V2 diagnostics exceed the profile record");
static_assert(sizeof(MoonEpCombineV2ProfileRecord) == 320U,
    "Combine V2 profile record ABI changed");

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_V2_PROFILE_H
