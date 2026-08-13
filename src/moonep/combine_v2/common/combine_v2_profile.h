#ifndef TILEXR_MOONEP_COMBINE_V2_PROFILE_H
#define TILEXR_MOONEP_COMBINE_V2_PROFILE_H

#include <cstdint>

namespace TileXRMoonEp {

constexpr uint32_t kMoonEpCombineV2ProfileMarker = 0x54584750U; // TXGP
constexpr uint16_t kMoonEpCombineV2ProfileVersion = 3U;
constexpr uint32_t kMoonEpCombineV2ProfileCyclesPerUs = 1000U;
constexpr uint32_t kMoonEpCombineV2ProfileTimePointCapacity = 32U;
constexpr uint32_t kMoonEpCombineV2ProfileStepCount = 8U;
constexpr uint32_t kMoonEpCombineV2ProfileStepPointStride = 2U;

enum MoonEpCombineV2ProfileTimePointIndex : uint32_t {
    MOONEP_COMBINE_V2_TIME_INIT_BEGIN = 0U,
    MOONEP_COMBINE_V2_TIME_INIT_END = 1U,
    MOONEP_COMBINE_V2_TIME_PREPARE_END = 2U,
    MOONEP_COMBINE_V2_TIME_STEP0_SEND_END = 3U,
    MOONEP_COMBINE_V2_TIME_STEP0_READY_END = 4U,
    MOONEP_COMBINE_V2_TIME_STEP_LOOP_END =
        MOONEP_COMBINE_V2_TIME_STEP0_SEND_END +
        kMoonEpCombineV2ProfileStepCount *
            kMoonEpCombineV2ProfileStepPointStride,
    MOONEP_COMBINE_V2_TIME_INBOUND_DONE_END =
        MOONEP_COMBINE_V2_TIME_STEP_LOOP_END + 1U,
    MOONEP_COMBINE_V2_TIME_FINAL_END =
        MOONEP_COMBINE_V2_TIME_INBOUND_DONE_END + 1U,
};

constexpr uint32_t kMoonEpCombineV2ProfileTimePointCount =
    MOONEP_COMBINE_V2_TIME_FINAL_END + 1U;

enum MoonEpCombineV2ProfileMetricIndex : uint32_t {
    MOONEP_COMBINE_V2_METRIC_SELECTION_LOAD = 0U,
    MOONEP_COMBINE_V2_METRIC_SELECTION_SELECT,
    MOONEP_COMBINE_V2_METRIC_SELF_ROUTE_DECODE,
    MOONEP_COMBINE_V2_METRIC_SELF_COPY,
    MOONEP_COMBINE_V2_METRIC_REMOTE_ROUTE_DECODE,
    MOONEP_COMBINE_V2_METRIC_REMOTE_DESCRIPTOR,
    MOONEP_COMBINE_V2_METRIC_REMOTE_WQE_BUILD,
    MOONEP_COMBINE_V2_METRIC_REMOTE_SUBMIT,
};

constexpr uint32_t kMoonEpCombineV2ProfileMetricCount = 8U;

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
    uint32_t metricCount;
    uint32_t reserved;
    int64_t timePoint[kMoonEpCombineV2ProfileTimePointCapacity];
    uint64_t metric[kMoonEpCombineV2ProfileMetricCount];
};

static_assert(kMoonEpCombineV2ProfileTimePointCount == 22U,
    "Combine V2 profile must contain 22 cumulative timestamps");
static_assert(MOONEP_COMBINE_V2_DIAG_OBSERVED <
        kMoonEpCombineV2ProfileTimePointCapacity,
    "Combine V2 diagnostics exceed the profile record");
static_assert(MOONEP_COMBINE_V2_METRIC_REMOTE_SUBMIT + 1U ==
        kMoonEpCombineV2ProfileMetricCount,
    "Combine V2 profile metric count mismatch");
static_assert(sizeof(MoonEpCombineV2ProfileRecord) == 384U,
    "Combine V2 profile record ABI changed");

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_COMBINE_V2_PROFILE_H
