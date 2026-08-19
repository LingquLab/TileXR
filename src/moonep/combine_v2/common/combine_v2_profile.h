#ifndef TILEXR_MOONEP_COMBINE_V2_PROFILE_H
#define TILEXR_MOONEP_COMBINE_V2_PROFILE_H

#include <cstdint>

namespace TileXRMoonEp {

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define TILEXR_MOONEP_COMBINE_V2_PROFILE_INLINE \
    __attribute__((always_inline)) inline __aicore__
#else
#define TILEXR_MOONEP_COMBINE_V2_PROFILE_INLINE inline constexpr
#endif

constexpr uint32_t kMoonEpCombineV2ProfileMarker = 0x54584750U; // TXGP
constexpr uint16_t kMoonEpCombineV2ProfileVersion = 6U;
constexpr uint32_t kMoonEpCombineV2ProfileCyclesPerUs = 1000U;
constexpr uint32_t kMoonEpCombineV2ProfileTimePointCapacity = 40U;
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
    MOONEP_COMBINE_V2_TIME_COLLECTIVE_DONE_END =
        MOONEP_COMBINE_V2_TIME_STEP_LOOP_END + 1U,
    MOONEP_COMBINE_V2_TIME_FINAL_END =
        MOONEP_COMBINE_V2_TIME_COLLECTIVE_DONE_END + 1U,
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
    MOONEP_COMBINE_V2_METRIC_CREDIT_WAIT,
    MOONEP_COMBINE_V2_METRIC_CREDIT_PUBLISH,
};

constexpr uint32_t kMoonEpCombineV2ProfileMetricCount = 10U;

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
    MOONEP_COMBINE_V2_DIAG_FULLMESH_WQE_BUILD_END,
    MOONEP_COMBINE_V2_DIAG_FULLMESH_SUBMIT_END,
    MOONEP_COMBINE_V2_DIAG_FULLMESH_CQ_SUCCESS,
};

enum MoonEpCombineV2ProfileTransport : uint32_t {
    MOONEP_COMBINE_V2_PROFILE_TRANSPORT_NONE = 0U,
    MOONEP_COMBINE_V2_PROFILE_TRANSPORT_CLOS = 1U,
    MOONEP_COMBINE_V2_PROFILE_TRANSPORT_FULLMESH = 2U,
};

constexpr uint32_t kMoonEpCombineV2ProfileRouteValid = 1U << 31U;
constexpr uint32_t kMoonEpCombineV2ProfileTransportShift = 0U;
constexpr uint32_t kMoonEpCombineV2ProfileTransportMask = 0x3U;
constexpr uint32_t kMoonEpCombineV2ProfileStepShift = 2U;
constexpr uint32_t kMoonEpCombineV2ProfileStepMask = 0xFU;
constexpr uint32_t kMoonEpCombineV2ProfilePeerShift = 6U;
constexpr uint32_t kMoonEpCombineV2ProfilePeerMask = 0xFFU;
constexpr uint32_t kMoonEpCombineV2ProfileSuccessorShift = 14U;
constexpr uint32_t kMoonEpCombineV2ProfileSuccessorMask = 0xFFU;
constexpr uint32_t kMoonEpCombineV2ProfileQpShift = 22U;
constexpr uint32_t kMoonEpCombineV2ProfileQpMask = 0x3FU;

TILEXR_MOONEP_COMBINE_V2_PROFILE_INLINE uint32_t
MoonEpCombineV2PackFullmeshProfileRoute(
    uint32_t step, uint32_t peer, uint32_t successor, uint32_t qp)
{
    return kMoonEpCombineV2ProfileRouteValid |
        (MOONEP_COMBINE_V2_PROFILE_TRANSPORT_FULLMESH <<
            kMoonEpCombineV2ProfileTransportShift) |
        ((step & kMoonEpCombineV2ProfileStepMask) <<
            kMoonEpCombineV2ProfileStepShift) |
        ((peer & kMoonEpCombineV2ProfilePeerMask) <<
            kMoonEpCombineV2ProfilePeerShift) |
        ((successor & kMoonEpCombineV2ProfileSuccessorMask) <<
            kMoonEpCombineV2ProfileSuccessorShift) |
        ((qp & kMoonEpCombineV2ProfileQpMask) <<
            kMoonEpCombineV2ProfileQpShift);
}

TILEXR_MOONEP_COMBINE_V2_PROFILE_INLINE uint32_t
MoonEpCombineV2ProfileRouteField(
    uint32_t route, uint32_t shift, uint32_t mask)
{
    return (route >> shift) & mask;
}

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
static_assert(MOONEP_COMBINE_V2_DIAG_FULLMESH_CQ_SUCCESS <
        kMoonEpCombineV2ProfileTimePointCapacity,
    "Combine V2 Fullmesh diagnostics exceed the profile record");
static_assert(MOONEP_COMBINE_V2_METRIC_CREDIT_PUBLISH + 1U ==
        kMoonEpCombineV2ProfileMetricCount,
    "Combine V2 profile metric count mismatch");
static_assert(sizeof(MoonEpCombineV2ProfileRecord) == 448U,
    "Combine V2 profile record ABI changed");

} // namespace TileXRMoonEp

#undef TILEXR_MOONEP_COMBINE_V2_PROFILE_INLINE

#endif // TILEXR_MOONEP_COMBINE_V2_PROFILE_H
