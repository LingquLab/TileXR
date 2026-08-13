#ifndef TILEXR_MOONEP_DISPATCH_PROFILE_H
#define TILEXR_MOONEP_DISPATCH_PROFILE_H

#include <cstdint>

namespace TileXRMoonEp {

constexpr uint32_t kDispatchProfileMarker = 0x54584450U; // TXDP
constexpr uint32_t kDispatchDfxMarker = 0x54584444U; // TXDD
constexpr uint32_t kDispatchKernelStatusMarker = 0x54584453U; // TXDS
constexpr uint16_t kDispatchDiagnosticVersion = 3U;
constexpr uint64_t kDispatchKernelStatusFeatureDfxEnabled = 1U << 0;
constexpr uint64_t kDispatchKernelStatusFeatureProfilingEnabled = 1U << 1;
constexpr uint64_t kDispatchKernelStatusFeatureFusedEpoch = 1U << 2;

enum DispatchSelectMode : uint32_t {
    kDispatchSelectScalarTiled = 0,
    kDispatchSelectVector = 1,
};

enum DispatchVectorFallbackReason : uint32_t {
    kDispatchVectorFallbackNone = 0,
    kDispatchVectorFallbackNotPowerOfTwo = 1,
    kDispatchVectorFallbackIndexRange = 2,
    kDispatchVectorFallbackUbBudget = 3,
    kDispatchVectorFallbackApiGranularity = 4,
};

enum DispatchDfxFlag : uint32_t {
    kDispatchDfxInvalidConfig = 1U << 0,
    kDispatchDfxInvalidRoute = 1U << 1,
    kDispatchDfxRouteCountMismatch = 1U << 2,
    kDispatchDfxQuietError = 1U << 3,
    kDispatchDfxCompletionTimeout = 1U << 4,
    kDispatchDfxUpstreamPlannerError = 1U << 5,
    kDispatchDfxCreditTimeout = 1U << 6,
    kDispatchDfxCqError = 1U << 7,
};

enum DispatchTimelineIndex : uint32_t {
    kDispatchTimelineStagingEnd = 0,
    kDispatchTimelineIssueStart = 1,
    kDispatchTimelineRemoteIssueEnd = 2,
    kDispatchTimelineIssueEnd = 3,
    kDispatchTimelineFlagWaitStart = 4,
    kDispatchTimelineFlagWaitEnd = 5,
    kDispatchTimelineDfxWriteEnd = 6,
    kDispatchTimelineSyncAllEnd = 7,
    kDispatchTimelineOutputStart = 8,
    kDispatchTimelineOutputEnd = 9,
    kDispatchTimelineQuietEnd = 10,
    kDispatchTimelinePointCount = 11,
};

struct alignas(64) DispatchProfileRecord {
    uint32_t marker;
    uint16_t version;
    uint16_t recordBytes;
    uint32_t payloadMode;
    uint32_t rank;
    uint32_t core;
    uint32_t blockDim;
    uint32_t flags;
    uint32_t selectMode;
    uint32_t vectorFallbackReason;
    uint32_t scratchIndex;
    uint32_t groupCount;
    uint64_t magic;
    uint64_t scannedRouteCount;
    uint64_t matchedRouteCount;
    uint64_t selectedRouteCount;
    uint64_t processedRouteCount;
    uint64_t issuedPutCount;
    uint64_t issuedPutBytes;
    uint64_t visitedPeerCount;
    uint64_t completionFlagCount;
    uint64_t kernelCycles;
    uint64_t stagingCycles;
    uint64_t putIssueCycles;
    uint64_t flagWaitCycles;
    uint64_t outputCopyCycles;
    uint64_t quietCycles;
    uint64_t reserved[11];
};

static_assert(kDispatchTimelinePointCount <= 11U,
    "Dispatch timeline no longer fits the profile record");
static_assert(sizeof(DispatchProfileRecord) == 256U,
    "Dispatch profile record ABI changed");

struct alignas(64) DispatchDfxRecord {
    uint32_t marker;
    uint16_t version;
    uint16_t recordBytes;
    uint32_t payloadMode;
    uint32_t rank;
    uint32_t core;
    uint32_t flags;
    uint32_t firstInvalidRouteId;
    int32_t firstInvalidRawDst;
    uint32_t firstQuietStatus;
    uint32_t firstQuietPhase;
    uint32_t timeoutPeer;
    uint32_t timeoutPhase;
    uint32_t reserved0;
    uint64_t expectedRouteCount;
    uint64_t processedRouteCount;
    uint64_t magic;
    uint64_t timeoutExpectedMagic;
    uint64_t timeoutObservedFlag;
    uint64_t reserved[4];
};

static_assert(sizeof(DispatchDfxRecord) == 128U,
    "Dispatch DFX record ABI changed");

struct alignas(64) DispatchKernelStatus {
    uint32_t marker;
    uint16_t version;
    uint16_t recordBytes;
    int32_t status;
    uint32_t payloadMode;
    uint64_t magic;
    uint64_t reserved[5];
};

static_assert(sizeof(DispatchKernelStatus) == 64U,
    "Dispatch kernel status ABI changed");

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_DISPATCH_PROFILE_H
