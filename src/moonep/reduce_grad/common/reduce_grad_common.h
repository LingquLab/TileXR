#ifndef TILEXR_MOONEP_REDUCE_GRAD_COMMON_H
#define TILEXR_MOONEP_REDUCE_GRAD_COMMON_H

#include <cstdint>

#include "tilexr_udma_types.h"

namespace TileXRMoonEp {

constexpr uint64_t kReduceGradUdmaThresholdBytes = UINT64_C(1) << 20;
constexpr uint64_t kReduceGradDefaultUdmaChunkBytes = UINT64_C(4) << 20;
constexpr uint64_t kReduceGradStateWindowBytes = UINT64_C(1) << 20;
constexpr uint64_t kReduceGradDataAsFlagRecordBytes = 512;
constexpr uint64_t kReduceGradDataAsFlagPayloadBytes = 480;
constexpr uint64_t kReduceGradUdmaAlignment = 512;
// Keep the caller-owned MR compatible with older Ascend950 registration granularity.
constexpr uint64_t kReduceGradUdmaWorkspaceAlignment = UINT64_C(2) << 20;
constexpr uint64_t kReduceGradUdmaStageCount = 2;
constexpr uint64_t kReduceGradUdmaSignalStageStride =
    TileXR::TILEXR_UDMA_CACHE_LINE_SIZE;
constexpr uint64_t kReduceGradUdmaPeerStateBytes =
    3 * kReduceGradUdmaStageCount * kReduceGradUdmaSignalStageStride;
constexpr uint64_t kReduceGradUdmaReadyOffset = 0;
constexpr uint64_t kReduceGradUdmaCompletionOffset =
    kReduceGradUdmaStageCount * kReduceGradUdmaSignalStageStride;
constexpr uint64_t kReduceGradUdmaPollScratchOffset =
    2 * kReduceGradUdmaStageCount * kReduceGradUdmaSignalStageStride;
static_assert(kReduceGradUdmaPollScratchOffset +
    kReduceGradUdmaStageCount * kReduceGradUdmaSignalStageStride <=
    kReduceGradUdmaPeerStateBytes, "ReduceGrad UDMA peer state exceeds its stride");
constexpr uint64_t kReduceGradKernelTileBytes = UINT64_C(30) * 1024;
constexpr uint32_t kReduceGradProjectionCount = 3;
constexpr uint32_t kReduceGradMaxUdmaQpCount = 8;
constexpr int64_t kReduceGradMaxAivBlockCount = 64;
constexpr uint32_t kReduceGradTransportPeer = 1;
constexpr uint32_t kReduceGradTransportUdma = 2;
constexpr int32_t kReduceGradDeviceInvalidState = 1;
constexpr int32_t kReduceGradDevicePeerTimeout = 2;
constexpr int32_t kReduceGradDeviceUdmaCqError = 3;

enum ReduceGradProjection : uint32_t {
    kReduceGradGate = 0,
    kReduceGradUp = 1,
    kReduceGradDown = 2,
};

struct ReduceGradLayout {
    int64_t rank = 0;
    int64_t rankSize = 0;
    int64_t expertCount = 0;
    int64_t expertsPerRank = 0;
    int64_t prefetchSlots = 0;
    int64_t blockDim = 0;
    int64_t controlBlockCount = 0;

    uint64_t rowElements[kReduceGradProjectionCount] = {};
    uint64_t rowBytes[kReduceGradProjectionCount] = {};
    uint32_t transports[kReduceGradProjectionCount] = {};
    uint32_t udmaQpCount = 0;

    uint64_t peerWindowBytes = 0;
    uint64_t peerRecordBaseOffset = 0;
    uint64_t peerHalfBytes = 0;
    uint64_t peerSlotStrideBytes = 0;
    uint64_t peerChunkPayloadBytes = 0;
    uint64_t peerChunkCounts[kReduceGradProjectionCount] = {};

    uint64_t udmaStateOffset = 0;
    uint64_t udmaOutboundOffset = 0;
    uint64_t udmaInboundOffset = 0;
    uint64_t udmaChunkBytes = 0;
    uint64_t udmaChunkCounts[kReduceGradProjectionCount] = {};
    uint64_t workspaceBytes = 0;
};

struct ReduceGradKernelArgs {
    uint8_t *commArgs;
    uint8_t *expertsToCopy;
    uint8_t *gate;
    uint8_t *up;
    uint8_t *down;
    uint8_t *workspace;
    uint8_t *status;
    int64_t rank;
    int64_t rankSize;
    int64_t expertCount;
    int64_t expertsPerRank;
    int64_t prefetchSlots;
    int64_t controlBlockCount;
    uint64_t gateRowElements;
    uint64_t upRowElements;
    uint64_t downRowElements;
    uint64_t gateRowBytes;
    uint64_t upRowBytes;
    uint64_t downRowBytes;
    uint32_t gateTransport;
    uint32_t upTransport;
    uint32_t downTransport;
    uint32_t udmaQpCount;
    uint64_t peerRecordBaseOffset;
    uint64_t peerHalfBytes;
    uint64_t peerSlotStrideBytes;
    uint64_t peerChunkPayloadBytes;
    uint64_t udmaStateOffset;
    uint64_t udmaOutboundOffset;
    uint64_t udmaInboundOffset;
    uint64_t udmaChunkBytes;
    uint64_t workspaceBytes;
    uint64_t waitIterations;
    int64_t magic;
};

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_REDUCE_GRAD_COMMON_H
