#ifndef TILEXR_EP_COMMON_EP_URMA_COMBINE_H
#define TILEXR_EP_COMMON_EP_URMA_COMBINE_H

#include <cstddef>
#include <cstdint>

#include "ep_urma_combine_config.h"

namespace TileXREp {

constexpr int64_t kEpUrmaCombineWorkspaceAlignment = 2 * 1024 * 1024;
constexpr int64_t kEpUrmaCombineHeaderBytes = 512;
constexpr int64_t kEpUrmaCombineCacheLineBytes = 64;
constexpr int64_t kEpUrmaCombineDataBlockBytes = 512;
constexpr int64_t kEpUrmaCombinePayloadBytes = 480;
constexpr int64_t kEpUrmaCombineFlagBytes = 32;
constexpr int64_t kEpUrmaCombineQuantHeaderBytes = 32;
constexpr int64_t kEpUrmaCombineAivCount = kEpUrmaCombineProfileCoreCount;
constexpr int64_t kEpUrmaCombinePackLaneCount = kEpUrmaCombineProfilePackReceiveCoreCount;
constexpr int64_t kEpUrmaCombineSendLaneCount = kEpUrmaCombineProfileSendCoreCount;
constexpr int64_t kEpUrmaCombineRequiredQpCount = kEpUrmaCombineSendLaneCount;
constexpr int64_t kEpUrmaCombineMaxHidden = 8192;
constexpr int64_t kEpUrmaCombineMaxTopK = 16;
constexpr int64_t kEpUrmaCombineMaxBlocksPerRoute = 255;
constexpr int64_t kEpUrmaCombineQuantModeInt8PerRoute = 1;

constexpr uint32_t kEpUrmaCombineTxRouteReady = 1;
constexpr uint32_t kEpUrmaCombineRxLaneDone = 2;
constexpr uint32_t kEpUrmaCombineSenderDone = 3;
constexpr uint32_t kEpUrmaCombineRxBufferReleased = 4;
constexpr uint32_t kEpUrmaCombinePublishDone = 5;
constexpr uint32_t kEpUrmaCombineStartLocalReady = 6;
constexpr uint32_t kEpUrmaCombineStartRankReady = 7;
constexpr uint32_t kEpUrmaCombineStartPublishDone = 8;
constexpr uint32_t kEpUrmaCombineStartRun = 9;
constexpr uint32_t kEpUrmaCombineCreditExpectedReady = 10;
constexpr uint32_t kEpUrmaCombineCreditShardDone = 11;
constexpr uint32_t kEpUrmaCombineCreditRun = 12;
constexpr uint32_t kEpUrmaCombineRxReleaseShardDone = 13;

constexpr uint64_t kEpUrmaCombineStatusOk = 0;
constexpr uint64_t kEpUrmaCombineStatusInvalidRoute = 1;
constexpr uint64_t kEpUrmaCombineStatusInvalidQuantHeader = 2;

struct alignas(kEpUrmaCombineQuantHeaderBytes) EpUrmaCombineQuantHeader {
    float scale;
    int32_t quantMode;
    int64_t reserved0;
    int64_t reserved1;
    int64_t reserved2;
};

constexpr int64_t kEpUrmaCombineTxReadyHeaderOffset =
    offsetof(EpUrmaCombineQuantHeader, reserved2);

struct alignas(kEpUrmaCombineCacheLineBytes) EpUrmaCombineControlLine {
    uint64_t value;
    uint64_t reserved[7];
};

static_assert(sizeof(EpUrmaCombineQuantHeader) == kEpUrmaCombineQuantHeaderBytes,
    "URMA combine quant header must be 32 bytes");
static_assert(kEpUrmaCombineTxReadyHeaderOffset == 24,
    "URMA combine in-data TX-ready word must remain at byte 24 of the quant header");
static_assert(sizeof(EpUrmaCombineControlLine) == kEpUrmaCombineCacheLineBytes,
    "URMA combine control line must be one cache line");
static_assert(kEpUrmaCombinePackLaneCount + kEpUrmaCombineSendLaneCount == kEpUrmaCombineAivCount,
    "URMA combine must occupy all configured AIV cores");

} // namespace TileXREp

#endif // TILEXR_EP_COMMON_EP_URMA_COMBINE_H
