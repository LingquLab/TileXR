#ifndef TILEXR_EP_COMMON_EP_WINDOW_H
#define TILEXR_EP_COMMON_EP_WINDOW_H

#include <cstdint>

namespace TileXREp {

constexpr int64_t kEpWindowAlignmentBytes = 32;
constexpr int64_t kEpAssistTupleInts = 4;
constexpr int64_t kEpWindowHeaderBytes = 64;
constexpr int64_t kEpSrcSlotHeaderBytes = 64;
constexpr int32_t kEpStepWindowCleared = 71;
constexpr int32_t kEpStepDispatchReady = 72;
constexpr int32_t kEpStepCombineWindowCleared = 73;
constexpr int32_t kEpStepCombineReady = 74;
constexpr int32_t kEpStepDispatchDrained = 75;
constexpr int32_t kEpStepCombineGatewayReady = 76;
constexpr int32_t kEpStepCombineRelayReady = 77;
constexpr int32_t kEpStepCombineDataAsFlagReady = 78;
constexpr int64_t kEpStatusOk = 0;
constexpr int64_t kEpStatusRemoteReadyTimeout = 1;
constexpr uint32_t kEpWindowMagic = 0x54584550U;

struct EpWindowHeader {
    uint32_t magic;
    int32_t rankSize;
    int64_t maxRoutesPerSrc;
    int64_t rowBytes;
    int64_t slotBytes;
    int64_t totalBytes;
    int64_t reserved0;
    int64_t reserved1;
    int64_t reserved2;
};

struct EpSrcSlotHeader {
    int32_t count;
    int32_t srcRank;
    int64_t payloadBytes;
    int64_t assistBytes;
    int64_t reserved0;
    int64_t reserved1;
    int64_t reserved2;
    int64_t reserved3;
    int64_t reserved4;
};

struct EpAssistTuple {
    int32_t srcRank;
    int32_t tokenId;
    int32_t topKId;
    int32_t expertId;
};

static_assert(sizeof(EpWindowHeader) == kEpWindowHeaderBytes, "EpWindowHeader must be 64 bytes");
static_assert(sizeof(EpSrcSlotHeader) == kEpSrcSlotHeaderBytes, "EpSrcSlotHeader must be 64 bytes");
static_assert(sizeof(EpAssistTuple) == kEpAssistTupleInts * static_cast<int64_t>(sizeof(int32_t)),
    "EpAssistTuple must contain 4 int32 fields");

} // namespace TileXREp

#endif // TILEXR_EP_COMMON_EP_WINDOW_H
