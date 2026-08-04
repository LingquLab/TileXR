#ifndef TILEXR_EP_HOST_EP_MEMORY_LAYOUT_H
#define TILEXR_EP_HOST_EP_MEMORY_LAYOUT_H

#include <cstdint>

#include "tilexr_types.h"

namespace TileXREp {

constexpr int64_t kEpMemoryStateWindowBytes = 1024 * 1024;
constexpr int64_t kEpMemoryWindowAlignmentBytes = 32;
constexpr int64_t kEpMemorySplitBlockBytes = 512;
constexpr int64_t kEpMemorySplitPayloadBytes = 480;
constexpr int64_t kEpMemoryFullUbBytes = 190 * 1024;
constexpr uint32_t kEpMemoryMaxCountCoreNum = 8;
constexpr uint32_t kEpMemoryMaxVectorCoreNum = 200;

struct EpMemoryDispatchReferenceConfig {
    int64_t localExpertNum = 0;
    int64_t rscvStatusNum = 0;
    uint32_t blockDim = 0;
    uint32_t aivUsedCumSum = 0;
    uint32_t aivUsedAllToAll = 0;
    uint32_t sharedUsedAivNum = 0;
    uint32_t moeUsedAivNum = 0;
    int64_t hCommuSize = 0;
    int64_t hOutSize = 0;
    int64_t scaleOutBytes = 0;
    int64_t tokenQuantAlignBytes = 0;
    int64_t expertPerSizeOnWin = 0;
    int64_t combineReserveBytes = 0;
    int64_t workspaceBytes = 0;
    int64_t totalWinSize = 0;
    int64_t dispatchHalfBytes = 0;
    int64_t maxSizeForUbBuffer = 0;
    int64_t totalUbSize = 0;
    int64_t allToAllUbBytes = 0;
};

struct EpMemoryCombineReferenceConfig {
    int64_t moeExpertNumPerRank = 0;
    uint32_t blockDim = 0;
    int64_t blockCntPerToken = 0;
    int64_t packedRowBytes = 0;
    int64_t combineReserveBytes = 0;
    int64_t workspaceBytes = 0;
    int64_t totalWinSize = 0;
    int64_t combineHalfBytes = 0;
    int64_t receiveUbBytes = 0;
    int64_t sendUbBytes = 0;
};

uint32_t TileXREpMemoryCountCoreNum(int64_t totalExpertNum, int64_t rscvStatusNum, uint32_t blockDim);

int TileXREpBuildMemoryDispatchReferenceConfig(int64_t rankSize, int64_t rank, int64_t bs, int64_t h,
    int64_t topK, int64_t moeExpertNum, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, TileXR::TileXRDataType dtype, TileXR::TileXRDataType expandXOutDtype,
    int64_t quantMode, uint32_t blockDim,
    EpMemoryDispatchReferenceConfig *out);

int TileXREpBuildMemoryCombineReferenceConfig(int64_t rankSize, int64_t rank, int64_t bs, int64_t h,
    int64_t topK, int64_t moeExpertNum, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, TileXR::TileXRDataType dtype, int64_t quantMode, uint32_t blockDim,
    EpMemoryCombineReferenceConfig *out);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_MEMORY_LAYOUT_H
