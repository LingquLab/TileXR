#include "ep_layout.h"

#include <limits>

#include "comm_args.h"
#include "ep_window.h"

namespace TileXREp {
namespace {

bool MulInt64(int64_t lhs, int64_t rhs, int64_t *out)
{
    if (out == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool AddInt64(int64_t lhs, int64_t rhs, int64_t *out)
{
    if (out == nullptr || lhs < 0 || rhs < 0) {
        return false;
    }
    if (rhs > std::numeric_limits<int64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool IsPositive(int64_t value)
{
    return value > 0;
}

} // namespace

int64_t TileXREpAlignUp(int64_t value, int64_t alignment)
{
    if (value < 0 || alignment <= 0) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    const int64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    int64_t aligned = 0;
    if (!AddInt64(value, alignment - remainder, &aligned)) {
        return TileXR::TILEXR_INVALID_VALUE;
    }
    return aligned;
}

int64_t TileXREpDataTypeSize(TileXR::TileXRDataType dtype)
{
    if (dtype == TileXR::TILEXR_DATA_TYPE_FP16 || dtype == TileXR::TILEXR_DATA_TYPE_BFP16) {
        return 2;
    }
    return TileXR::TILEXR_INVALID_VALUE;
}

bool TileXREpIsSupportedDataType(TileXR::TileXRDataType dtype)
{
    return TileXREpDataTypeSize(dtype) > 0;
}

int TileXREpBuildWindowConfig(int64_t rankSize, int64_t maxTokensPerSrc, int64_t hiddenSize, int64_t topK,
    int64_t expertNum, TileXR::TileXRDataType dtype, EpWindowConfig *config)
{
    if (config == nullptr || !IsPositive(rankSize) || !IsPositive(maxTokensPerSrc) || !IsPositive(hiddenSize) ||
        !IsPositive(topK) || !IsPositive(expertNum) || expertNum % rankSize != 0 || !TileXREpIsSupportedDataType(dtype)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    EpWindowConfig next {};
    next.rankSize = rankSize;
    next.localExpertNum = expertNum / rankSize;
    next.topK = topK;
    next.hiddenSize = hiddenSize;
    next.dtypeBytes = TileXREpDataTypeSize(dtype);

    if (!MulInt64(maxTokensPerSrc, topK, &next.maxRoutesPerSrc)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    if (!MulInt64(hiddenSize, next.dtypeBytes, &next.rowBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t payloadBytes = 0;
    if (!MulInt64(next.maxRoutesPerSrc, next.rowBytes, &payloadBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.payloadBytesPerSlot = TileXREpAlignUp(payloadBytes, kEpWindowAlignmentBytes);
    if (next.payloadBytesPerSlot == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t assistTupleBytes = 0;
    int64_t assistBytes = 0;
    if (!MulInt64(kEpAssistTupleInts, static_cast<int64_t>(sizeof(int32_t)), &assistTupleBytes) ||
        !MulInt64(next.maxRoutesPerSrc, assistTupleBytes, &assistBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.assistBytesPerSlot = TileXREpAlignUp(assistBytes, kEpWindowAlignmentBytes);
    if (next.assistBytesPerSlot == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t slotPayloadEnd = 0;
    if (!AddInt64(kEpSrcSlotHeaderBytes, next.payloadBytesPerSlot, &slotPayloadEnd) ||
        !AddInt64(slotPayloadEnd, next.assistBytesPerSlot, &next.slotBytes)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    next.slotBytes = TileXREpAlignUp(next.slotBytes, kEpWindowAlignmentBytes);
    if (next.slotBytes == TileXR::TILEXR_INVALID_VALUE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    int64_t slotsBytes = 0;
    if (!MulInt64(rankSize, next.slotBytes, &slotsBytes) ||
        !AddInt64(kEpWindowHeaderBytes, slotsBytes, &next.totalBytes) ||
        next.totalBytes > TileXR::IPC_BUFF_MAX_SIZE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    *config = next;
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpDstRank(int expertId, int localExpertNum)
{
    if (expertId < 0 || localExpertNum <= 0) {
        return static_cast<int>(TileXR::TILEXR_INVALID_VALUE);
    }
    return expertId / localExpertNum;
}

int TileXREpLocalExpert(int expertId, int localExpertNum)
{
    if (expertId < 0 || localExpertNum <= 0) {
        return static_cast<int>(TileXR::TILEXR_INVALID_VALUE);
    }
    return expertId % localExpertNum;
}

} // namespace TileXREp
