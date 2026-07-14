#include <cstdint>
#include <iostream>

#include "comm_args.h"
#include "ep_layout.h"
#include "tilexr_types.h"
#include "tilexr_udma_types.h"

namespace {

int g_failures = 0;

void CheckInt64(const char *label, int64_t actual, int64_t expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckInt(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckBool(const char *label, bool actual, bool expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void TestExpertMapping()
{
    CheckInt("expert 0 dst", TileXREp::TileXREpDstRank(0, 4), 0);
    CheckInt("expert 0 local", TileXREp::TileXREpLocalExpert(0, 4), 0);
    CheckInt("expert 3 dst", TileXREp::TileXREpDstRank(3, 4), 0);
    CheckInt("expert 3 local", TileXREp::TileXREpLocalExpert(3, 4), 3);
    CheckInt("expert 4 dst", TileXREp::TileXREpDstRank(4, 4), 1);
    CheckInt("expert 4 local", TileXREp::TileXREpLocalExpert(4, 4), 0);
    CheckInt("expert 7 dst", TileXREp::TileXREpDstRank(7, 4), 1);
    CheckInt("expert 7 local", TileXREp::TileXREpLocalExpert(7, 4), 3);
    CheckInt("negative expert dst", TileXREp::TileXREpDstRank(-1, 4), TileXR::TILEXR_INVALID_VALUE);
    CheckInt("zero local expert dst", TileXREp::TileXREpDstRank(1, 0), TileXR::TILEXR_INVALID_VALUE);
}

void TestDataTypes()
{
    CheckBool("fp16 supported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_FP16), true);
    CheckBool("bf16 supported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_BFP16), true);
    CheckBool("fp32 unsupported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_FP32), false);
    CheckInt64("fp16 bytes", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_FP16), 2);
    CheckInt64("bf16 bytes", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_BFP16), 2);
    CheckInt64("int32 bytes invalid", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_INT32),
        TileXR::TILEXR_INVALID_VALUE);
}

void TestDataAsFlagSizing()
{
    CheckInt64("daf empty invalid", TileXREp::TileXREpDataAsFlagSlotBytes(0),
        TileXR::TILEXR_INVALID_VALUE);
    CheckInt64("daf 1 byte", TileXREp::TileXREpDataAsFlagSlotBytes(1), 512);
    CheckInt64("daf exact payload", TileXREp::TileXREpDataAsFlagSlotBytes(480), 512);
    CheckInt64("daf two blocks", TileXREp::TileXREpDataAsFlagSlotBytes(481), 1024);
    CheckInt64("daf total rank 4", TileXREp::TileXREpDataAsFlagTotalBytes(4, 481), 4096);
    CheckInt64("daf pingpong rank 4", TileXREp::TileXREpDataAsFlagPingPongTotalBytes(4, 481), 8192);
}

void TestCombineDataAsFlagWorkspace()
{
    const int64_t rankSize = 2;
    const int64_t totalBytes = 704;
    const int64_t slotBytes = 320;
    const int64_t firstOperation = TileXREp::TileXREpUdmaOperationBytes(totalBytes, rankSize, slotBytes);
    const int64_t alignedTotal = TileXREp::TileXREpAlignUp(totalBytes, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    const int64_t dafTotal = TileXREp::TileXREpDataAsFlagTotalBytes(rankSize, slotBytes);
    const int64_t alignedDafTotal = TileXREp::TileXREpAlignUp(dafTotal, TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);
    const int64_t recvOffset = firstOperation;
    const int64_t sendOffset = recvOffset + alignedTotal;
    const int64_t sendRound0Offset = sendOffset;
    const int64_t sendRound1Offset = sendOffset + alignedDafTotal;
    const int64_t remoteRecvOffset = sendOffset + 2 * alignedDafTotal;
    const int64_t remoteRecvRound0Offset = remoteRecvOffset;
    const int64_t remoteRecvRound1Offset = remoteRecvOffset + alignedDafTotal;
    const int64_t statusOffset = remoteRecvOffset + 2 * alignedDafTotal;
    const int64_t required = TileXREp::TileXREpAlignUp(
        statusOffset + static_cast<int64_t>(sizeof(uint64_t)), TileXR::TILEXR_UDMA_CACHE_LINE_SIZE);

    CheckInt64("combine daf recv offset",
        TileXREp::TileXREpCombineDataAsFlagRecvWindowOffset(totalBytes, rankSize, slotBytes), recvOffset);
    CheckInt64("combine daf send offset",
        TileXREp::TileXREpCombineDataAsFlagSendWindowOffset(totalBytes, rankSize, slotBytes), sendOffset);
    CheckInt64("combine daf send round 0 offset",
        TileXREp::TileXREpCombineDataAsFlagSendRoundOffset(totalBytes, rankSize, slotBytes, 0), sendRound0Offset);
    CheckInt64("combine daf send round 1 offset",
        TileXREp::TileXREpCombineDataAsFlagSendRoundOffset(totalBytes, rankSize, slotBytes, 1), sendRound1Offset);
    CheckInt64("combine daf send round invalid",
        TileXREp::TileXREpCombineDataAsFlagSendRoundOffset(totalBytes, rankSize, slotBytes, 2),
        TileXR::TILEXR_INVALID_VALUE);
    CheckInt64("combine daf remote recv offset",
        TileXREp::TileXREpCombineDataAsFlagRemoteRecvWindowOffset(totalBytes, rankSize, slotBytes), remoteRecvOffset);
    CheckInt64("combine daf remote recv round 0 offset",
        TileXREp::TileXREpCombineDataAsFlagRemoteRecvRoundOffset(totalBytes, rankSize, slotBytes, 0),
        remoteRecvRound0Offset);
    CheckInt64("combine daf remote recv round 1 offset",
        TileXREp::TileXREpCombineDataAsFlagRemoteRecvRoundOffset(totalBytes, rankSize, slotBytes, 1),
        remoteRecvRound1Offset);
    CheckInt64("combine daf remote recv round invalid",
        TileXREp::TileXREpCombineDataAsFlagRemoteRecvRoundOffset(totalBytes, rankSize, slotBytes, -1),
        TileXR::TILEXR_INVALID_VALUE);
    CheckInt64("combine daf status offset",
        TileXREp::TileXREpCombineDataAsFlagStatusOffset(totalBytes, rankSize, slotBytes), statusOffset);
    CheckInt64("combine daf required",
        TileXREp::TileXREpCombineDataAsFlagRequiredWorkspaceBytes(totalBytes, rankSize, slotBytes), required);
}

void TestWindowConfig()
{
    TileXREp::EpWindowConfig config {};
    const int ret = TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP16, &config);
    CheckInt("valid config ret", ret, TileXR::TILEXR_SUCCESS);
    CheckInt64("rank size", config.rankSize, 2);
    CheckInt64("bs", config.bs, 4);
    CheckInt64("hidden size", config.h, 8);
    CheckInt64("topk", config.topK, 2);
    CheckInt64("moe experts", config.moeExpertNum, 8);
    CheckInt64("local experts", config.localExpertNum, 4);
    CheckInt64("dtype bytes", config.dtypeBytes, 2);
    CheckInt64("max routes", config.maxRoutesPerSrc, 8);
    CheckInt64("row bytes", config.rowBytes, 16);
    CheckInt64("payload bytes", config.payloadBytesPerSlot, 128);
    CheckInt64("assist bytes", config.assistBytesPerSlot, 128);
    CheckInt64("slot bytes", config.slotBytes, 320);
    CheckInt64("total bytes", config.totalBytes, 704);
}

void TestRejectsInvalidConfig()
{
    TileXREp::EpWindowConfig config {};
    CheckInt("null out", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP16, nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("non-divisible experts", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 7, TileXR::TILEXR_DATA_TYPE_FP16, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("unsupported dtype", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP32, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("rank size too large", TileXREp::TileXREpBuildWindowConfig(
        TileXR::TILEXR_MAX_RANK_SIZE + 1, 4, 8, 2, TileXR::TILEXR_MAX_RANK_SIZE + 1,
        TileXR::TILEXR_DATA_TYPE_FP16, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("oversized window", TileXREp::TileXREpBuildWindowConfig(
        2, 1024 * 1024, 64, 8, 8, TileXR::TILEXR_DATA_TYPE_FP16, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestExpertMapping();
    TestDataTypes();
    TestDataAsFlagSizing();
    TestCombineDataAsFlagWorkspace();
    TestWindowConfig();
    TestRejectsInvalidConfig();
    return g_failures == 0 ? 0 : 1;
}
