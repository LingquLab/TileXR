#include <cstdint>
#include <iostream>
#include <vector>

#include "comm_args.h"
#include "ep_layout.h"
#include "ep_urma_combine.h"
#include "ep_urma_combine_layout.h"
#include "tilexr_types.h"

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

void TestUrmaCombineWorkspaceConfig()
{
    TileXREp::EpUrmaCombineWorkspaceConfig config {};
    const int ret = TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(2, 4, 8, 2, 3, &config);
    CheckInt("URMA combine valid config", ret, TileXR::TILEXR_SUCCESS);
    CheckInt64("URMA combine quant bytes", config.quantDataBytes, 32);
    CheckInt64("URMA combine comm bytes", config.commBytes, 64);
    CheckInt64("URMA combine blocks", config.blockCount, 1);
    CheckInt64("URMA combine route stride", config.routeStride, 512);
    CheckInt64("URMA combine route count", config.routeCount, 8);
    CheckInt64("URMA combine rx bytes", config.rxWindowBytes, 4096);
    CheckInt64("URMA combine rx0", config.rxWindowOffsets[0], 512);
    CheckInt64("URMA combine rx1", config.rxWindowOffsets[1], 4608);
    CheckInt64("URMA combine round0", config.roundDoneOffsets[0], 8704);
    CheckInt64("URMA combine round1", config.roundDoneOffsets[1], 8832);
    CheckInt64("URMA combine rx lane done", config.rxLaneDoneOffset, 8960);
    CheckInt64("URMA combine sender done", config.senderDoneOffset,
        config.rxLaneDoneOffset + TileXREp::kEpUrmaCombinePackLaneCount *
            TileXREp::kEpUrmaCombineCacheLineBytes);
    CheckInt64("URMA combine round publish", config.roundPublishOffset, 13056);
    CheckInt64("URMA combine round credit", config.roundCreditOffset,
        TileXREp::kEpUrmaCombineDeferredRoundCredit ? 13120 : 0);
    CheckInt64("URMA combine start gate", config.startGateOffset,
        TileXREp::kEpUrmaCombineStartGate ?
            (TileXREp::kEpUrmaCombineDeferredRoundCredit ? 13184 : 13120) : 0);
    CheckInt64("URMA combine error", config.errorStatusOffset,
        TileXREp::kEpUrmaCombineStartGate ?
            (TileXREp::kEpUrmaCombineDeferredRoundCredit ? 13312 : 13248) :
            (TileXREp::kEpUrmaCombineDeferredRoundCredit ? 13184 : 13120));
    CheckInt64("URMA combine fixed bytes", config.fixedBytes,
        TileXREp::kEpUrmaCombineStartGate ?
            (TileXREp::kEpUrmaCombineDeferredRoundCredit ? 13376 : 13312) :
            (TileXREp::kEpUrmaCombineDeferredRoundCredit ? 13248 : 13184));
    CheckInt64("URMA combine fixed control boundary", config.fixedBytes,
        config.errorStatusOffset + TileXREp::kEpUrmaCombineCacheLineBytes);
    CheckInt64("URMA combine tx ready", config.txReadyOffset,
        config.fixedBytes);
    CheckInt64("URMA combine tx data", config.txDataOffset, 13824);
    CheckInt64("URMA combine required bytes", config.requiredBytes, 15360);
}

void TestUrmaCombineStartGateLayout()
{
    TileXREp::EpUrmaCombineWorkspaceConfig config {};
    CheckInt("URMA start gate config", TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(
        64, 128, 7168, 8, 16, &config), TileXR::TILEXR_SUCCESS);
    if (TileXREp::kEpUrmaCombineStartGate) {
        CheckInt64("URMA start gate follows publish", config.startGateOffset,
            (TileXREp::kEpUrmaCombineDeferredRoundCredit ? config.roundCreditOffset :
                config.roundPublishOffset) + TileXREp::kEpUrmaCombineCacheLineBytes);
        CheckInt64("URMA start gate rank lines", config.errorStatusOffset,
            config.startGateOffset + config.rankSize * TileXREp::kEpUrmaCombineCacheLineBytes);
        CheckInt64("URMA start gate required QPs", TileXREp::kEpUrmaCombineRequiredQpCount,
            TileXREp::kEpUrmaCombineSendLaneCount);
    } else {
        CheckInt64("URMA disabled start gate offset", config.startGateOffset, 0);
        CheckInt64("URMA disabled start gate keeps error", config.errorStatusOffset,
            (TileXREp::kEpUrmaCombineDeferredRoundCredit ? config.roundCreditOffset :
                config.roundPublishOffset) + TileXREp::kEpUrmaCombineCacheLineBytes);
        CheckInt64("URMA disabled start gate QPs", TileXREp::kEpUrmaCombineRequiredQpCount,
            TileXREp::kEpUrmaCombineSendLaneCount);
    }
}

void TestParallelRoundPublishPeerShards()
{
    constexpr int64_t kParallelSendLanes = TileXREp::kEpUrmaCombineSendLaneCount;
    const int64_t rankSizes[] = {1, 2, 8, 15, 16, 17, 22, 23, 64, 128};
    for (int64_t rankSize : rankSizes) {
        for (int64_t rank = 0; rank < rankSize; ++rank) {
            std::vector<int> visits(static_cast<std::size_t>(rankSize), 0);
            int64_t publishCount = 0;
            for (int64_t senderId = 0; senderId < kParallelSendLanes; ++senderId) {
                for (int64_t peer = senderId; peer < rankSize; peer += kParallelSendLanes) {
                    if (peer == rank) {
                        continue;
                    }
                    ++visits[static_cast<std::size_t>(peer)];
                    ++publishCount;
                }
            }
            CheckInt64("parallel publish excludes self", visits[static_cast<std::size_t>(rank)], 0);
            CheckInt64("parallel publish peer count", publishCount, rankSize - 1);
            for (int64_t peer = 0; peer < rankSize; ++peer) {
                if (peer != rank) {
                    CheckInt("parallel publish peer owned once",
                        visits[static_cast<std::size_t>(peer)], 1);
                }
            }
        }
    }
    CheckInt("parallel publish control step follows release",
        static_cast<int>(TileXREp::kEpUrmaCombinePublishDone),
        static_cast<int>(TileXREp::kEpUrmaCombineRxBufferReleased + 1));
}

void TestDeferredRoundCreditRxCompletionShards()
{
    const int64_t rankSizes[] = {1, 2, 8, 15, 16, 17, 22, 23, 64, 128};
    for (int64_t rankSize : rankSizes) {
        const int64_t publisherCount = rankSize < TileXREp::kEpUrmaCombineSendLaneCount ?
            rankSize : TileXREp::kEpUrmaCombineSendLaneCount;
        std::vector<int> visits(
            static_cast<std::size_t>(TileXREp::kEpUrmaCombinePackLaneCount), 0);
        for (int64_t senderId = 0; senderId < publisherCount; ++senderId) {
            for (int64_t lane = senderId; lane < TileXREp::kEpUrmaCombinePackLaneCount;
                 lane += publisherCount) {
                ++visits[static_cast<std::size_t>(lane)];
            }
        }
        CheckInt64("RX completion publisher count", publisherCount,
            rankSize < TileXREp::kEpUrmaCombineSendLaneCount ?
                rankSize : TileXREp::kEpUrmaCombineSendLaneCount);
        for (int visitsForLane : visits) {
            CheckInt("RX completion lane owned once", visitsForLane, 1);
        }
    }
}

void TestUrmaCombineWorkspaceRejectsInvalidConfig()
{
    TileXREp::EpUrmaCombineWorkspaceConfig config {};
    CheckInt("URMA combine null out", TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(2, 4, 8, 2, 3, nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("URMA combine negative sends",
        TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(2, 4, 8, 2, -1, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("URMA combine oversized hidden", TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(
        2, 4, TileXREp::kEpUrmaCombineMaxHidden + 1, 2, 3, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestUrmaCombineDataAsFlagPayloadBoundary()
{
    TileXREp::EpUrmaCombineWorkspaceConfig exact {};
    TileXREp::EpUrmaCombineWorkspaceConfig spill {};
    CheckInt("URMA exact payload boundary",
        TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(2, 2, 448, 1, 2, &exact),
        TileXR::TILEXR_SUCCESS);
    CheckInt("URMA payload spill",
        TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(2, 2, 449, 1, 2, &spill),
        TileXR::TILEXR_SUCCESS);
    CheckInt64("URMA exact comm bytes", exact.commBytes, 480);
    CheckInt64("URMA exact block count", exact.blockCount, 1);
    CheckInt64("URMA spill comm bytes", spill.commBytes, 512);
    CheckInt64("URMA spill block count", spill.blockCount, 2);
    CheckInt64("URMA spill route stride", spill.routeStride, 1024);
}

void TestUrmaCombineDynamicTxDoesNotMoveRemoteRegions()
{
    TileXREp::EpUrmaCombineWorkspaceConfig small {};
    TileXREp::EpUrmaCombineWorkspaceConfig large {};
    CheckInt("URMA small dynamic tx", TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(4, 8, 7168, 8, 3, &small),
        TileXR::TILEXR_SUCCESS);
    CheckInt("URMA large dynamic tx", TileXREp::TileXREpBuildUrmaCombineWorkspaceConfig(4, 8, 7168, 8, 99, &large),
        TileXR::TILEXR_SUCCESS);
    CheckInt64("URMA dynamic tx keeps rx0", small.rxWindowOffsets[0], large.rxWindowOffsets[0]);
    CheckInt64("URMA dynamic tx keeps rx1", small.rxWindowOffsets[1], large.rxWindowOffsets[1]);
    CheckInt64("URMA dynamic tx keeps round0", small.roundDoneOffsets[0], large.roundDoneOffsets[0]);
    CheckInt64("URMA dynamic tx keeps fixed bytes", small.fixedBytes, large.fixedBytes);
    CheckBool("URMA dynamic tx grows required bytes", large.requiredBytes > small.requiredBytes, true);
}

} // namespace

int main()
{
    TestExpertMapping();
    TestDataTypes();
    TestWindowConfig();
    TestRejectsInvalidConfig();
    TestUrmaCombineWorkspaceConfig();
    TestUrmaCombineStartGateLayout();
    TestParallelRoundPublishPeerShards();
    TestDeferredRoundCreditRxCompletionShards();
    TestUrmaCombineWorkspaceRejectsInvalidConfig();
    TestUrmaCombineDataAsFlagPayloadBoundary();
    TestUrmaCombineDynamicTxDoesNotMoveRemoteRegions();
    return g_failures == 0 ? 0 : 1;
}
