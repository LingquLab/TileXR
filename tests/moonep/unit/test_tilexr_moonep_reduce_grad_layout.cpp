#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "comm_args.h"
#include "reduce_grad_common.h"
#include "reduce_grad_layout.h"
#include "tilexr_moonep.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string &message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

void CheckStatus(const std::string &label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

void TestThresholdAndMixedLayout()
{
    const uint64_t rows[TileXRMoonEp::kReduceGradProjectionCount] = {
        (TILEXR_MOONEP_REDUCE_GRAD_UDMA_THRESHOLD_BYTES - sizeof(float)) / sizeof(float),
        TILEXR_MOONEP_REDUCE_GRAD_UDMA_THRESHOLD_BYTES / sizeof(float),
        TILEXR_MOONEP_REDUCE_GRAD_UDMA_THRESHOLD_BYTES / sizeof(float) + 1,
    };
    TileXRMoonEp::ReduceGradLayout layout {};
    CheckStatus("mixed layout", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 2, 4, rows, UINT64_C(512) << 20, 0, &layout), TileXR::TILEXR_SUCCESS);
    Check(layout.transports[0] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER,
        "below-threshold row must use peer memory");
    Check(layout.transports[1] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER,
        "exactly 1 MiB row must use peer memory");
    Check(layout.transports[2] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA,
        "row above 1 MiB must use UDMA");
    Check(layout.udmaQpCount == 0,
        "layout builder must leave negotiated UDMA QP count to the host");
    Check(layout.rowBytes[1] == UINT64_C(1) << 20,
        "exact-threshold row byte calculation mismatch");
    Check(layout.rowBytes[2] == (UINT64_C(1) << 20) + sizeof(float),
        "above-threshold row byte calculation mismatch");
    Check(layout.peerRecordBaseOffset == UINT64_C(1) << 20,
        "peer state reservation mismatch");
    Check(layout.peerHalfBytes > 0 && layout.peerSlotStrideBytes >= 512 &&
        layout.peerChunkPayloadBytes > 0, "peer layout must provide usable records");
    Check(layout.udmaChunkBytes >= (UINT64_C(1) << 20) &&
        layout.udmaChunkBytes <= UINT32_MAX, "UDMA chunk bounds mismatch");
    Check(layout.udmaOutboundOffset % TileXRMoonEp::kReduceGradUdmaAlignment == 0 &&
        layout.udmaInboundOffset % TileXRMoonEp::kReduceGradUdmaAlignment == 0 &&
        layout.workspaceBytes % TileXRMoonEp::kReduceGradUdmaWorkspaceAlignment == 0,
        "UDMA workspace alignment mismatch");
    Check(layout.udmaInboundOffset > layout.udmaOutboundOffset &&
        layout.workspaceBytes > layout.udmaInboundOffset,
        "UDMA workspace regions overlap or are empty");
    Check(TileXRMoonEp::kReduceGradUdmaSignalStageStride ==
            TileXR::TILEXR_UDMA_CACHE_LINE_SIZE &&
        TileXRMoonEp::kReduceGradUdmaCompletionOffset >=
            2 * TileXR::TILEXR_UDMA_CACHE_LINE_SIZE &&
        TileXRMoonEp::kReduceGradUdmaPollScratchOffset >=
            TileXRMoonEp::kReduceGradUdmaCompletionOffset +
                2 * TileXR::TILEXR_UDMA_CACHE_LINE_SIZE &&
        TileXRMoonEp::kReduceGradUdmaPeerStateBytes >=
            6 * TileXR::TILEXR_UDMA_CACHE_LINE_SIZE,
        "each UDMA stage signal must use a distinct cache line");
}

void TestCapacityInjection()
{
    const uint64_t rows[TileXRMoonEp::kReduceGradProjectionCount] = {64, 128, 256};
    TileXRMoonEp::ReduceGradLayout mainLayout {};
    TileXRMoonEp::ReduceGradLayout pr90Layout {};
    CheckStatus("main capacity", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 8, 64, rows, UINT64_C(100) << 20, 0, &mainLayout), TileXR::TILEXR_SUCCESS);
    CheckStatus("PR90 capacity", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 8, 64, rows, UINT64_C(512) << 20, 0, &pr90Layout), TileXR::TILEXR_SUCCESS);
    Check(pr90Layout.peerHalfBytes > mainLayout.peerHalfBytes,
        "injected PR90 capacity must increase the peer half");
    Check(pr90Layout.peerSlotStrideBytes > mainLayout.peerSlotStrideBytes,
        "injected PR90 capacity must increase the slot stride");
    Check(TileXRMoonEp::TileXRMoonEpReduceGradPeerWindowBytes() ==
        static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE),
        "capacity resolver must follow IPC_BUFF_MAX_SIZE");
}

void TestPureTransportLayouts()
{
    const uint64_t peerRows[TileXRMoonEp::kReduceGradProjectionCount] = {1, 2, 3};
    const uint64_t udmaRows[TileXRMoonEp::kReduceGradProjectionCount] = {
        (UINT64_C(1) << 18) + 1,
        (UINT64_C(1) << 18) + 2,
        (UINT64_C(1) << 18) + 3,
    };
    TileXRMoonEp::ReduceGradLayout layout {};
    CheckStatus("peer without capacity", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 2, 4, peerRows, 0, 0, &layout), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckStatus("UDMA without peer capacity", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 2, 4, udmaRows, 0, UINT64_C(2) << 20, &layout), TileXR::TILEXR_SUCCESS);
    Check(layout.peerHalfBytes == 0 && layout.workspaceBytes > 0,
        "UDMA-only layout must not reserve peer records");
    Check(layout.controlBlockCount == 1,
        "two-rank UDMA layout must assign one control block");
}

void TestSingleRankLargeRowsStayLocal()
{
    const uint64_t largeRow =
        TILEXR_MOONEP_REDUCE_GRAD_UDMA_THRESHOLD_BYTES / sizeof(float) + 1;
    const uint64_t rows[TileXRMoonEp::kReduceGradProjectionCount] = {
        largeRow, largeRow + 1, largeRow + 2};
    TileXRMoonEp::ReduceGradLayout layout {};
    CheckStatus("single-rank large rows", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 1, 4, rows, UINT64_C(100) << 20, 0, &layout), TileXR::TILEXR_SUCCESS);
    Check(layout.transports[0] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER &&
        layout.transports[1] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER &&
        layout.transports[2] == TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER,
        "single-rank rows must stay on the local peer path regardless of row size");
    Check(layout.workspaceBytes == 0 && layout.udmaChunkBytes == 0,
        "single-rank rows must not request a UDMA workspace");
    Check(layout.controlBlockCount == 0 && layout.peerChunkPayloadBytes > 0,
        "single-rank layout must reserve all blocks for local reduction");
}

void TestInvalidInputs()
{
    const uint64_t rows[TileXRMoonEp::kReduceGradProjectionCount] = {1, 1, 1};
    TileXRMoonEp::ReduceGradLayout layout {};
    CheckStatus("null output", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 2, 4, rows, UINT64_C(100) << 20, 0, nullptr), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckStatus("bad rank", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        2, 2, 4, rows, UINT64_C(100) << 20, 0, &layout), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckStatus("nondivisible experts", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 2, 3, rows, UINT64_C(100) << 20, 0, &layout), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    uint64_t overflowRows[TileXRMoonEp::kReduceGradProjectionCount] = {
        std::numeric_limits<uint64_t>::max(), 1, 1};
    CheckStatus("row byte overflow", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 2, 4, overflowRows, UINT64_C(100) << 20, 0, &layout),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    const uint64_t udmaRows[TileXRMoonEp::kReduceGradProjectionCount] = {
        (UINT64_C(1) << 18) + 1,
        (UINT64_C(1) << 18) + 1,
        (UINT64_C(1) << 18) + 1,
    };
    CheckStatus("small UDMA chunk", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 2, 4, udmaRows, 0, (UINT64_C(1) << 20) - 1, &layout),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckStatus("oversized UDMA chunk", TileXRMoonEp::TileXRMoonEpBuildReduceGradLayout(
        0, 2, 4, udmaRows, 0, UINT64_C(1) << 32, &layout),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestThresholdAndMixedLayout();
    TestCapacityInjection();
    TestPureTransportLayouts();
    TestSingleRankLargeRowsStayLocal();
    TestInvalidInputs();
    return g_failures == 0 ? 0 : 1;
}
