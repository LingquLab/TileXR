#include <cstdint>
#include <iostream>
#include <limits>

#include "ep_plan_layout.h"
#include "ep_plan_peer_mailbox.h"
#include "tilexr_ep_plan.h"
#include "tilexr_types.h"

namespace {
int g_failures = 0;

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        ++g_failures;
    }
}

TileXRMoonEPPlanConfig ValidConfig()
{
    TileXRMoonEPPlanConfig config {};
    config.prefetchSlots = 4;
    config.rankTokenCapacity = 16;
    config.nvS = 32;
    config.tokenPadding = 4;
    config.tokenRouteLimitPerPair = 0;
    config.cardsPerServer = 8;
    config.cardsPerCabinet = 64;
    config.crossCandidateCount = 3;
    return config;
}

void TestValidLayout()
{
    const TileXRMoonEPPlanConfig config = ValidConfig();
    TileXREp::Plan::PlanWorkspaceLayout layout {};
    const int ret = TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout);
    Check(ret == TileXR::TILEXR_SUCCESS, "valid layout must succeed");
    Check(layout.local.totalBytes > 0, "local bytes must be non-zero");
    Check(layout.registeredMeta.totalBytes > 0, "registered bytes must be non-zero");
    Check(layout.local.totalBytes % 64 == 0, "local total must be 64-byte aligned");
    Check(layout.registeredMeta.totalBytes % 64 == 0, "registered total must be 64-byte aligned");
    Check(layout.registeredMeta.planCallHeaders.offset == 0, "fixed headers must be first");
    Check(layout.registeredMeta.tpe.offset >= layout.registeredMeta.planCallHeaders.bytes,
        "TPE must follow fixed headers");
    Check(layout.local.tokenSegments.bytes == 16 * sizeof(TileXREp::Plan::TokenSegmentMove),
        "local TokenSegmentMove capacity must be S*K");
}

void TestPublicQueryMatchesLayout()
{
    const TileXRMoonEPPlanConfig config = ValidConfig();
    TileXREp::Plan::PlanWorkspaceLayout layout {};
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout) == TileXR::TILEXR_SUCCESS,
        "internal layout query must succeed");
    uint64_t localBytes = 0;
    uint64_t registeredBytes = 0;
    const int ret = TileXRMoeEpPlanV2GetWorkspaceSize(8, 8, 2, 16, &config, &localBytes, &registeredBytes);
    Check(ret == TileXR::TILEXR_SUCCESS, "public workspace query must succeed");
    Check(localBytes == layout.local.totalBytes, "public local bytes mismatch");
    Check(registeredBytes == layout.registeredMeta.totalBytes, "public registered bytes mismatch");

    alignas(TileXRMoonEPPlanConfig) unsigned char configStorage[sizeof(TileXRMoonEPPlanConfig) + 1] = {};
    const TileXRMoonEPPlanConfig *misalignedConfig =
        reinterpret_cast<const TileXRMoonEPPlanConfig *>(configStorage + 1);
    Check(TileXRMoeEpPlanV2GetWorkspaceSize(8, 8, 2, 16, misalignedConfig,
        &localBytes, &registeredBytes) == TileXR::TILEXR_ERROR_PARA_CHECK_FAIL,
        "misaligned public config pointer must fail before dereference");

    alignas(uint64_t) unsigned char localStorage[sizeof(uint64_t) + 1] = {};
    uint64_t *misalignedLocalBytes = reinterpret_cast<uint64_t *>(localStorage + 1);
    Check(TileXRMoeEpPlanV2GetWorkspaceSize(8, 8, 2, 16, &config,
        misalignedLocalBytes, &registeredBytes) == TileXR::TILEXR_ERROR_PARA_CHECK_FAIL,
        "misaligned local workspace-size output must fail before write");

    alignas(uint64_t) unsigned char registeredStorage[sizeof(uint64_t) + 1] = {};
    uint64_t *misalignedRegisteredBytes = reinterpret_cast<uint64_t *>(registeredStorage + 1);
    Check(TileXRMoeEpPlanV2GetWorkspaceSize(8, 8, 2, 16, &config,
        &localBytes, misalignedRegisteredBytes) == TileXR::TILEXR_ERROR_PARA_CHECK_FAIL,
        "misaligned registered workspace-size output must fail before write");
}

void TestInvalidShapes()
{
    TileXREp::Plan::PlanWorkspaceLayout layout {};
    TileXRMoonEPPlanConfig config = ValidConfig();

    config.rankTokenCapacity = 15;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "CAP != S*K must fail");

    config = ValidConfig();
    config.prefetchSlots = 17;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout) ==
        TileXR::TILEXR_SUCCESS, "B > E remains a valid explicit workspace shape");

    config = ValidConfig();
    config.rankTokenCapacity = 33;
    config.nvS = 33;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(1, 1, 33, 1, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "K > 32 must fail");

    config = ValidConfig();
    config.prefetchSlots = 0;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "B == 0 must fail");

    config = ValidConfig();
    config.prefetchSlots = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    config.rankTokenCapacity = 1;
    config.nvS = 1;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(1, 1, 1, 1, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL,
        "B above the int32 slot-index representation must fail consistently");

    config = ValidConfig();
    config.nvS = 15;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "NvS < CAP must fail");

    config = ValidConfig();
    config.nvS = std::numeric_limits<int32_t>::max();
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(2, 8, 2, 16, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "R*NvS > INT32_MAX must fail");

    config = ValidConfig();
    config.prefetchSlots = 1;
    config.rankTokenCapacity = 1;
    config.nvS = 1;
    const int64_t expertNumOverUdmaByteCount =
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max() / sizeof(int32_t)) + 1;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(2, 1, 1, expertNumOverUdmaByteCount,
        config, &layout) == TileXR::TILEXR_ERROR_PARA_CHECK_FAIL,
        "one TPE row larger than UDMA uint32 byteCount must fail");

    config = ValidConfig();
    config.tokenRouteLimitPerPair = 17;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "route limit > CAP must fail");

    config = ValidConfig();
    config.cardsPerServer = 4;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "unsupported topology constants must fail");

    config = ValidConfig();
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, std::numeric_limits<int64_t>::max(), 2, 16,
        config, &layout) == TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "S*K overflow must fail");

    config = ValidConfig();
    config.prefetchSlots = 1;
    config.rankTokenCapacity = 1;
    config.nvS = 1;
    config.tokenPadding = 1;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(512, 1, 1, 512, config, &layout) ==
        TileXR::TILEXR_SUCCESS, "logical 512-rank workspace layout must succeed");
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(513, 1, 1, 513, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "rank count beyond logical 512 limit must fail");
}

void TestRankSlotEncodingBoundary()
{
    TileXREp::Plan::PlanWorkspaceLayout layout {};
    TileXRMoonEPPlanConfig config = ValidConfig();
    config.prefetchSlots = 1;
    config.rankTokenCapacity = 1;
    config.nvS = std::numeric_limits<int32_t>::max();
    config.tokenPadding = 1;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(1, 1, 1, 1, config, &layout) ==
        TileXR::TILEXR_SUCCESS, "R*NvS == INT32_MAX must succeed");

    config.nvS = static_cast<int64_t>(std::numeric_limits<int32_t>::max() / 2) + 1;
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(2, 1, 1, 2, config, &layout) ==
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "R*NvS == INT32_MAX + 1 must fail");
}

void TestRequiredShapeMatrix()
{
    const int64_t rankSizes[] = {1, 2, 8, 16, 128, 512};
    const int64_t topKs[] = {1, 2, 16, 32};
    const int64_t tokenCounts[] = {1, 32, 256, 8192};
    const int64_t paddings[] = {1, 8, 16, 32, 64};
    TileXREp::Plan::PlanWorkspaceLayout layout {};

    for (int64_t rankSize : rankSizes) {
        for (int64_t topK : topKs) {
            for (int64_t s : tokenCounts) {
                for (int64_t padding : paddings) {
                    TileXRMoonEPPlanConfig config = ValidConfig();
                    config.prefetchSlots = 1;
                    config.rankTokenCapacity = s * topK;
                    config.nvS = s * topK;
                    config.tokenPadding = padding;
                    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(
                        rankSize, s, topK, rankSize, config, &layout) == TileXR::TILEXR_SUCCESS,
                        "required logical shape matrix entry must succeed");

                    config.prefetchSlots = rankSize == 1 ? 1 : 2;
                    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(
                        rankSize, s, topK, rankSize, config, &layout) == TileXR::TILEXR_SUCCESS,
                        "required B=EPn-style shape matrix entry must succeed");
                }
            }
        }
    }
}


void TestPeerMailboxLayout()
{
    const int64_t rankSizes[] = {2, 8, 32};
    for (int64_t rankSize : rankSizes) {
        const TileXREp::Plan::PlanPeerMailboxLayout layout =
            TileXREp::Plan::BuildPlanPeerMailboxLayout(rankSize, 16);
        Check(layout.rowBytes % TileXREp::Plan::kPlanPeerMailboxTransferBytes == 0,
            "peer mailbox row must be 512-byte aligned");
        Check(layout.rowBytes == TileXREp::Plan::kPlanPeerMailboxRowBytes,
            "peer mailbox row stride must be configuration-independent");
        Check(layout.inputBytes <= layout.rowBytes &&
                layout.inputBytes % TileXREp::Plan::kPlanPeerMailboxTransferBytes == 0,
            "peer mailbox input transfer must fit and remain 512-byte aligned");
        Check(layout.totalBytes == static_cast<uint64_t>(rankSize) * layout.rowBytes,
            "peer mailbox total bytes must contain one row per source rank");
        Check(layout.header + TileXREp::Plan::kPlanHeaderStrideBytes <= layout.globalRankId,
            "peer mailbox header and global rank id must not overlap");
        Check(layout.globalRankId + sizeof(int32_t) <= layout.status,
            "peer mailbox global rank id and status must not overlap");
        Check(layout.status + TileXREp::Plan::kPlanStatusStrideBytes <= layout.tpe,
            "peer mailbox status and TPE must not overlap");
        Check(layout.tpe + 16 * sizeof(int32_t) <= layout.inputBytes,
            "peer mailbox TPE must fit in the input transfer");
        for (int64_t sourceRank = 1; sourceRank < rankSize; ++sourceRank) {
            Check(TileXREp::Plan::PlanPeerMailboxRowOffset(layout, sourceRank) >=
                    TileXREp::Plan::PlanPeerMailboxRowOffset(layout, sourceRank - 1) + layout.rowBytes,
                "peer mailbox source rows must not overlap");
        }
    }
    const TileXREp::Plan::PlanPeerMailboxLayout maximum =
        TileXREp::Plan::BuildPlanPeerMailboxLayout(TileXR::TILEXR_MAX_RANK_SIZE, 16);
    Check(maximum.totalBytes == static_cast<uint64_t>(TileXR::IPC_BUFF_MAX_SIZE),
        "maximum-rank peer mailbox must exactly fit the IPC data window");

    const int64_t expertCapacity = static_cast<int64_t>(
        (maximum.rowBytes - maximum.tpe) / sizeof(int32_t));
    const TileXREp::Plan::PlanPeerMailboxLayout oversized =
        TileXREp::Plan::BuildPlanPeerMailboxLayout(8, expertCapacity + 1);
    Check(oversized.inputBytes > oversized.rowBytes,
        "oversized peer mailbox input must be rejected by validation");

    const TileXREp::Plan::PlanPeerMailboxLayout differentExpertNum =
        TileXREp::Plan::BuildPlanPeerMailboxLayout(8, 32);
    Check(differentExpertNum.header == maximum.header &&
            differentExpertNum.globalRankId == maximum.globalRankId &&
            differentExpertNum.status == maximum.status &&
            differentExpertNum.tpe == maximum.tpe &&
            differentExpertNum.rowBytes == maximum.rowBytes,
        "peer mailbox addresses must not depend on expertNum");
}

void TestWorkspaceByteValidation()
{
    const TileXRMoonEPPlanConfig config = ValidConfig();
    TileXREp::Plan::PlanWorkspaceLayout layout {};
    Check(TileXREp::Plan::BuildPlanWorkspaceLayout(8, 8, 2, 16, config, &layout) == TileXR::TILEXR_SUCCESS,
        "layout setup must succeed");
    Check(TileXREp::Plan::ValidatePlanWorkspaceBytes(layout, layout.local.totalBytes,
        layout.registeredMeta.totalBytes) == TileXR::TILEXR_SUCCESS, "exact workspace sizes must pass");
    Check(TileXREp::Plan::ValidatePlanWorkspaceBytes(layout, layout.local.totalBytes - 1,
        layout.registeredMeta.totalBytes) == TileXR::TILEXR_ERROR_PARA_CHECK_FAIL,
        "one-byte-short local workspace must fail");
    Check(TileXREp::Plan::ValidatePlanWorkspaceBytes(layout, layout.local.totalBytes,
        layout.registeredMeta.totalBytes - 1) == TileXR::TILEXR_ERROR_PARA_CHECK_FAIL,
        "one-byte-short registered workspace must fail");
}
} // namespace

int main()
{
    TestValidLayout();
    TestPublicQueryMatchesLayout();
    TestInvalidShapes();
    TestRankSlotEncodingBoundary();
    TestRequiredShapeMatrix();
    TestPeerMailboxLayout();
    TestWorkspaceByteValidation();
    if (g_failures != 0) {
        std::cerr << g_failures << " plan layout tests failed" << std::endl;
        return 1;
    }
    std::cout << "Plan layout tests passed" << std::endl;
    return 0;
}
