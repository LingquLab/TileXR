#include <cstdint>

#include "tilexr_ep_plan.h"

int main()
{
    constexpr int64_t rankSize = 4;
    constexpr int64_t s = 8;
    constexpr int64_t topK = 2;
    constexpr int64_t nvS = 16;

    uint64_t globalTokenId = TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID;
    if (TileXREp::Plan::EncodeMoonEPGlobalTokenId(
            1, 2, 1, rankSize, s, topK, &globalTokenId) != PLAN_OK ||
        globalTokenId != 21) {
        return 1;
    }

    int32_t srcRank = -1;
    int32_t tokenId = -1;
    int32_t topKId = -1;
    if (TileXREp::Plan::DecodeMoonEPGlobalTokenId(globalTokenId,
            rankSize, s, topK, &srcRank, &tokenId, &topKId) != PLAN_OK ||
        srcRank != 1 || tokenId != 2 || topKId != 1) {
        return 2;
    }

    const int32_t rawDst = 2 * static_cast<int32_t>(nvS) + 5;
    TileXREp::Plan::MoonEPRouteTarget target {};
    if (TileXREp::Plan::DecodeMoonEPDst(rawDst, nvS, rankSize, &target) != PLAN_OK ||
        target.rawDst != rawDst || target.dstRank != 2 || target.recvSlot != 5 ||
        target.sendHidden != 1 || target.writeRouteWeight != 1) {
        return 3;
    }

    TileXREp::Plan::MoonEPRouteDescriptor descriptor {};
    if (TileXREp::Plan::BuildMoonEPRouteDescriptor(1, 2, 1, ~rawDst,
            rankSize, s, topK, nvS, &descriptor) != PLAN_OK ||
        descriptor.srcRank != 1 || descriptor.tokenId != 2 || descriptor.topKId != 1 ||
        descriptor.globalTokenId != globalTokenId || descriptor.rawDst != rawDst ||
        descriptor.dstRank != 2 || descriptor.recvSlot != 5 ||
        descriptor.sendHidden != 0 || descriptor.writeRouteWeight != 1) {
        return 4;
    }

    return 0;
}
