#include "ep_dispatch_host.h"

#include "comm_args.h"
#include "tilexr_types.h"

namespace TileXREp {

int TileXREpValidateBasicDispatchParams(const EpDispatchParams &params)
{
    if (params.x == nullptr || params.expertIds == nullptr || params.comm == nullptr || params.expandXOut == nullptr ||
        params.expertTokenNumsOut == nullptr || params.epRecvCountsOut == nullptr ||
        params.assistInfoForCombineOut == nullptr || params.stream == nullptr || params.bs <= 0 || params.h <= 0 ||
        params.topK <= 0 || params.moeExpertNum <= 0 || !TileXREpIsSupportedDataType(params.dtype)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TileXR::TILEXR_SUCCESS;
}

int TileXREpValidateDispatchConfig(const EpDispatchParams &params, const TileXR::CommArgs &commArgs,
    EpWindowConfig *window)
{
    if (window == nullptr || commArgs.rankSize <= 0 || commArgs.rankSize > TileXR::TILEXR_MAX_RANK_SIZE ||
        commArgs.rank < 0 || commArgs.rank >= commArgs.rankSize) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    for (int rank = 0; rank < commArgs.rankSize; ++rank) {
        if (commArgs.peerMems[rank] == nullptr) {
            return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
        }
    }

    return TileXREpBuildWindowConfig(commArgs.rankSize, params.bs, params.h, params.topK, params.moeExpertNum,
        params.dtype, window);
}

} // namespace TileXREp
