#ifndef TILEXR_MOONEP_STAGE_HOST_H
#define TILEXR_MOONEP_STAGE_HOST_H

#include "moonep_peer_window.h"
#include "tilexr_api.h"
#include "tilexr_moonep.h"
#include "tilexr_types.h"

namespace TileXRMoonEp {
namespace detail {

inline bool StageLocalityValid(const TileXR::CommArgs &args)
{
    return args.rankSize > 0 && args.rankSize <= TileXR::TILEXR_MAX_RANK_SIZE &&
        args.rank >= 0 && args.rank < args.rankSize && args.localRankSize > 0 &&
        args.localRankSize <= args.rankSize && args.rankSize % args.localRankSize == 0 &&
        args.localRank >= 0 && args.localRank < args.localRankSize;
}

inline bool StagePeerWindowsReady(const TileXR::CommArgs &args)
{
    for (int rank = 0; rank < args.rankSize; ++rank) {
        if (args.peerMems[rank] == nullptr) {
            return false;
        }
    }
    return true;
}

} // namespace detail

template <typename LaunchContext>
int PrepareStageHost(TileXRCommPtr comm, LaunchContext *context)
{
    if (context == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    int ret = TileXRGetCommArgsHost(comm, context->hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = LaunchContext {};
        return ret;
    }
    if (context->hostArgs == nullptr || !detail::StageLocalityValid(*context->hostArgs)) {
        *context = LaunchContext {};
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    if ((context->hostArgs->extraFlag & TileXR::ExtraFlag::TOPO_910A5) == 0) {
        *context = LaunchContext {};
        return TILEXR_MOONEP_ERROR_NOT_SUPPORTED;
    }
    if (!detail::StagePeerWindowsReady(*context->hostArgs)) {
        *context = LaunchContext {};
        return TILEXR_MOONEP_ERROR_INTERNAL;
    }
    return TILEXR_MOONEP_SUCCESS;
}

template <typename LaunchContext>
int PrepareStageDevice(TileXRCommPtr comm, LaunchContext *context)
{
    if (context == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    int ret = TileXRGetCommArgsDev(comm, context->devArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = LaunchContext {};
        return ret;
    }
    if (context->devArgs == nullptr) {
        *context = LaunchContext {};
        return TILEXR_MOONEP_ERROR_INTERNAL;
    }
    ret = TileXRCommNextMagic(comm, &context->magic);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = LaunchContext {};
        return ret;
    }
    if (context->magic <= 0) {
        *context = LaunchContext {};
        return TILEXR_MOONEP_ERROR_INTERNAL;
    }
    context->waitIterations = kMoonEpPeerWaitIterations;
    return TILEXR_MOONEP_SUCCESS;
}

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_STAGE_HOST_H
