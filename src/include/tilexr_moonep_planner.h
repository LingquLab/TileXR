#ifndef TILEXR_MOONEP_PLANNER_H
#define TILEXR_MOONEP_PLANNER_H

#ifdef __cplusplus

#include <cstdint>

#include "acl/acl_base.h"
#include "tilexr_api.h"

extern "C" {

enum TileXRMoonEpPlannerStatus : int32_t {
    TILEXR_MOONEP_PLANNER_STATUS_SUCCESS = 0,
    TILEXR_MOONEP_PLANNER_STATUS_TIMEOUT_BASE = 1000,
};

int TileXRMoonEpPlannerGetWorkspaceSizeV2(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, uint64_t *workspaceBytes, int64_t *dispatchedCapacity);

int TileXRMoonEpPlannerV2(const int32_t *topkExpertIds, const int32_t *tokensPerExpert,
    TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount,
    void *workspace, uint64_t workspaceBytes, int32_t *dst, int32_t *cuSeqlens,
    int32_t *expertsToCopy, int32_t *remoteStats, int32_t *plannerStatus,
    uint64_t waitIterations, aclrtStream stream);

int TileXRMoonEpPlannerGetWorkspaceSizeV3(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, int64_t b, int64_t tokenPadding,
    uint64_t *workspaceBytes, int64_t *nvS);

int TileXRMoonEpPlannerGetDstLocalOffsetV3(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, int64_t b, int64_t tokenPadding,
    uint64_t *dstLocalOffset);

int TileXRMoonEpPlannerV3(const int32_t *topkExpertIds, const int32_t *tokensPerExpert,
    TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount,
    int64_t b, int64_t tokenPadding, void *workspace, uint64_t workspaceBytes,
    int32_t *dst, int32_t *cuSeqlens, int32_t *expertsToCopy,
    int32_t *zeroFillRanges, int32_t *remoteStats, int32_t *dupCounts,
    int32_t *plannerStatus, uint64_t waitIterations, aclrtStream stream);

}

#endif

#endif // TILEXR_MOONEP_PLANNER_H
