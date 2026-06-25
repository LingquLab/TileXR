#ifndef TILEXR_EP_H
#define TILEXR_EP_H

#ifdef __cplusplus

#include <cstdint>

#include "acl/acl_base.h"
#include "tilexr_api.h"
#include "tilexr_types.h"

// This public API is C++ header-compatible because it reuses TileXR namespace datatypes.
extern "C" {

int TileXRMoeEpDispatch(void *x, int32_t *expertIds, TileXRCommPtr comm,
    int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
    void *expandXOut, int64_t *expertTokenNumsOut, int32_t *epRecvCountsOut,
    int32_t *assistInfoForCombineOut, TileXR::TileXRDataType dtype, aclrtStream stream);

int TileXRMoeEpCombine(void *expertOut, int32_t *assistInfoForCombine, int32_t *epRecvCounts,
    TileXRCommPtr comm, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
    void *yOut, TileXR::TileXRDataType dtype, aclrtStream stream);

int TileXRMoeEpDispatchV2(void *x, int32_t *expertIds, void *scales, bool *xActiveMask, void *expertScales,
    TileXRCommPtr comm, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum, int64_t epWorldSize,
    int64_t epRankId, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, void *expandXOut,
    void *dynamicScalesOut, int32_t *assistInfoForCombineOut, int64_t *expertTokenNumsOut, int32_t *epRecvCountsOut,
    int32_t *tpRecvCountsOut, void *expandScalesOut, void *workspace, TileXR::TileXRDataType dtype,
    aclrtStream stream);

}

#endif

#endif // TILEXR_EP_H
