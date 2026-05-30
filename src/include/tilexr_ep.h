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

}

#endif

#endif // TILEXR_EP_H
