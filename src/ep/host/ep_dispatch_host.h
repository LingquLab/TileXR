#ifndef TILEXR_EP_HOST_EP_DISPATCH_HOST_H
#define TILEXR_EP_HOST_EP_DISPATCH_HOST_H

#include <cstdint>

#include "acl/acl_base.h"
#include "ep_layout.h"
#include "tilexr_api.h"

namespace TileXREp {

struct EpDispatchParams {
    void *x = nullptr;
    int32_t *expertIds = nullptr;
    TileXRCommPtr comm = nullptr;
    int64_t bs = 0;
    int64_t h = 0;
    int64_t topK = 0;
    int64_t moeExpertNum = 0;
    void *expandXOut = nullptr;
    int64_t *expertTokenNumsOut = nullptr;
    int32_t *epRecvCountsOut = nullptr;
    int32_t *assistInfoForCombineOut = nullptr;
    TileXR::TileXRDataType dtype = TileXR::TILEXR_DATA_TYPE_RESERVED;
    aclrtStream stream = nullptr;
};

struct EpHostLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    EpWindowConfig window {};
};

int TileXREpValidateBasicDispatchParams(const EpDispatchParams &params);
int TileXREpValidateDispatchConfig(const EpDispatchParams &params, const TileXR::CommArgs &commArgs,
    EpWindowConfig *window);
int TileXREpPrepareLaunchContext(const EpDispatchParams &params, EpHostLaunchContext *context);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_DISPATCH_HOST_H
