#ifndef TILEXR_MOONEP_DISPATCH_URMA_LAUNCH_H
#define TILEXR_MOONEP_DISPATCH_URMA_LAUNCH_H

#include "acl/acl_base.h"
#include "../common/dispatch_common.h"
#include "dispatch_layout.h"
#include "tilexr_api.h"

namespace TileXRMoonEp {

struct DispatchUrmaLaunchParams {
    GM_ADDR commArgs = nullptr;
    const void *input = nullptr;
    const int32_t *dst = nullptr;
    const int32_t *zeroFillRanges = nullptr;
    void *workspace = nullptr;
    void *output = nullptr;
    int32_t *planStatus = nullptr;
    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    DispatchPayloadMode mode = DispatchPayloadMode::Hidden;
    DispatchPeerMode peerMode = DispatchPeerMode::Legacy;
    uint32_t groupWidth = kDispatchDefaultGroupWidth;
    int64_t zeroFillRangeCount = 0;
    MoonEpDispatchUrmaLayout layout {};
};

int TileXRMoonEpLaunchDispatchUrmaKernel(const DispatchUrmaLaunchParams &params);

uint64_t TileXRMoonEpDispatchCompletionTimeoutTicks();

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_DISPATCH_URMA_LAUNCH_H
