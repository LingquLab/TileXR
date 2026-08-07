#ifndef TILEXR_MOONEP_REDUCE_GRAD_HOST_H
#define TILEXR_MOONEP_REDUCE_GRAD_HOST_H

#include <cstdint>

#include "acl/acl_base.h"
#include "reduce_grad_layout.h"
#include "tilexr_api.h"
#include "tilexr_moonep.h"
#include "tilexr_udma_reg.h"

namespace TileXRMoonEp {

struct ReduceGradParams {
    TileXRCommPtr comm = nullptr;
    const TileXRMoonEpPlanV1 *plan = nullptr;
    TileXRMoonEpTensorV1 *gradients[kReduceGradProjectionCount] = {};
    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    TileXRMoonEpTensorV1 *status = nullptr;
    uint64_t waitIterations = 0;
    uint64_t requestedUdmaChunkBytes = 0;
    aclrtStream stream = nullptr;
};

struct ReduceGradLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    const TileXR::TileXRUDMARegistry *registry = nullptr;
    ReduceGradLayout layout {};
};

int TileXRMoonEpPrepareReduceGradLayout(TileXRCommPtr comm,
    const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *const gradients[kReduceGradProjectionCount],
    uint64_t requestedUdmaChunkBytes, ReduceGradLayout *layout);

int TileXRMoonEpPrepareReduceGradLaunchContext(const ReduceGradParams &params,
    ReduceGradLaunchContext *context);

int TileXRMoonEpLaunchReduceGradKernel(const ReduceGradParams &params,
    const ReduceGradLaunchContext &context);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_REDUCE_GRAD_HOST_H
