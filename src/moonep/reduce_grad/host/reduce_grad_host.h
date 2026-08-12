#ifndef TILEXR_MOONEP_REDUCE_GRAD_HOST_H
#define TILEXR_MOONEP_REDUCE_GRAD_HOST_H

#include <cstdint>

#include "acl/acl_base.h"
#include "reduce_grad_layout.h"
#include "tilexr_api.h"
#include "tilexr_moonep.h"
#include "tilexr_udma_reg.h"

namespace TileXRMoonEp {

struct ReduceGradPrepareParams {
    TileXRCommPtr comm = nullptr;
    const TileXRMoonEpPlanV1 *plan = nullptr;
    TileXRMoonEpTensorV1 *gradients[kReduceGradProjectionCount] = {};
    TileXRMoonEpReduceGradSourceSliceV2 sources[kReduceGradProjectionCount] = {};
    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    uint64_t requestedChunkBytes = 0;
};

struct ReduceGradLaunchParams {
    const TileXRMoonEpPlanV1 *plan = nullptr;
    TileXRMoonEpTensorV1 *gradients[kReduceGradProjectionCount] = {};
    TileXRMoonEpReduceGradSourceSliceV2 sources[kReduceGradProjectionCount] = {};
    TileXRMoonEpTensorV1 *status = nullptr;
    uint64_t waitIterations = 0;
    aclrtStream stream = nullptr;
};

struct ReduceGradPreparedContext {
    TileXRCommPtr comm = nullptr;
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    TileXRUDMAProfileHandle profileHandle = 0;
    TileXR::TileXRUDMAProfileView profileView {};
    ReduceGradLayout layout {};

    int64_t planN = 0;
    int64_t planK = 0;
    void *expertsToCopy = nullptr;
    TileXRMoonEpTensorV1 gradients[kReduceGradProjectionCount] = {};
    TileXRMoonEpReduceGradSourceSliceV2 sources[kReduceGradProjectionCount] = {};
    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
};

int TileXRMoonEpPrepareReduceGradLayout(TileXRCommPtr comm,
    const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *const gradients[kReduceGradProjectionCount],
    uint64_t requestedChunkBytes, ReduceGradLayout *layout);

int TileXRMoonEpCreateReduceGradPreparedContext(
    const ReduceGradPrepareParams &params, ReduceGradPreparedContext **context);

int TileXRMoonEpDestroyReduceGradPreparedContext(
    ReduceGradPreparedContext *context);

int TileXRMoonEpValidateReduceGradLaunch(const ReduceGradLaunchParams &params,
    const ReduceGradPreparedContext &context);

int TileXRMoonEpLaunchReduceGradKernel(const ReduceGradLaunchParams &params,
    const ReduceGradPreparedContext &context);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_REDUCE_GRAD_HOST_H
