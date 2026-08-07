#ifndef TILEXR_MOONEP_REDUCE_GRAD_HOST_H
#define TILEXR_MOONEP_REDUCE_GRAD_HOST_H

#include "acl/acl_base.h"
#include "reduce_grad_layout.h"
#include "tilexr_api.h"

namespace TileXRMoonEp {

struct ReduceGradParams {
    TileXRCommPtr comm = nullptr;
    const int32_t *expertsToCopy = nullptr;
    void *fullGateGrad = nullptr;
    void *fullUpGrad = nullptr;
    void *fullDownGrad = nullptr;
    void *gateReduceBuffer = nullptr;
    void *upReduceBuffer = nullptr;
    void *downReduceBuffer = nullptr;
    int32_t *status = nullptr;
    aclrtStream stream = nullptr;
};

struct ReduceGradLaunchContext {
    TileXR::CommArgs *hostArgs = nullptr;
    GM_ADDR devArgs = nullptr;
    ReduceGradLayout layout {};
    uint64_t waitIterations = 0;
    int64_t magic = 0;
};

int TileXRMoonEpPrepareReduceGradLaunch(const TileXRMoonEpReduceGradArgsV1 *args,
    aclrtStream stream, ReduceGradParams *params, ReduceGradLaunchContext *context);

int TileXRMoonEpRunReduceGradV1(
    const TileXRMoonEpReduceGradArgsV1 *args, aclrtStream stream);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_REDUCE_GRAD_HOST_H
