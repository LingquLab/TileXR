#ifndef TILEXR_MOONEP_REDUCE_GRAD_LAUNCH_H
#define TILEXR_MOONEP_REDUCE_GRAD_LAUNCH_H

#include "reduce_grad_host.h"

namespace TileXRMoonEp {

int TileXRMoonEpLaunchReduceGradKernel(
    const ReduceGradParams &params, const ReduceGradLaunchContext &context);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_REDUCE_GRAD_LAUNCH_H
