#ifndef TILEXR_EP_PLANNER_HOST_EP_PLAN_KERNEL_LAUNCH_H
#define TILEXR_EP_PLANNER_HOST_EP_PLAN_KERNEL_LAUNCH_H

#include "ep_plan_host.h"

namespace TileXREp {
namespace Plan {

int LaunchPlanKernel(
    TileXRCommPtr comm, const PlanHostArguments &arguments, const PlanHostContext &context);

} // namespace Plan
} // namespace TileXREp

#endif // TILEXR_EP_PLANNER_HOST_EP_PLAN_KERNEL_LAUNCH_H
