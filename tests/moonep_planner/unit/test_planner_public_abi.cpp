#include <cstdint>
#include <type_traits>

#include "tilexr_moonep_planner.h"

using WorkspaceQueryV2 = int (*)(TileXRCommPtr, int64_t, int64_t, int64_t,
    uint64_t *, int64_t *);
using PlannerV2 = int (*)(const int32_t *, const int32_t *, TileXRCommPtr,
    int64_t, int64_t, int64_t, void *, uint64_t, int32_t *, int32_t *,
    int32_t *, int32_t *, int32_t *, uint64_t, aclrtStream);

static_assert(std::is_same<decltype(&TileXRMoonEpPlannerGetWorkspaceSizeV2),
    WorkspaceQueryV2>::value, "Planner workspace V2 ABI changed");
static_assert(std::is_same<decltype(&TileXRMoonEpPlannerV2),
    PlannerV2>::value, "Planner V2 ABI changed");
static_assert(TILEXR_MOONEP_PLANNER_STATUS_SUCCESS == 0,
    "Planner success status ABI changed");
static_assert(TILEXR_MOONEP_PLANNER_STATUS_TIMEOUT_BASE > 0,
    "Planner timeout status must encode a peer rank");

int main()
{
    return 0;
}

