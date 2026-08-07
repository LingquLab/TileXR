#ifndef TILEXR_MOONEP_PLANNER_LAYOUT_H
#define TILEXR_MOONEP_PLANNER_LAYOUT_H

#include <cstdint>

namespace TileXRMoonEp {

struct PlannerLayout {
    int64_t rankSize = 0;
    int64_t s = 0;
    int64_t k = 0;
    int64_t expertCount = 0;
    int64_t expertsPerRank = 0;
    int64_t routeCount = 0;
    int64_t dispatchedCapacity = 0;
    int64_t blockDim = 0;

    uint64_t tpePrefixOffset = 0;
    uint64_t blockHistogramOffset = 0;
    uint64_t allocPrefixOffset = 0;
    uint64_t expertOffsetsOffset = 0;
    uint64_t zOffset = 0;
    uint64_t groupTotalsOffset = 0;
    uint64_t workspaceBytes = 0;
    uint64_t peerPublicationBytes = 0;
};

uint64_t TileXRMoonEpAlignUp(uint64_t value, uint64_t alignment);

int TileXRMoonEpBuildPlannerLayout(int64_t rankSize, int64_t s, int64_t k,
    int64_t expertCount, PlannerLayout *out);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PLANNER_LAYOUT_H
