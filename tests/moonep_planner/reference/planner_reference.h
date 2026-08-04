#ifndef TILEXR_TEST_MOONEP_PLANNER_REFERENCE_H
#define TILEXR_TEST_MOONEP_PLANNER_REFERENCE_H

#include <cstdint>
#include <string>
#include <vector>

namespace TileXRMoonEpTest {

struct ReferencePlan {
    int64_t dispatchedCapacity = 0;
    std::vector<int32_t> dst;
    std::vector<int32_t> cuSeqlens;
    std::vector<int32_t> expertsToCopy;
    std::vector<int32_t> remoteStats;
};

bool BuildReferencePlan(int32_t rank, int32_t rankSize, int64_t s, int64_t k,
    int64_t expertCount, const std::vector<int32_t> &allTopk,
    ReferencePlan *plan, std::string *error);

std::vector<int32_t> MakeRouting(const std::string &pattern, int32_t rankSize,
    int64_t s, int64_t k, int64_t expertCount, uint32_t seed);

} // namespace TileXRMoonEpTest

#endif // TILEXR_TEST_MOONEP_PLANNER_REFERENCE_H
