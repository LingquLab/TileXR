#include <cstdint>
#include <type_traits>

#include "tilexr_moonep_combine_v2.h"

using CombineWorkspaceQueryV2 = int (*)(int64_t, int64_t, int64_t,
    int64_t, uint32_t, uint64_t *, uint64_t *, uint64_t *, uint64_t *);
using CombineV2 = int (*)(void *, const int32_t *, TileXRCommPtr, int64_t,
    int64_t, int64_t, int64_t, uint32_t, uint64_t *, uint32_t, aclrtStream);
using CombineV1 = int (*)(const TileXRMoonEpCombineArgsV1 *, aclrtStream);

static_assert(std::is_same<decltype(&TileXRMoonEpCombineGetWorkspaceSizeV2),
    CombineWorkspaceQueryV2>::value, "Combine V2 workspace ABI changed");
static_assert(std::is_same<decltype(&TileXRMoonEpCombineV2),
    CombineV2>::value, "Combine V2 launch ABI changed");
static_assert(std::is_same<decltype(&TileXRMoonEpCombineV1),
    CombineV1>::value, "Combine V1 ABI changed");

int main()
{
    return 0;
}
