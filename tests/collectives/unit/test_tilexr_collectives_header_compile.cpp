#include "tilexr_collectives.h"

namespace {

using AllGatherFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXRCommPtr, aclrtStream);
using AllToAllFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXRCommPtr, aclrtStream);

} // namespace

int main()
{
    AllGatherFn allGather = &TileXRAllGather;
    AllToAllFn allToAll = &TileXRAllToAll;
    return (allGather != nullptr && allToAll != nullptr) ? 0 : 1;
}
