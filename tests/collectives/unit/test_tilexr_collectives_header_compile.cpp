#include "tilexr_collectives.h"
#include "tilexr_collectives_perf.h"
#include "tilexr_perf_trace.h"

namespace {

using AllGatherFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXRCommPtr, aclrtStream);
using AllToAllFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXRCommPtr, aclrtStream);

} // namespace

int main()
{
    AllGatherFn allGather = &TileXRAllGather;
    AllToAllFn allToAll = &TileXRAllToAll;

    TileXR::TileXRPerfTraceHeader header {};
    TileXRCollectivePerfConfig config {};
    config.enabled = 1;

    return (allGather != nullptr && allToAll != nullptr &&
            header.magic == TileXR::TILEXR_PERF_TRACE_MAGIC && config.enabled == 1) ? 0 : 1;
}
