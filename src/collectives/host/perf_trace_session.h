#ifndef TILEXR_COLLECTIVES_HOST_PERF_TRACE_SESSION_H
#define TILEXR_COLLECTIVES_HOST_PERF_TRACE_SESSION_H

#include <cstddef>
#include <string>
#include <vector>

#include "acl/acl_base.h"
#include "comm_args.h"
#include "tilexr_collectives_perf.h"
#include "tilexr_perf_trace.h"
#include "tilexr_types.h"

namespace TileXRCollectives {
namespace Host {

struct PerfTraceSession {
    TileXRCollectivePerfConfig config {};
    std::string outputDir;
    std::string aiCommand;
    std::vector<TileXR::TileXRPerfCoreStageStats> hostStats;
    TileXR::TileXRPerfTraceHeader header {};
    void *deviceBuffer = nullptr;
    size_t deviceBufferBytes = 0;
};

PerfTraceSession *GetActivePerfTraceSession();
void SetActivePerfTraceSessionForHost(PerfTraceSession *session);
int PreparePerfTraceLaunch(PerfTraceSession *session, const TileXR::CommArgs &commArgs,
                           TileXR::TileXRType opType, TileXR::TileXRDataType dataType,
                           uint32_t blockDim, int64_t count, aclrtStream stream,
                           const void **deviceTrace);

} // namespace Host
} // namespace TileXRCollectives

#endif // TILEXR_COLLECTIVES_HOST_PERF_TRACE_SESSION_H
