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
    bool ownsDeviceBuffer = false;
    bool deviceTraceReady = false;
    std::vector<void *> retiredDeviceBuffers;
};

struct PerfTraceRuntimeHooks {
    aclError (*mallocDevice)(void **ptr, size_t bytes);
    aclError (*freeDevice)(void *ptr);
    aclError (*copyHostToDeviceAsync)(void *dst, size_t dstBytes, const void *src, size_t bytes,
                                      aclrtStream stream);
    aclError (*memsetDeviceAsync)(void *dst, size_t dstBytes, int value, size_t bytes, aclrtStream stream);
    aclError (*copyDeviceToHost)(void *dst, size_t dstBytes, const void *src, size_t bytes);
};

PerfTraceSession *GetActivePerfTraceSession();
void SetActivePerfTraceSessionForHost(PerfTraceSession *session);
void SetPerfTraceRuntimeHooksForTest(const PerfTraceRuntimeHooks *hooks);
int PreparePerfTraceLaunch(PerfTraceSession *session, const TileXR::CommArgs &commArgs,
                           TileXR::TileXRType opType, TileXR::TileXRDataType dataType,
                           uint32_t blockDim, int64_t count, aclrtStream stream,
                           const void **deviceTrace);

} // namespace Host
} // namespace TileXRCollectives

#endif // TILEXR_COLLECTIVES_HOST_PERF_TRACE_SESSION_H
