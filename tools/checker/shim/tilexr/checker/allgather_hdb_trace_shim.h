#ifndef TILEXR_CHECKER_ALLGATHER_HDB_TRACE_SHIM_H
#define TILEXR_CHECKER_ALLGATHER_HDB_TRACE_SHIM_H

#define TILEXR_CHECKER_TRACE_SOURCE_FILE "src/collectives/kernels/91093/allgather_hierarchy_double_ring.h"
#define TILEXR_CHECKER_TRACE_TARGET_HEADER "91093/allgather_hierarchy_double_ring.h"
#include "tilexr/checker/collective_trace_adapter.h"

namespace tilexr {
namespace checker {
namespace trace_adapter_allgather_hdb {

static const int kKnownHookCount = 3;
static const int kManualReviewCandidateCount = 0;
static const int kKnownHook0Lines[] = {126, 160, 191, 218};
static const int kKnownHook1Lines[] = {139, 174, 207, 234};
static const int kKnownHook2Lines[] = {141, 145, 147, 177, 208, 237};
static const TraceAdapterHookObservation kKnownHooks[] = {
    {"WaitSyncFlag", kKnownHook0Lines, 4},
    {"CpGM2GMPingPong", kKnownHook1Lines, 4},
    {"SetSyncFlag", kKnownHook2Lines, 6}
};

inline const TraceAdapterMetadata &Metadata()
{
    static const TraceAdapterMetadata metadata = {
        "allgather_hierarchy_double_ring",
        "src/collectives/kernels/91093/allgather_hierarchy_double_ring.h",
        "91093/allgather_hierarchy_double_ring.h",
        "TILEXR_CHECKER_ALLGATHER_HDB_TRACE_SHIM_H",
        kKnownHooks, 3,
        nullptr, 0
    };
    return metadata;
}

inline bool Audit(const char **reason)
{
    return AuditTraceAdapterMetadata(Metadata(), reason);
}

}  // namespace trace_adapter_allgather_hdb
}  // namespace checker
}  // namespace tilexr

#undef TILEXR_CHECKER_TRACE_TARGET_HEADER
#undef TILEXR_CHECKER_TRACE_SOURCE_FILE

#endif  // TILEXR_CHECKER_ALLGATHER_HDB_TRACE_SHIM_H
