#ifndef TILEXR_CHECKER_ALLREDUCE_BIG_DATA_TRACE_SHIM_H
#define TILEXR_CHECKER_ALLREDUCE_BIG_DATA_TRACE_SHIM_H

#define TILEXR_CHECKER_TRACE_SOURCE_FILE "src/collectives/kernels/allreduce_big_data.h"
#define TILEXR_CHECKER_TRACE_TARGET_HEADER "allreduce_big_data.h"
#include "tilexr/checker/collective_trace_adapter.h"

namespace tilexr {
namespace checker {
namespace trace_adapter_allreduce_big_data {

static const int kKnownHookCount = 3;
static const int kManualReviewCandidateCount = 0;
static const int kKnownHook0Lines[] = {110, 156, 161, 191, 207, 233, 235, 237};
static const int kKnownHook1Lines[] = {138};
static const int kKnownHook2Lines[] = {140};
static const TraceAdapterHookObservation kKnownHooks[] = {
    {"CpGM2GMPingPong", kKnownHook0Lines, 8},
    {"SetSyncFlag", kKnownHook1Lines, 1},
    {"WaitSyncFlag", kKnownHook2Lines, 1}
};

inline const TraceAdapterMetadata &Metadata()
{
    static const TraceAdapterMetadata metadata = {
        "allreduce_big_data",
        "src/collectives/kernels/allreduce_big_data.h",
        "allreduce_big_data.h",
        "TILEXR_CHECKER_ALLREDUCE_BIG_DATA_TRACE_SHIM_H",
        kKnownHooks, 3,
        nullptr, 0
    };
    return metadata;
}

inline bool Audit(const char **reason)
{
    return AuditTraceAdapterMetadata(Metadata(), reason);
}

}  // namespace trace_adapter_allreduce_big_data
}  // namespace checker
}  // namespace tilexr

#undef TILEXR_CHECKER_TRACE_TARGET_HEADER
#undef TILEXR_CHECKER_TRACE_SOURCE_FILE

#endif  // TILEXR_CHECKER_ALLREDUCE_BIG_DATA_TRACE_SHIM_H
