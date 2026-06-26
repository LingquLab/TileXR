#include <type_traits>

#include "tilexr/checker/allgather_hdb_trace_shim.h"

int main() {
    static_assert(STAGE_NUM == 4, "hierarchy double ring stage count changed");
    static_assert(RING_NUM == 2, "hierarchy double ring ring count changed");
    static_assert(static_cast<int>(STAGE::HCCS_RING) == 0,
                  "HCCS_RING stage value changed");
    static_assert(static_cast<int>(STAGE::HCCS_TO_OUT) == 1,
                  "HCCS_TO_OUT stage value changed");
    static_assert(static_cast<int>(STAGE::HCCS_TO_SIO) == 2,
                  "HCCS_TO_SIO stage value changed");
    static_assert(static_cast<int>(STAGE::SIO_TO_OUT) == 3,
                  "SIO_TO_OUT stage value changed");
    static_assert(std::is_class<AllGatherHierarchyDoubleRing<int> >::value,
                  "AllGatherHierarchyDoubleRing<int> must remain instantiable");
    const tilexr::checker::TraceAdapterMetadata &metadata =
        tilexr::checker::trace_adapter_allgather_hdb::Metadata();
    static_assert(tilexr::checker::trace_adapter_allgather_hdb::kKnownHookCount >= 3,
                  "allgather hdb trace shim must publish observed hook metadata");
    const char *reason = nullptr;
    if (!tilexr::checker::trace_adapter_allgather_hdb::Audit(&reason)) {
        return 1;
    }
    if (reason != nullptr) {
        return 2;
    }
    if (metadata.adapter_name == nullptr || metadata.source_file == nullptr ||
        metadata.target_header == nullptr || metadata.include_guard == nullptr) {
        return 3;
    }
    if (metadata.known_hook_count <
        tilexr::checker::trace_adapter_allgather_hdb::kKnownHookCount) {
        return 4;
    }
    if (metadata.manual_review_candidate_count !=
        tilexr::checker::trace_adapter_allgather_hdb::kManualReviewCandidateCount) {
        return 5;
    }
    return 0;
}
