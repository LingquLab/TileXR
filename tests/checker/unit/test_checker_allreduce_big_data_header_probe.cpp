#include <type_traits>

#include "tilexr/checker/allreduce_big_data_trace_shim.h"

int main() {
    static_assert(std::is_class<AllReduceBigData<int> >::value,
                  "AllReduceBigData<int> must remain instantiable");
    static_assert(std::is_class<AllReduceBigData<int, int> >::value,
                  "AllReduceBigData<int, int> must remain instantiable");
    const tilexr::checker::TraceAdapterMetadata &metadata =
        tilexr::checker::trace_adapter_allreduce_big_data::Metadata();
    static_assert(
        tilexr::checker::trace_adapter_allreduce_big_data::kKnownHookCount >= 3,
        "allreduce_big_data trace shim must publish observed hook metadata");
    const char *reason = nullptr;
    if (!tilexr::checker::trace_adapter_allreduce_big_data::Audit(&reason)) {
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
        tilexr::checker::trace_adapter_allreduce_big_data::kKnownHookCount) {
        return 4;
    }
    if (metadata.manual_review_candidate_count <
        tilexr::checker::trace_adapter_allreduce_big_data::kManualReviewCandidateCount) {
        return 5;
    }
    return 0;
}
