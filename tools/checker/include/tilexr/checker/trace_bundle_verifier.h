#ifndef TILEXR_CHECKER_TRACE_BUNDLE_VERIFIER_H
#define TILEXR_CHECKER_TRACE_BUNDLE_VERIFIER_H

#include <cstddef>
#include <string>
#include <vector>

#include "tilexr/checker/status.h"

namespace tilexr {
namespace checker {

struct TraceBundleVerificationResult {
    bool complete = false;
    std::vector<std::string> missing_artifacts;
    std::vector<std::string> missing_manifest_sections;
    std::vector<std::string> unchecked_items;
    bool probe_compile_attempted = false;
    bool probe_compile_passed = false;
    bool runtime_smoke_passed = false;
    size_t runtime_smoke_event_count = 0;
    size_t runtime_smoke_case_count = 0;
    std::string summary;
};

CheckerStatus VerifyTraceBundle(const std::string &bundle_dir,
                                const std::string &adapter_name,
                                TraceBundleVerificationResult *result);
CheckerStatus VerifyTraceBundle(const std::string &bundle_dir,
                                const std::string &adapter_name,
                                const std::string &repo_root,
                                TraceBundleVerificationResult *result);
CheckerStatus VerifyInstalledTraceAlgorithm(const std::string &repo_root,
                                            const std::string &adapter_name,
                                            TraceBundleVerificationResult *result);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_TRACE_BUNDLE_VERIFIER_H
