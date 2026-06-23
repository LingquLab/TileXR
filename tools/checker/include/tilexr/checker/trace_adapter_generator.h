#ifndef TILEXR_CHECKER_TRACE_ADAPTER_GENERATOR_H
#define TILEXR_CHECKER_TRACE_ADAPTER_GENERATOR_H

#include <string>
#include <vector>

#include "tilexr/checker/status.h"

namespace tilexr {
namespace checker {

struct TraceAdapterSpec {
    std::string adapter_name;
    std::string source_file;
    std::string source_scan_file;
    std::string target_header;
    bool strict_source_observation = false;
};

struct TraceRunnerSkeletonSpec {
    std::string function_name;
    std::string shim_include;
    std::string materializer_name;
};

struct TraceHeaderProbeSpec {
    std::string adapter_name;
    std::string shim_include;
    std::string expected_source_file;
    std::string expected_target_header;
};

struct TraceSourceObservationItem {
    std::string symbol;
    std::string event;
    std::vector<int> lines;
    std::vector<std::string> snippets;
    std::string note;
    std::string macro_name;
    bool manual_review_required = false;
};

struct TraceSourceAnalysis {
    std::string adapter_name;
    std::string source_file;
    std::string source_scan_file;
    std::string target_header;
    std::string shim_header;
    bool available = false;
    bool manual_review_required = false;
    std::vector<TraceSourceObservationItem> auto_hook_candidates;
    std::vector<TraceSourceObservationItem> manual_review_candidates;
};

CheckerStatus NormalizeTraceAdapterSpec(TraceAdapterSpec *spec);
CheckerStatus AnalyzeTraceSource(const TraceAdapterSpec &spec,
                                 const std::string &shim_header,
                                 TraceSourceAnalysis *analysis);
CheckerStatus RenderTraceSourceAnalysisJson(const TraceSourceAnalysis &analysis,
                                            std::string *output);
CheckerStatus RenderTraceSourceAnalysisText(const TraceSourceAnalysis &analysis,
                                            std::string *output);
CheckerStatus RenderTraceAdapter(const TraceAdapterSpec &spec, std::string *output);
CheckerStatus WriteTraceAdapterFile(const TraceAdapterSpec &spec, const std::string &path);
CheckerStatus RenderTraceAdapterManifest(const TraceAdapterSpec &spec,
                                         const std::string &shim_header,
                                         std::string *output);
CheckerStatus WriteTraceAdapterManifestFile(const TraceAdapterSpec &spec,
                                            const std::string &shim_header,
                                            const std::string &path);
CheckerStatus RenderTraceRunnerSkeleton(const TraceRunnerSkeletonSpec &spec,
                                        std::string *output);
CheckerStatus WriteTraceRunnerSkeletonFile(const TraceRunnerSkeletonSpec &spec,
                                           const std::string &path);
CheckerStatus RenderTraceHeaderProbeSkeleton(const TraceHeaderProbeSpec &spec,
                                             std::string *output);
CheckerStatus WriteTraceHeaderProbeSkeletonFile(const TraceHeaderProbeSpec &spec,
                                                const std::string &path);
CheckerStatus RenderTraceOnboardingPlan(const TraceAdapterSpec &adapter_spec,
                                        const TraceRunnerSkeletonSpec &runner_spec,
                                        const std::string &shim_header,
                                        const std::string &manifest_path,
                                        const std::string &runner_path,
                                        std::string *output);
CheckerStatus WriteTraceOnboardingPlanFile(const TraceAdapterSpec &adapter_spec,
                                           const TraceRunnerSkeletonSpec &runner_spec,
                                           const std::string &shim_header,
                                           const std::string &manifest_path,
                                           const std::string &runner_path,
                                           const std::string &path);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_TRACE_ADAPTER_GENERATOR_H
