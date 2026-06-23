#ifndef TILEXR_CHECKER_CLI_H
#define TILEXR_CHECKER_CLI_H

#include <iosfwd>
#include <string>

#include "tilexr/checker/case.h"
#include "tilexr/checker/event.h"
#include "tilexr/checker/executor.h"
#include "tilexr/checker/trace_adapter_generator.h"

namespace tilexr {
namespace checker {

struct CliOptions {
    CheckerCase test_case;
    std::string output_dir;
    bool inject_read_before_copy = false;
    bool inject_peer_user_read = false;
    bool inject_pipe_wait = false;
    bool inject_ep_window_read = false;
    bool scaffold_trace_bundle = false;
    bool generate_trace_adapter = false;
    bool verify_trace_bundle = false;
    bool verify_installed_trace = false;
    bool verify_all_installed_traces = false;
    bool list_installed_traces = false;
    bool list_capabilities = false;
    bool analyze_trace_source = false;
    bool validate_event_trace = false;
    TraceAdapterSpec trace_adapter_spec;
    std::string repo_root;
    std::string event_trace_input;
    std::string verification_report_output;
    std::string capability_report_output;
    std::string trace_analysis_output;
    std::string event_trace_template_output;
    std::string trace_adapter_output;
    std::string trace_adapter_manifest_output;
    TraceRunnerSkeletonSpec trace_runner_spec;
    std::string trace_runner_output;
    TraceHeaderProbeSpec trace_probe_spec;
    std::string trace_probe_output;
    std::string trace_onboarding_output;
    bool show_help = false;
};

struct ReportPaths {
    std::string summary_txt;
    std::string findings_json;
    std::string events_jsonl;
    std::string checker_report_json;
};

CheckerStatus ParseCliArgs(int argc, const char *const *argv, CliOptions *options);

CheckerStatus WriteReportFiles(const std::string &output_dir,
                               const CheckerCase &test_case,
                               const RunResult &result,
                               const EventLog &events,
                               ReportPaths *paths);
CheckerStatus WriteReportFiles(const std::string &output_dir,
                               const CheckerCase &test_case,
                               const RunResult &result,
                               const EventLog &events,
                               const std::string &source_root,
                               ReportPaths *paths);
CheckerStatus WriteReportFiles(const std::string &output_dir,
                               const CheckerCase &test_case,
                               const RunResult &result,
                               const EventLog &events,
                               const std::string &source_root,
                               const std::string &mode,
                               const std::string &event_trace_input,
                               ReportPaths *paths);

int RunCheckerCli(const CliOptions &options, std::ostream *stdout_stream,
                  std::ostream *stderr_stream);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_CLI_H
