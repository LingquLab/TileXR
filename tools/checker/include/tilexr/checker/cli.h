#ifndef TILEXR_CHECKER_CLI_H
#define TILEXR_CHECKER_CLI_H

#include <iosfwd>
#include <string>

#include "tilexr/checker/case.h"
#include "tilexr/checker/event.h"
#include "tilexr/checker/executor.h"

namespace tilexr {
namespace checker {

struct CliOptions {
    CheckerCase test_case;
    std::string output_dir;
    bool inject_read_before_copy = false;
    bool inject_peer_user_read = false;
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

int RunCheckerCli(const CliOptions &options, std::ostream *stdout_stream,
                  std::ostream *stderr_stream);

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_CLI_H
