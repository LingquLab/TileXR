#include <iostream>

#include "tilexr/checker/cli.h"

int main(int argc, const char *const *argv) {
    tilexr::checker::CliOptions options;
    tilexr::checker::CheckerStatus status =
        tilexr::checker::ParseCliArgs(argc, argv, &options);
    if (!status.ok()) {
        std::cerr << status.message << "\n";
        return status.code == tilexr::checker::CheckerStatusCode::kUnsupported ? 2 : 3;
    }
    return tilexr::checker::RunCheckerCli(options, &std::cout, &std::cerr);
}
