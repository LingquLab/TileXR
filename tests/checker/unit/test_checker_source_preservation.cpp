#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

std::string SourcePath(const std::string &relative_path) {
    return std::string(TILEXR_SOURCE_ROOT) + "/" + relative_path;
}

std::string ReadFile(const std::string &path) {
    std::ifstream input(path.c_str());
    if (!input) {
        std::cerr << "failed to open " << path << "\n";
        ++g_failures;
        return std::string();
    }
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

void ExpectNotContains(const std::string &text, const std::string &needle,
                       const std::string &path) {
    if (text.find(needle) != std::string::npos) {
        std::cerr << path << " unexpectedly contains " << needle << "\n";
        ++g_failures;
    }
}

}  // namespace

int main() {
    const std::vector<std::string> production_files = {
        "src/collectives/kernels/allgather.h",
        "src/collectives/kernels/allreduce_one_shot.h",
        "src/collectives/kernels/allreduce_two_shot.h",
        "src/collectives/host/collective_launcher.cpp",
        "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp",
        "src/ep/host/tilexr_ep_dispatch.cpp",
        "src/ep/host/ep_dispatch_host.cpp",
    };
    const std::vector<std::string> rejected_tokens = {
        "TILEXR_BUILD_CHECKER",
        "TILEXR_CHECKER",
        "checker::",
        "tools/checker",
    };

    for (size_t file_index = 0; file_index < production_files.size(); ++file_index) {
        const std::string path = SourcePath(production_files[file_index]);
        const std::string text = ReadFile(path);
        for (size_t token_index = 0; token_index < rejected_tokens.size(); ++token_index) {
            ExpectNotContains(text, rejected_tokens[token_index], path);
        }
    }

    return g_failures == 0 ? 0 : 1;
}
