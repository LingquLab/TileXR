#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {
int g_failures = 0;

std::string SourcePath(const std::string &relative_path) {
    return std::string(TILEXR_SOURCE_ROOT) + "/" + relative_path;
}

std::string ReadFile(const std::string &path) {
    std::ifstream input(path.c_str());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

void ExpectContains(const std::string &text, const std::string &needle,
                    const std::string &path) {
    if (text.find(needle) == std::string::npos) {
        std::cerr << path << " missing " << needle << "\n";
        ++g_failures;
    }
}
}  // namespace

int main() {
    const std::string root_cmake = ReadFile(SourcePath("CMakeLists.txt"));
    ExpectContains(root_cmake, "TILEXR_BUILD_CHECKER", "CMakeLists.txt");
    const std::string checker_cmake = ReadFile(SourcePath("tools/checker/CMakeLists.txt"));
    ExpectContains(checker_cmake, "tilexr-checker-core", "tools/checker/CMakeLists.txt");
    return g_failures == 0 ? 0 : 1;
}
