#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

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

bool HasAllowedExtension(const std::string &path) {
    const std::vector<std::string> extensions = {
        ".h",
        ".hpp",
        ".cpp",
        ".cc",
        ".cce",
    };
    for (size_t i = 0; i < extensions.size(); ++i) {
        const std::string &extension = extensions[i];
        if (path.size() >= extension.size() &&
            path.compare(path.size() - extension.size(), extension.size(), extension) == 0) {
            return true;
        }
    }
    return false;
}

void AppendProductionSourceFiles(const std::string &directory, std::vector<std::string> *files) {
    DIR *dir = opendir(directory.c_str());
    if (dir == nullptr) {
        std::cerr << "failed to open directory " << directory << "\n";
        ++g_failures;
        return;
    }

    while (true) {
        struct dirent *entry = readdir(dir);
        if (entry == nullptr) {
            break;
        }
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        const std::string path = directory + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) != 0) {
            std::cerr << "failed to stat " << path << "\n";
            ++g_failures;
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            AppendProductionSourceFiles(path, files);
        } else if (S_ISREG(st.st_mode) && HasAllowedExtension(path)) {
            files->push_back(path);
        }
    }
    closedir(dir);
}

}  // namespace

int main() {
    std::vector<std::string> production_files;
    AppendProductionSourceFiles(SourcePath("src/collectives"), &production_files);
    AppendProductionSourceFiles(SourcePath("src/ep"), &production_files);

    if (production_files.empty()) {
        std::cerr << "no production source files scanned\n";
        ++g_failures;
    }

    const std::vector<std::string> rejected_tokens = {
        "TILEXR_BUILD_CHECKER",
        "TILEXR_CHECKER",
        "checker::",
        "tools/checker",
    };

    for (size_t file_index = 0; file_index < production_files.size(); ++file_index) {
        const std::string path = production_files[file_index];
        const std::string text = ReadFile(path);
        for (size_t token_index = 0; token_index < rejected_tokens.size(); ++token_index) {
            ExpectNotContains(text, rejected_tokens[token_index], path);
        }
    }

    return g_failures == 0 ? 0 : 1;
}
