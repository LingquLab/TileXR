#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef TILEXR_SOURCE_ROOT
#define TILEXR_SOURCE_ROOT "."
#endif

namespace {

int g_failures = 0;

#define CHECK_CONTAINS(text, needle) \
    do { \
        if ((text).find(needle) == std::string::npos) { \
            std::cerr << "CHECK_CONTAINS failed at line " << __LINE__ << ": " << needle << std::endl; \
            ++g_failures; \
        } \
    } while (0)

std::string ReadFile(const std::string& path)
{
    std::ifstream in(path.c_str());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

} // namespace

int main()
{
    const std::string internal = ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/comm/tilexr_internal.cpp");
    CHECK_CONTAINS(internal, "\"Ascend950DT_9592\"");
    CHECK_CONTAINS(internal, "\"Ascend950PR_9599\"");
    CHECK_CONTAINS(internal, "ChipName::CHIP_950");

    if (g_failures != 0) {
        std::cerr << g_failures << " chip map source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR chip map source checks passed" << std::endl;
    return 0;
}
