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
    const std::string comm = ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/comm/tilexr_comm.cpp");
    CHECK_CONTAINS(comm, "TILEXR_ENABLE_UDMA");
    CHECK_CONTAINS(comm, "TileXR UDMA disabled by environment");
    CHECK_CONTAINS(comm, "IsEnvEnabled(\"TILEXR_ENABLE_UDMA\", true)");
    CHECK_CONTAINS(comm, "TILEXR_ENABLE_IPC");
    CHECK_CONTAINS(comm, "TileXR IPC memory disabled by environment");
    CHECK_CONTAINS(comm, "IsEnvEnabled(\"TILEXR_ENABLE_IPC\", true)");
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA env source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA env source checks passed" << std::endl;
    return 0;
}
