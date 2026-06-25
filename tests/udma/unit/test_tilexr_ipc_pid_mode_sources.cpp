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

#define CHECK_NOT_CONTAINS(text, needle) \
    do { \
        if ((text).find(needle) != std::string::npos) { \
            std::cerr << "CHECK_NOT_CONTAINS failed at line " << __LINE__ << ": " << needle << std::endl; \
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
    const std::string cmake = ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/tests/udma/CMakeLists.txt");
    CHECK_CONTAINS(comm, "TILEXR_IPC_PID_MODE");
    CHECK_CONTAINS(comm, "physicalInfo_.chipName < ChipName::CHIP_950");
    CHECK_CONTAINS(comm, "rtSetIpcMemPid");
    CHECK_CONTAINS(comm, "rtSetIpcMemorySuperPodPid");
    CHECK_CONTAINS(comm, "fallback to rtSetIpcMemPid");
    CHECK_CONTAINS(comm, "OpenIpcMem failed after sdid setup, retry with pid setup");
    CHECK_CONTAINS(comm, "\"pid_retry\"");
    CHECK_CONTAINS(comm, "SetMemoryName(retryName)");
    CHECK_CONTAINS(comm, "GetName(retryName, names)");
    CHECK_NOT_CONTAINS(comm, "do not support pcie > 2 rank");
    CHECK_CONTAINS(cmake, "TILEXR_UDMA_FORCE_ENABLE");

    const std::string transport =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/comm/udma/tilexr_udma_transport.cpp");
    const std::string udma = ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/include/tilexr_udma.h");
    CHECK_CONTAINS(transport, "RaCtxRmemImport");
    CHECK_CONTAINS(udma, "UDMAQuietStatus");

    if (g_failures != 0) {
        std::cerr << g_failures << " IPC PID mode source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR IPC PID mode source checks passed" << std::endl;
    return 0;
}
