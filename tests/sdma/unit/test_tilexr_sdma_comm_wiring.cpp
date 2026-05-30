#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string& path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string& path)
{
    std::ifstream input(RepoPath(path).c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << RepoPath(path) << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string& path, const std::string& needle)
{
    const auto text = ReadFile(path);
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected text not found in " << path << ": " << needle << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    CheckContains("src/comm/tilexr_comm.h", "std::unique_ptr<TileXRSDMATransport> sdmaTransport_");
    CheckContains("src/comm/tilexr_comm.cpp", "int TileXRComm::InitSDMA()");
    CheckContains("src/comm/tilexr_comm.cpp", "commArgs_.sdmaWorkspacePtr = sdmaWorkspaceDev_");
    CheckContains("src/comm/tilexr_comm.cpp", "commArgs_.extraFlag |= ExtraFlag::SDMA");
    CheckContains("src/comm/comm_wrap.cpp", "*available = c->IsSDMAAvailable()");
    CheckContains("src/comm/comm_wrap.cpp", "*workspace = c->GetSDMAWorkspacePtr()");

    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA comm wiring checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA comm wiring checks passed" << std::endl;
    return 0;
}
