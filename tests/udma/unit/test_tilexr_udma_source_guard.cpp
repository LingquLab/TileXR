#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

void CheckContains(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected text not found in " << path << ": " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "unexpected text in " << path << ": " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void CheckNoNeedles(const std::string& path, const std::vector<std::string>& needles)
{
    const auto text = ReadFile(path);
    for (const auto& needle : needles) {
        CheckNotContains(path, text, needle);
    }
}

void TestTileXRCommUsesUDMAContextBoundary()
{
    const std::string headerPath = "src/comm/tilexr_comm.h";
    const auto header = ReadFile(headerPath);

    CheckContains(headerPath, header, "class TileXRUDMAContext;");
    CheckContains(headerPath, header, "std::unique_ptr<TileXRUDMAContext> udmaContext_");

    const std::vector<std::string> forbiddenHeaderText = {
        "class TileXRUDMATransport;",
        "GM_ADDR udmaInfoDev_",
        "GM_ADDR udmaRegistryDev_",
        "GM_ADDR udmaRegisteredPtr_",
        "TileXRUDMARegistry udmaRegistry_",
        "std::unique_ptr<TileXRUDMATransport>",
    };
    for (const auto& needle : forbiddenHeaderText) {
        CheckNotContains(headerPath, header, needle);
    }
}

void TestUDMATransportStaysBehindContext()
{
    CheckNoNeedles("src/comm/tilexr_comm.cpp", {
        "#include \"udma/tilexr_udma_transport.h\"",
        "new (nothrow) TileXRUDMATransport",
        "udmaTransport_->",
        "udmaTransport_.",
    });

    CheckContains("src/comm/udma/tilexr_udma_context.h",
                  ReadFile("src/comm/udma/tilexr_udma_context.h"),
                  "class TileXRUDMAContext");
    CheckContains("src/comm/udma/tilexr_udma_context.cpp",
                  ReadFile("src/comm/udma/tilexr_udma_context.cpp"),
                  "#include \"udma/tilexr_udma_transport.h\"");
}

void TestUDMAContextShutdownIsLocalOnly()
{
    const std::string path = "src/comm/udma/tilexr_udma_context.cpp";
    const auto text = ReadFile(path);
    const auto shutdownPos = text.find("void TileXRUDMAContext::Shutdown()");
    const auto registerPos = text.find("int TileXRUDMAContext::RegisterMemory", shutdownPos);
    if (shutdownPos == std::string::npos || registerPos == std::string::npos) {
        std::cerr << "failed to locate TileXRUDMAContext::Shutdown body" << std::endl;
        ++g_failures;
        return;
    }

    const auto shutdownBody = text.substr(shutdownPos, registerPos - shutdownPos);
    CheckNotContains(path, shutdownBody, "UnregisterMemory(");
    CheckContains(path, shutdownBody, "transport_->Shutdown();");
}

void TestUDMAReviewFeedbackGuards()
{
    const std::string contextPath = "src/comm/udma/tilexr_udma_context.cpp";
    const auto context = ReadFile(contextPath);
    CheckContains(contextPath, context,
                  "TILEXR_LOG(ERROR) << \"TileXRUDMARegister called while UDMA is unavailable\"");
    CheckContains(contextPath, context,
                  "TILEXR_LOG(ERROR) << \"TileXRUDMARegister is not supported in InitThread mode\"");
    CheckContains(contextPath, context, "RollbackTransportRegistration(");
    CheckContains(contextPath, context,
                  "TILEXR_LOG(ERROR) << \"TileXRUDMAUnregister failed to clear comm args: \"");
    CheckContains(contextPath, context,
                  "TILEXR_LOG(ERROR) << \"TileXR UDMA memory unregistration failed: \"");
    CheckContains(contextPath, context,
                  "TILEXR_LOG(ERROR) << \"TileXR UDMA restore registration called with invalid state\"");
    CheckContains(contextPath, context,
                  "TILEXR_LOG(ERROR) << \"TileXR UDMA failed to restore previous registration: \"");
    CheckContains(contextPath, context, "TILEXR_LOG(ERROR) << \"Free UDMA registry failed: \"");

    const std::string commPath = "src/comm/tilexr_comm.cpp";
    const auto comm = ReadFile(commPath);
    CheckContains(commPath, comm, "TILEXR_LOG(ERROR) << \"TileXR UDMA update comm args failed: \"");
    CheckContains(commPath, comm,
                  "TILEXR_LOG(ERROR) << \"ApplyUDMACommArgsStateCallback missing user data\"");
    CheckContains(commPath, comm,
                  "TILEXR_LOG(ERROR) << \"TileXRUDMARegister called while UDMA is unavailable\"");
}

void TestPublicHeadersDoNotExposeUDMAContext()
{
    const std::vector<std::string> publicHeaders = {
        "src/include/tilexr_api.h",
        "src/include/tilexr_udma.h",
        "src/include/tilexr_udma_reg.h",
        "src/include/tilexr_udma_types.h",
        "src/include/comm_args.h",
    };
    for (const auto& path : publicHeaders) {
        const auto text = ReadFile(path);
        CheckNotContains(path, text, "tilexr_udma_context.h");
        CheckNotContains(path, text, "TileXRUDMAContext");
    }
}

void TestCommSourcesDoNotUseShmem()
{
    const std::vector<std::string> paths = {
        "src/comm/CMakeLists.txt",
        "src/comm/tilexr_comm.cpp",
        "src/comm/comm_wrap.cpp",
        "src/comm/tilexr_comm.h",
        "src/comm/udma/tilexr_udma_context.cpp",
        "src/comm/udma/tilexr_udma_context.h",
        "src/comm/udma/tilexr_udma_transport.cpp",
        "src/comm/udma/tilexr_udma_transport.h",
    };
    const std::vector<std::string> forbidden = {
        "shmem",
        "shmem.h",
        "libshmem",
        "aclshmem",
        "ACLSHMEM",
    };
    for (const auto& path : paths) {
        CheckNoNeedles(path, forbidden);
    }
}

} // namespace

int main()
{
    TestTileXRCommUsesUDMAContextBoundary();
    TestUDMATransportStaysBehindContext();
    TestUDMAContextShutdownIsLocalOnly();
    TestUDMAReviewFeedbackGuards();
    TestPublicHeadersDoNotExposeUDMAContext();
    TestCommSourcesDoNotUseShmem();
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA source guard checks passed" << std::endl;
    return 0;
}
