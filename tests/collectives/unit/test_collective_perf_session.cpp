#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "perf_trace_session.h"
#include "tilexr_collectives_perf.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

template <typename T>
void CheckEq(const char *label, T actual, T expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckTrue(const char *label, bool condition)
{
    if (!condition) {
        std::cerr << label << " failed" << std::endl;
        ++g_failures;
    }
}

std::string ReadFile(const std::string &path)
{
    std::ifstream input(path.c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << path << std::endl;
        ++g_failures;
        return {};
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const char *label, const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << label << " missing " << needle << std::endl;
        ++g_failures;
    }
}

void TestCollectivePerfSessionLifecycle()
{
    const std::string outputDir = "/tmp/tilexr_perf_session_test";
    std::remove((outputDir + "/trace.json").c_str());
    std::remove((outputDir + "/summary.csv").c_str());
    std::remove((outputDir + "/analysis.md").c_str());
    std::remove((outputDir + "/report.html").c_str());
    std::remove((outputDir + "/ai_prompt.md").c_str());

    TileXRCollectivePerfSession session = nullptr;
    CheckEq("create rejects null config", TileXRCollectivePerfSessionCreate(nullptr, &session),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    std::string aiCommand = "tilexr-analyze --dry-run";
    TileXRCollectivePerfConfig config {};
    config.enabled = 1;
    config.outputDir = outputDir.c_str();
    config.emitAiPrompt = 1;
    config.sampleEveryN = 1;
    config.aiCommand = aiCommand.c_str();

    CheckEq("create succeeds", TileXRCollectivePerfSessionCreate(&config, &session), TileXR::TILEXR_SUCCESS);
    CheckTrue("session created", session != nullptr);

    TileXRCollectives::Host::PerfTraceSession *impl =
        static_cast<TileXRCollectives::Host::PerfTraceSession *>(session);
    CheckTrue("aiCommand copied to session", impl->config.aiCommand != nullptr);
    CheckTrue("aiCommand uses owned storage", impl->config.aiCommand != aiCommand.c_str());
    aiCommand = "mutated caller command";
    CheckEq("owned aiCommand content stable", std::string(impl->config.aiCommand),
        std::string("tilexr-analyze --dry-run"));

    CheckEq("set active session succeeds", TileXRCollectivePerfSetActiveSession(session), TileXR::TILEXR_SUCCESS);
    CheckTrue("active getter returns session", TileXRCollectives::Host::GetActivePerfTraceSession() == impl);
    CheckEq("write report succeeds", TileXRCollectivePerfWriteReport(session), TileXR::TILEXR_SUCCESS);
    CheckContains("report.html", ReadFile(outputDir + "/report.html"), "Bottleneck First");
    CheckContains("trace.json", ReadFile(outputDir + "/trace.json"), "tilexr_perf_trace_report.v1");
    CheckEq("destroy succeeds", TileXRCollectivePerfSessionDestroy(session), TileXR::TILEXR_SUCCESS);
    CheckTrue("destroy clears active session", TileXRCollectives::Host::GetActivePerfTraceSession() == nullptr);
}

} // namespace

int main()
{
    TestCollectivePerfSessionLifecycle();
    return g_failures == 0 ? 0 : 1;
}
