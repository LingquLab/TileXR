#include "perf_trace_session.h"

#include <exception>
#include <memory>
#include <string>

#include "perf_trace_report.h"
#include "tilexr_types.h"

namespace TileXRCollectives {
namespace Host {
namespace {

PerfTraceSession *g_activeSession = nullptr;

} // namespace

PerfTraceSession *GetActivePerfTraceSession()
{
    return g_activeSession;
}

void SetActivePerfTraceSessionForHost(PerfTraceSession *session)
{
    g_activeSession = session;
}

} // namespace Host
} // namespace TileXRCollectives

extern "C" int TileXRCollectivePerfSessionCreate(const TileXRCollectivePerfConfig *config,
                                                 TileXRCollectivePerfSession *session)
{
    if (config == nullptr || session == nullptr || config->enabled == 0 || config->outputDir == nullptr ||
        config->outputDir[0] == '\0' || config->sampleEveryN == 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    try {
        std::unique_ptr<TileXRCollectives::Host::PerfTraceSession> impl(
            new TileXRCollectives::Host::PerfTraceSession);
        impl->config = *config;
        impl->outputDir = config->outputDir;
        impl->config.outputDir = impl->outputDir.c_str();
        if (config->aiCommand != nullptr) {
            impl->aiCommand = config->aiCommand;
            impl->config.aiCommand = impl->aiCommand.c_str();
        }
        *session = impl.release();
        return TileXR::TILEXR_SUCCESS;
    } catch (const std::exception &) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    } catch (...) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
}

extern "C" int TileXRCollectivePerfSessionDestroy(TileXRCollectivePerfSession session)
{
    TileXRCollectives::Host::PerfTraceSession *impl =
        static_cast<TileXRCollectives::Host::PerfTraceSession *>(session);
    if (TileXRCollectives::Host::GetActivePerfTraceSession() == impl) {
        TileXRCollectives::Host::SetActivePerfTraceSessionForHost(nullptr);
    }
    delete impl;
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRCollectivePerfSetActiveSession(TileXRCollectivePerfSession session)
{
    TileXRCollectives::Host::SetActivePerfTraceSessionForHost(
        static_cast<TileXRCollectives::Host::PerfTraceSession *>(session));
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRCollectivePerfWriteReport(TileXRCollectivePerfSession session)
{
    if (session == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    try {
        TileXRCollectives::Host::PerfTraceSession *impl =
            static_cast<TileXRCollectives::Host::PerfTraceSession *>(session);
        TileXRCollectives::Host::PerfReportOptions options {};
        options.outputDir = impl->outputDir;
        options.emitAiPrompt = impl->config.emitAiPrompt != 0;
        return TileXRCollectives::Host::WritePerfTraceReports(impl->header, impl->hostStats, options);
    } catch (const std::exception &) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    } catch (...) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
}
