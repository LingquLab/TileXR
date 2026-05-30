#include "perf_trace_session.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <string>

#include "collective_utils.h"
#include "comm_args.h"
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

int PreparePerfTraceLaunch(PerfTraceSession *session, const TileXR::CommArgs &commArgs,
                           TileXR::TileXRType opType, TileXR::TileXRDataType dataType,
                           uint32_t blockDim, int64_t count, aclrtStream stream,
                           const void **deviceTrace)
{
    (void)stream;
    if (deviceTrace == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *deviceTrace = nullptr;

    if (session == nullptr || session->config.enabled == 0) {
        return TileXR::TILEXR_SUCCESS;
    }

    session->hostStats.clear();
    if (commArgs.rank < 0 || commArgs.rankSize <= 0 || commArgs.rank >= commArgs.rankSize ||
        commArgs.rankSize > TileXR::TILEXR_MAX_RANK_SIZE || blockDim == 0 || count < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const int64_t messageBytes = CountToBytes(count, dataType);
    if (messageBytes < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    session->header = TileXR::TileXRPerfTraceHeader {};
    session->header.rank = static_cast<uint32_t>(commArgs.rank);
    session->header.rankSize = static_cast<uint32_t>(commArgs.rankSize);
    session->header.blockDim = blockDim;
    session->header.maxCoreCount = blockDim;
    session->header.opType = static_cast<uint32_t>(opType);
    session->header.dataType = static_cast<uint32_t>(dataType);
    session->header.messageBytes = static_cast<uint64_t>(messageBytes);
    session->header.cycleToUsDivisor =
        (commArgs.extraFlag & TileXR::ExtraFlag::TOPO_910A5) != 0 ? 1000u : 50u;

    if (session->header.stageCount == 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const size_t rankSize = static_cast<size_t>(commArgs.rankSize);
    const size_t coreCount = static_cast<size_t>(blockDim);
    const size_t stageCount = static_cast<size_t>(session->header.stageCount);
    const size_t maxCount = std::numeric_limits<size_t>::max();
    if (rankSize > maxCount / coreCount || rankSize * coreCount > maxCount / stageCount) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }

    const size_t statsCount = rankSize * coreCount * stageCount;
    if (statsCount > session->hostStats.max_size()) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }

    try {
        session->hostStats.assign(statsCount, TileXR::TileXRPerfCoreStageStats {});
    } catch (const std::exception &) {
        session->hostStats.clear();
        return TileXR::TILEXR_ERROR_INTERNAL;
    } catch (...) {
        session->hostStats.clear();
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    *deviceTrace = session->deviceBuffer;
    return TileXR::TILEXR_SUCCESS;
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
