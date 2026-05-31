#include "perf_trace_session.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <string>

#include "acl/acl_rt.h"
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

    session->header.statsOffset = sizeof(TileXR::TileXRPerfTraceHeader);
    if (session->hostStats.size() > maxCount / sizeof(TileXR::TileXRPerfCoreStageStats)) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    const size_t statsBytes = session->hostStats.size() * sizeof(TileXR::TileXRPerfCoreStageStats);
    session->header.statsBytes = static_cast<uint64_t>(statsBytes);
    if (session->header.statsOffset > std::numeric_limits<uint64_t>::max() - session->header.statsBytes) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    const uint64_t requiredBytes64 = session->header.statsOffset + session->header.statsBytes;
    if (requiredBytes64 > static_cast<uint64_t>(maxCount)) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    const size_t requiredBytes = static_cast<size_t>(requiredBytes64);
    if (session->deviceBufferBytes < requiredBytes) {
        if (session->deviceBuffer != nullptr) {
            aclrtFree(session->deviceBuffer);
            session->deviceBuffer = nullptr;
            session->deviceBufferBytes = 0;
        }
        aclError allocRet = aclrtMalloc(&session->deviceBuffer, requiredBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (allocRet != ACL_SUCCESS) {
            return TileXR::TILEXR_ERROR_INTERNAL;
        }
        session->deviceBufferBytes = requiredBytes;
    }
    aclError copyRet = aclrtMemcpyAsync(session->deviceBuffer, sizeof(session->header),
        &session->header, sizeof(session->header), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (copyRet != ACL_SUCCESS) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    void *statsDevice = static_cast<uint8_t *>(session->deviceBuffer) + session->header.statsOffset;
    aclError memsetRet = aclrtMemsetAsync(statsDevice, static_cast<size_t>(session->header.statsBytes),
        0, static_cast<size_t>(session->header.statsBytes), stream);
    if (memsetRet != ACL_SUCCESS) {
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
    if (impl != nullptr && impl->deviceBuffer != nullptr) {
        aclrtFree(impl->deviceBuffer);
        impl->deviceBuffer = nullptr;
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
        if (impl->deviceBuffer != nullptr && impl->header.statsBytes > 0 && !impl->hostStats.empty()) {
            const void *statsDevice = static_cast<const uint8_t *>(impl->deviceBuffer) + impl->header.statsOffset;
            aclError copyRet = aclrtMemcpy(impl->hostStats.data(), static_cast<size_t>(impl->header.statsBytes),
                statsDevice, static_cast<size_t>(impl->header.statsBytes), ACL_MEMCPY_DEVICE_TO_HOST);
            if (copyRet != ACL_SUCCESS) {
                return TileXR::TILEXR_ERROR_INTERNAL;
            }
        }
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
