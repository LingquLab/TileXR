#include "perf_trace_session.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

aclError AclMallocDevice(void **ptr, size_t bytes)
{
    return aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
}

aclError AclFreeDevice(void *ptr)
{
    return aclrtFree(ptr);
}

aclError AclCopyHostToDeviceAsync(void *dst, size_t dstBytes, const void *src, size_t bytes,
                                  aclrtStream stream)
{
    return aclrtMemcpyAsync(dst, dstBytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
}

aclError AclMemsetDeviceAsync(void *dst, size_t dstBytes, int value, size_t bytes, aclrtStream stream)
{
    return aclrtMemsetAsync(dst, dstBytes, value, bytes, stream);
}

aclError AclCopyDeviceToHost(void *dst, size_t dstBytes, const void *src, size_t bytes)
{
    return aclrtMemcpy(dst, dstBytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

const PerfTraceRuntimeHooks kDefaultRuntimeHooks {
    AclMallocDevice,
    AclFreeDevice,
    AclCopyHostToDeviceAsync,
    AclMemsetDeviceAsync,
    AclCopyDeviceToHost,
};

const PerfTraceRuntimeHooks *g_runtimeHooks = &kDefaultRuntimeHooks;

bool ComputeStatsBytes(size_t statsCount, size_t *statsBytes)
{
    if (statsBytes == nullptr) {
        return false;
    }
    const size_t maxCount = std::numeric_limits<size_t>::max();
    if (statsCount > maxCount / sizeof(TileXR::TileXRPerfCoreStageStats)) {
        return false;
    }
    *statsBytes = statsCount * sizeof(TileXR::TileXRPerfCoreStageStats);
    return true;
}

bool ComputeRequiredBytes(uint64_t statsOffset, uint64_t statsBytes, size_t *requiredBytes)
{
    if (requiredBytes == nullptr || statsOffset > std::numeric_limits<uint64_t>::max() - statsBytes) {
        return false;
    }
    const uint64_t requiredBytes64 = statsOffset + statsBytes;
    if (requiredBytes64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *requiredBytes = static_cast<size_t>(requiredBytes64);
    return true;
}

bool DisableKernelProfiling()
{
    const char *value = std::getenv("TILEXR_COLLECTIVES_DISABLE_KERNEL_PROFILING");
    return value != nullptr &&
        (std::string(value) == "1" || std::string(value) == "true" || std::string(value) == "yes");
}

int64_t ComputeLaunchMessageBytes(TileXR::TileXRType opType, TileXR::TileXRDataType dataType,
                                  int64_t count, int rankSize)
{
    if (opType == TileXR::TileXRType::BROADCAST) {
        return count;
    }

    int64_t elementCount = count;
    if (opType == TileXR::TileXRType::REDUCE_SCATTER) {
        if (rankSize <= 0 || count > std::numeric_limits<int64_t>::max() / static_cast<int64_t>(rankSize)) {
            return -1;
        }
        elementCount = count * static_cast<int64_t>(rankSize);
    }
    return CountToBytes(elementCount, dataType);
}

} // namespace

PerfTraceSession *GetActivePerfTraceSession()
{
    return g_activeSession;
}

void SetActivePerfTraceSessionForHost(PerfTraceSession *session)
{
    g_activeSession = session;
}

void SetPerfTraceRuntimeHooksForTest(const PerfTraceRuntimeHooks *hooks)
{
    g_runtimeHooks = hooks == nullptr ? &kDefaultRuntimeHooks : hooks;
}

aclError FreePerfTraceDeviceBufferForHost(void *ptr)
{
    return g_runtimeHooks->freeDevice(ptr);
}

aclError CopyPerfTraceStatsToHost(void *dst, size_t dstBytes, const void *src, size_t bytes)
{
    return g_runtimeHooks->copyDeviceToHost(dst, dstBytes, src, bytes);
}

bool ValidatePerfTraceStatsLayout(const PerfTraceSession *session, size_t *statsBytes)
{
    if (session == nullptr || session->deviceBuffer == nullptr ||
        !ComputeStatsBytes(session->hostStats.size(), statsBytes) ||
        session->header.statsBytes != static_cast<uint64_t>(*statsBytes)) {
        return false;
    }
    size_t requiredBytes = 0;
    return ComputeRequiredBytes(session->header.statsOffset, session->header.statsBytes, &requiredBytes) &&
           requiredBytes <= session->deviceBufferBytes;
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

    session->deviceTraceReady = false;
    session->hostStats.clear();
    if (commArgs.rank < 0 || commArgs.rankSize <= 0 || commArgs.rank >= commArgs.rankSize ||
        commArgs.rankSize > TileXR::TILEXR_MAX_RANK_SIZE || blockDim == 0 || count < 0) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const int64_t messageBytes = ComputeLaunchMessageBytes(opType, dataType, count, commArgs.rankSize);
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
        (commArgs.extraFlag & TileXR::ExtraFlag::PERF_CYCLE_A5) != 0 ? 1000u : 50u;

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

    session->header.statsOffset = TileXR::TILEXR_PERF_TRACE_STATS_OFFSET;
    size_t statsBytes = 0;
    if (!ComputeStatsBytes(session->hostStats.size(), &statsBytes)) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    session->header.statsBytes = static_cast<uint64_t>(statsBytes);
    size_t requiredBytes = 0;
    if (!ComputeRequiredBytes(session->header.statsOffset, session->header.statsBytes, &requiredBytes)) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    if (DisableKernelProfiling()) {
        session->deviceTraceReady = false;
        *deviceTrace = nullptr;
        return TileXR::TILEXR_SUCCESS;
    }
    if (session->deviceBuffer == nullptr || !session->ownsDeviceBuffer || session->deviceBufferBytes < requiredBytes) {
        void *newBuffer = nullptr;
        aclError allocRet = g_runtimeHooks->mallocDevice(&newBuffer, requiredBytes);
        if (allocRet != ACL_SUCCESS || newBuffer == nullptr) {
            return TileXR::TILEXR_ERROR_INTERNAL;
        }
        if (session->ownsDeviceBuffer && session->deviceBuffer != nullptr) {
            try {
                session->retiredDeviceBuffers.push_back(session->deviceBuffer);
            } catch (const std::exception &) {
                g_runtimeHooks->freeDevice(newBuffer);
                return TileXR::TILEXR_ERROR_INTERNAL;
            } catch (...) {
                g_runtimeHooks->freeDevice(newBuffer);
                return TileXR::TILEXR_ERROR_INTERNAL;
            }
        }
        session->deviceBuffer = newBuffer;
        session->deviceBufferBytes = requiredBytes;
        session->ownsDeviceBuffer = true;
    }
    aclError copyRet = g_runtimeHooks->copyHostToDeviceAsync(session->deviceBuffer, sizeof(session->header),
        &session->header, sizeof(session->header), stream);
    if (copyRet != ACL_SUCCESS) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    void *statsDevice = static_cast<uint8_t *>(session->deviceBuffer) + session->header.statsOffset;
    aclError memsetRet = g_runtimeHooks->memsetDeviceAsync(statsDevice, static_cast<size_t>(session->header.statsBytes),
        0, static_cast<size_t>(session->header.statsBytes), stream);
    if (memsetRet != ACL_SUCCESS) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    session->deviceTraceReady = true;
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
    bool freeFailed = false;
    if (impl != nullptr) {
        if (impl->ownsDeviceBuffer && impl->deviceBuffer != nullptr) {
            freeFailed = TileXRCollectives::Host::FreePerfTraceDeviceBufferForHost(impl->deviceBuffer) != ACL_SUCCESS ||
                         freeFailed;
            impl->deviceBuffer = nullptr;
        }
        for (std::vector<void *>::iterator it = impl->retiredDeviceBuffers.begin();
             it != impl->retiredDeviceBuffers.end(); ++it) {
            if (*it != nullptr) {
                freeFailed = TileXRCollectives::Host::FreePerfTraceDeviceBufferForHost(*it) != ACL_SUCCESS ||
                             freeFailed;
            }
        }
    }
    delete impl;
    return freeFailed ? TileXR::TILEXR_ERROR_INTERNAL : TileXR::TILEXR_SUCCESS;
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
        if (impl->deviceTraceReady) {
            size_t expectedStatsBytes = 0;
            if (!TileXRCollectives::Host::ValidatePerfTraceStatsLayout(impl, &expectedStatsBytes)) {
                return TileXR::TILEXR_ERROR_INTERNAL;
            }
            const void *statsDevice = static_cast<const uint8_t *>(impl->deviceBuffer) + impl->header.statsOffset;
            aclError copyRet = TileXRCollectives::Host::CopyPerfTraceStatsToHost(
                impl->hostStats.data(), expectedStatsBytes, statsDevice, expectedStatsBytes);
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

extern "C" int TileXRCollectivePerfWriteIncompleteReport(TileXRCollectivePerfSession session, const char *reason)
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
        options.incomplete = true;
        options.incompleteReason = reason == nullptr || reason[0] == '\0' ? "kernel launch did not complete" : reason;
        return TileXRCollectives::Host::WritePerfTraceReports(impl->header, impl->hostStats, options);
    } catch (const std::exception &) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    } catch (...) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
}
