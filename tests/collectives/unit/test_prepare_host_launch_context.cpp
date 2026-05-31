#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "collective_kernel.h"
#include "collective_launcher.h"
#include "perf_trace_session.h"
#include "tilexr_collectives_perf.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

void CheckStatus(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

void CheckPointer(const char *label, const void *actual, const void *expected)
{
    if (actual != expected) {
        std::cerr << label << " was " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

void CheckMagic(const char *label, int64_t actual, int64_t expected)
{
    if (actual != expected) {
        std::cerr << label << " wrote magic " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

void CheckResetContext(const TileXRCollectives::Host::HostLaunchContext &context)
{
    CheckPointer("context.hostArgs", context.hostArgs, nullptr);
    CheckPointer("context.devArgs", context.devArgs, nullptr);
    CheckMagic("context.magic", context.magic, 0);
}

void CheckTrue(const char *label, bool condition)
{
    if (!condition) {
        std::cerr << label << " failed" << std::endl;
        ++g_failures;
    }
}

struct FakeDeviceAllocation {
    std::vector<uint8_t> bytes;
    bool freed = false;
};

std::vector<FakeDeviceAllocation> g_fakeAllocations;
int g_mallocCalls = 0;
int g_freeCalls = 0;
int g_copyHostToDeviceCalls = 0;
int g_memsetCalls = 0;
int g_copyDeviceToHostCalls = 0;
aclError g_nextCopyHostToDeviceResult = ACL_SUCCESS;
aclError g_nextMemsetResult = ACL_SUCCESS;

void ResetFakeRuntime()
{
    g_fakeAllocations.clear();
    g_mallocCalls = 0;
    g_freeCalls = 0;
    g_copyHostToDeviceCalls = 0;
    g_memsetCalls = 0;
    g_copyDeviceToHostCalls = 0;
    g_nextCopyHostToDeviceResult = ACL_SUCCESS;
    g_nextMemsetResult = ACL_SUCCESS;
}

FakeDeviceAllocation *FindAllocation(const void *ptr, size_t bytes)
{
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    for (std::vector<FakeDeviceAllocation>::iterator it = g_fakeAllocations.begin();
         it != g_fakeAllocations.end(); ++it) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(it->bytes.data());
        const uintptr_t end = begin + it->bytes.size();
        if (address >= begin && address <= end && bytes <= end - address) {
            return &(*it);
        }
    }
    return nullptr;
}

aclError FakeMallocDevice(void **ptr, size_t bytes)
{
    ++g_mallocCalls;
    if (ptr == nullptr || bytes == 0) {
        return 1;
    }
    g_fakeAllocations.push_back(FakeDeviceAllocation {});
    g_fakeAllocations.back().bytes.resize(bytes);
    *ptr = g_fakeAllocations.back().bytes.data();
    return ACL_SUCCESS;
}

aclError FakeFreeDevice(void *ptr)
{
    ++g_freeCalls;
    FakeDeviceAllocation *allocation = FindAllocation(ptr, 0);
    if (allocation == nullptr || allocation->freed) {
        return 1;
    }
    allocation->freed = true;
    return ACL_SUCCESS;
}

aclError FakeCopyHostToDeviceAsync(void *dst, size_t dstBytes, const void *src, size_t bytes, aclrtStream)
{
    ++g_copyHostToDeviceCalls;
    if (g_nextCopyHostToDeviceResult != ACL_SUCCESS) {
        aclError ret = g_nextCopyHostToDeviceResult;
        g_nextCopyHostToDeviceResult = ACL_SUCCESS;
        return ret;
    }
    FakeDeviceAllocation *allocation = FindAllocation(dst, bytes);
    if (allocation == nullptr || allocation->freed || src == nullptr || bytes > dstBytes) {
        return 1;
    }
    std::memcpy(dst, src, bytes);
    return ACL_SUCCESS;
}

aclError FakeMemsetDeviceAsync(void *dst, size_t dstBytes, int value, size_t bytes, aclrtStream)
{
    ++g_memsetCalls;
    if (g_nextMemsetResult != ACL_SUCCESS) {
        aclError ret = g_nextMemsetResult;
        g_nextMemsetResult = ACL_SUCCESS;
        return ret;
    }
    FakeDeviceAllocation *allocation = FindAllocation(dst, bytes);
    if (allocation == nullptr || allocation->freed || bytes > dstBytes) {
        return 1;
    }
    std::memset(dst, value, bytes);
    return ACL_SUCCESS;
}

aclError FakeCopyDeviceToHost(void *dst, size_t dstBytes, const void *src, size_t bytes)
{
    ++g_copyDeviceToHostCalls;
    FakeDeviceAllocation *allocation = FindAllocation(src, bytes);
    if (allocation == nullptr || allocation->freed || dst == nullptr || bytes > dstBytes) {
        return 1;
    }
    std::memcpy(dst, src, bytes);
    return ACL_SUCCESS;
}

const TileXRCollectives::Host::PerfTraceRuntimeHooks kFakeRuntimeHooks {
    FakeMallocDevice,
    FakeFreeDevice,
    FakeCopyHostToDeviceAsync,
    FakeMemsetDeviceAsync,
    FakeCopyDeviceToHost,
};

void CheckKernelArgsHasPerfTrace()
{
    TileXRCollectives::Host::AscendCCLKernelArgs args {};
    CheckPointer("args.perfTrace", args.perfTrace, nullptr);
    CheckTrue("perfTrace follows offset",
              offsetof(TileXRCollectives::Host::AscendCCLKernelArgs, perfTrace) >
              offsetof(TileXRCollectives::Host::AscendCCLKernelArgs, offset));
}

void CheckPerfTraceLaunchMetadata()
{
    ResetFakeRuntime();
    TileXRCollectives::Host::SetPerfTraceRuntimeHooksForTest(&kFakeRuntimeHooks);

    TileXR::CommArgs commArgs {};
    commArgs.rank = 1;
    commArgs.rankSize = 2;
    commArgs.extraFlag = TileXR::ExtraFlag::PERF_CYCLE_A5;

    TileXRCollectives::Host::PerfTraceSession session {};
    session.config.enabled = 1;

    const void *deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch enabled",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 4096, nullptr, &deviceTrace),
                TileXR::TILEXR_SUCCESS);
    CheckPointer("enabled deviceTrace", deviceTrace, session.deviceBuffer);
    CheckTrue("metadata trace ready", session.deviceTraceReady);
    CheckMagic("trace rank", session.header.rank, 1);
    CheckMagic("trace rankSize", session.header.rankSize, 2);
    CheckMagic("trace blockDim", session.header.blockDim, 4);
    CheckMagic("trace maxCoreCount", session.header.maxCoreCount, 4);
    CheckMagic("trace opType", session.header.opType,
               static_cast<int64_t>(TileXR::TileXRType::ALL_GATHER));
    CheckMagic("trace dataType", session.header.dataType, TileXR::TILEXR_DATA_TYPE_FP16);
    CheckMagic("trace messageBytes", session.header.messageBytes, 8192);
    CheckMagic("trace cycleToUsDivisor", session.header.cycleToUsDivisor, 1000);
    CheckMagic("trace stats size", static_cast<int64_t>(session.hostStats.size()),
               static_cast<int64_t>(commArgs.rankSize) * 4 * TileXR::TILEXR_PERF_STAGE_COUNT);

    CheckStatus("PreparePerfTraceLaunch null deviceTrace",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 4096, nullptr, nullptr),
                TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    commArgs.extraFlag = TileXR::ExtraFlag::TOPO_910A5;
    deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch topo a5 flag uses generic cycle",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 4096, nullptr, &deviceTrace),
                TileXR::TILEXR_SUCCESS);
    CheckMagic("topo a5 flag cycleToUsDivisor", session.header.cycleToUsDivisor, 50);
    commArgs.extraFlag = TileXR::ExtraFlag::PERF_CYCLE_A5;

    commArgs.rank = -1;
    deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch invalid rank",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 4096, nullptr, &deviceTrace),
                TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckPointer("invalid rank deviceTrace", deviceTrace, nullptr);
    CheckMagic("invalid rank stats size", static_cast<int64_t>(session.hostStats.size()), 0);
    commArgs.rank = 1;

    deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch invalid count",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, -1, nullptr, &deviceTrace),
                TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckPointer("invalid count deviceTrace", deviceTrace, nullptr);
    CheckMagic("invalid count stats size", static_cast<int64_t>(session.hostStats.size()), 0);

    commArgs.rankSize = 0;
    deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch zero rankSize",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 4096, nullptr, &deviceTrace),
                TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckPointer("zero rankSize deviceTrace", deviceTrace, nullptr);
    CheckMagic("zero rankSize stats size", static_cast<int64_t>(session.hostStats.size()), 0);

    session.header.rankSize = 7;
    commArgs.rankSize = TileXR::TILEXR_MAX_RANK_SIZE + 1;
    deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch oversized rankSize",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 4096, nullptr, &deviceTrace),
                TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckPointer("oversized rankSize deviceTrace", deviceTrace, nullptr);
    CheckMagic("oversized rankSize stats size", static_cast<int64_t>(session.hostStats.size()), 0);
    CheckMagic("oversized rankSize preserves header", session.header.rankSize, 7);

    session.config.enabled = 0;
    deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch disabled",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 4096, nullptr, &deviceTrace),
                TileXR::TILEXR_SUCCESS);
    CheckPointer("disabled deviceTrace", deviceTrace, nullptr);
    TileXRCollectives::Host::SetPerfTraceRuntimeHooksForTest(nullptr);
}

void CheckPerfTraceRuntimeHooks()
{
    ResetFakeRuntime();
    TileXRCollectives::Host::SetPerfTraceRuntimeHooksForTest(&kFakeRuntimeHooks);

    TileXRCollectivePerfConfig config {};
    config.enabled = 1;
    config.outputDir = "/tmp/tilexr_perf_prepare_context_test";
    config.sampleEveryN = 1;
    TileXRCollectivePerfSession opaqueSession = nullptr;
    CheckStatus("create perf session for hooks",
                TileXRCollectivePerfSessionCreate(&config, &opaqueSession), TileXR::TILEXR_SUCCESS);
    TileXRCollectives::Host::PerfTraceSession *session =
        static_cast<TileXRCollectives::Host::PerfTraceSession *>(opaqueSession);

    TileXR::CommArgs commArgs {};
    commArgs.rank = 0;
    commArgs.rankSize = 2;

    const void *deviceTrace = nullptr;
    CheckStatus("PreparePerfTraceLaunch fake runtime",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 2, 1024, nullptr, &deviceTrace),
                TileXR::TILEXR_SUCCESS);
    CheckPointer("fake runtime deviceTrace", deviceTrace, session->deviceBuffer);
    CheckTrue("fake runtime owns buffer", session->ownsDeviceBuffer);
    CheckTrue("fake runtime ready", session->deviceTraceReady);
    CheckMagic("fake runtime malloc calls", g_mallocCalls, 1);
    CheckMagic("fake runtime header copy calls", g_copyHostToDeviceCalls, 1);
    CheckMagic("fake runtime memset calls", g_memsetCalls, 1);
    CheckMagic("fake runtime free before destroy", g_freeCalls, 0);

    const size_t firstBufferBytes = session->deviceBufferBytes;
    void *firstBuffer = session->deviceBuffer;
    CheckStatus("PreparePerfTraceLaunch fake runtime grows",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 1024, nullptr, &deviceTrace),
                TileXR::TILEXR_SUCCESS);
    CheckTrue("fake runtime buffer grew", session->deviceBufferBytes > firstBufferBytes);
    CheckTrue("fake runtime swapped buffer", session->deviceBuffer != firstBuffer);
    CheckMagic("fake runtime retired buffers", static_cast<int64_t>(session->retiredDeviceBuffers.size()), 1);
    CheckMagic("fake runtime no eager free", g_freeCalls, 0);

    CheckStatus("write report copies ready trace",
                TileXRCollectivePerfWriteReport(opaqueSession), TileXR::TILEXR_SUCCESS);
    CheckMagic("fake runtime copy back calls", g_copyDeviceToHostCalls, 1);

    session->header.statsBytes += 1;
    CheckStatus("write report rejects malformed statsBytes",
                TileXRCollectivePerfWriteReport(opaqueSession), TileXR::TILEXR_ERROR_INTERNAL);
    session->header.statsBytes -= 1;

    g_nextCopyHostToDeviceResult = 7;
    deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch copy failure",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 1024, nullptr, &deviceTrace),
                TileXR::TILEXR_ERROR_INTERNAL);
    CheckPointer("copy failure deviceTrace", deviceTrace, nullptr);
    CheckTrue("copy failure not ready", !session->deviceTraceReady);
    CheckStatus("write report skips failed trace",
                TileXRCollectivePerfWriteReport(opaqueSession), TileXR::TILEXR_SUCCESS);
    CheckMagic("failed trace copy back skipped", g_copyDeviceToHostCalls, 1);

    CheckStatus("destroy fake runtime session",
                TileXRCollectivePerfSessionDestroy(opaqueSession), TileXR::TILEXR_SUCCESS);
    CheckMagic("fake runtime frees current and retired", g_freeCalls, 2);
    TileXRCollectives::Host::SetPerfTraceRuntimeHooksForTest(nullptr);
}

} // namespace

int main()
{
    CheckKernelArgsHasPerfTrace();
    CheckPerfTraceLaunchMetadata();
    CheckPerfTraceRuntimeHooks();

    TileXRCollectives::Host::HostLaunchContext context;

    CheckStatus("PrepareHostLaunchContext(nullptr, context)",
                TileXRCollectives::Host::PrepareHostLaunchContext(nullptr, context),
                TileXR::TILEXR_ERROR_INTERNAL);
    CheckResetContext(context);

    TileXRCommPtr comm = nullptr;
    CheckStatus("TileXRCommInit(0, 1, &comm)",
                TileXRCommInit(0, 1, &comm),
                TileXR::TILEXR_SUCCESS);
    if (comm == nullptr) {
        std::cerr << "TileXRCommInit returned success but left comm null" << std::endl;
        return 1;
    }

    context.hostArgs = reinterpret_cast<TileXR::CommArgs *>(0x1);
    context.devArgs = reinterpret_cast<GM_ADDR>(0x2);
    context.magic = 99;
    CheckStatus("PrepareHostLaunchContext(uninitialized comm, context)",
                TileXRCollectives::Host::PrepareHostLaunchContext(comm, context),
                TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    CheckResetContext(context);

    int64_t magic = -1;
    CheckStatus("TileXRCommNextMagic(comm, &magic)",
                TileXRCommNextMagic(comm, &magic),
                TileXR::TILEXR_SUCCESS);
    CheckMagic("TileXRCommNextMagic(comm, &magic)", magic, 1);

    CheckStatus("TileXRCommDestroy(comm)",
                TileXRCommDestroy(comm),
                TileXR::TILEXR_SUCCESS);
    return g_failures == 0 ? 0 : 1;
}
