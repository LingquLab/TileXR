#include <cstddef>
#include <cstdint>
#include <iostream>

#include "collective_kernel.h"
#include "collective_launcher.h"
#include "perf_trace_session.h"
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
    TileXR::CommArgs commArgs {};
    commArgs.rank = 1;
    commArgs.rankSize = 2;
    commArgs.extraFlag = TileXR::ExtraFlag::TOPO_910A5;

    TileXRCollectives::Host::PerfTraceSession session {};
    session.config.enabled = 1;
    session.deviceBuffer = reinterpret_cast<void *>(0x1234);

    const void *deviceTrace = reinterpret_cast<const void *>(0x1);
    CheckStatus("PreparePerfTraceLaunch enabled",
                TileXRCollectives::Host::PreparePerfTraceLaunch(
                    &session, commArgs, TileXR::TileXRType::ALL_GATHER,
                    TileXR::TILEXR_DATA_TYPE_FP16, 4, 4096, nullptr, &deviceTrace),
                TileXR::TILEXR_SUCCESS);
    CheckPointer("enabled deviceTrace", deviceTrace, session.deviceBuffer);
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
}

} // namespace

int main()
{
    CheckKernelArgsHasPerfTrace();
    CheckPerfTraceLaunchMetadata();

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
