#include "prefetch_weight_host.h"

#include "prefetch_weight_launch.h"
#if defined(TILEXR_MOONEP_PREFETCH_UDMA_HOST_DEBUG)
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

#include "acl/acl.h"
#include "prefetch_weight_common.h"
#include "tilexr_udma_types.h"
#endif

namespace TileXRMoonEp {
#if defined(TILEXR_MOONEP_PREFETCH_UDMA_HOST_DEBUG)
namespace {

bool PrefetchUdmaDebugEnabled()
{
    const char *value = std::getenv("TILEXR_MOONEP_DEBUG_PREFETCH_UDMA");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

template <typename T>
bool CopyDeviceValue(uint64_t address, T *value)
{
    return address != 0 && value != nullptr &&
        aclrtMemcpy(value, sizeof(*value), reinterpret_cast<const void *>(address),
            sizeof(*value), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

void DumpPrefetchUdmaQueues(
    const PrefetchWeightLaunchContext &context, const char *phase)
{
    if (context.hostArgs == nullptr || context.hostArgs->udmaInfoPtr == nullptr ||
        phase == nullptr) {
        return;
    }

    TileXR::UDMAInfo info {};
    if (!CopyDeviceValue(reinterpret_cast<uint64_t>(context.hostArgs->udmaInfoPtr), &info) ||
        info.qpNum == 0 || info.sqPtr == 0 || info.scqPtr == 0) {
        std::fprintf(stderr, "[TileXR Prefetch UDMA rank %d] %s info_copy_failed\n",
            context.layout.rank, phase);
        return;
    }

    std::set<uint64_t> seenHeads;
    for (int32_t peer = 0; peer < context.layout.rankSize; ++peer) {
        for (uint32_t qp = 0; qp < info.qpNum; ++qp) {
            const uint64_t entry = static_cast<uint64_t>(peer) * info.qpNum + qp;
            TileXR::UDMAWQCtx wq {};
            TileXR::UDMACQCtx cq {};
            if (!CopyDeviceValue(info.sqPtr + entry * sizeof(wq), &wq) ||
                !CopyDeviceValue(info.scqPtr + entry * sizeof(cq), &cq) ||
                wq.headAddr == 0 || !seenHeads.insert(wq.headAddr).second) {
                continue;
            }

            uint32_t sqHead = 0;
            uint32_t sqTail = 0;
            uint32_t wqeCount = 0;
            uint32_t cqTail = 0;
            const bool scalarsOk = CopyDeviceValue(wq.headAddr, &sqHead) &&
                CopyDeviceValue(wq.tailAddr, &sqTail) &&
                CopyDeviceValue(wq.wqeCntAddr, &wqeCount) &&
                CopyDeviceValue(cq.tailAddr, &cqTail);
            std::fprintf(stderr,
                "[TileXR Prefetch UDMA rank %d] %s peer=%d qp=%u "
                "scalars_ok=%d head=%u tail=%u outstanding=%u wqe=%u cq_tail=%u\n",
                context.layout.rank, phase, peer, qp, scalarsOk ? 1 : 0,
                sqHead, sqTail, sqHead - sqTail, wqeCount, cqTail);
        }
    }
}

void DumpPrefetchUdmaRecords(int32_t rank,
    const std::vector<Kernel::PrefetchWeightUdmaDebugRecord> &records)
{
    for (const auto &record : records) {
        if (record.magic != Kernel::kPrefetchWeightUdmaDebugMagic) {
            continue;
        }
        std::fprintf(stderr,
            "[TileXR Prefetch target rank %d] worker=%u queue=%u peer=%d "
            "queue_id=0x%llx initial=%u posts=%u,%u,%u post_count=%u "
            "quiet=%u cq_before=%u completion_count=%u "
            "sq_head_before=%u sq_tail_before=%u poll=%u "
            "cq_after=%u sq_tail_after=%u wqe_after=%u\n",
            rank, record.worker, record.queue, record.peer,
            static_cast<unsigned long long>(record.queueId),
            record.initialTarget, record.targetAfterPost[0],
            record.targetAfterPost[1], record.targetAfterPost[2],
            record.postCount, record.quietTarget, record.cqTailAtQuiet,
            record.completionCount, record.sqHeadAtQuiet,
            record.sqTailAtQuiet, record.pollStatus, record.cqTailAfter,
            record.sqTailAfter, record.wqeCountAfter);
    }
}

} // namespace
#endif

int TileXRMoonEpRunPrefetchWeightV1(
    const TileXRMoonEpPrefetchWeightArgsV1 *args, aclrtStream stream)
{
    PrefetchWeightParams params {};
    PrefetchWeightLaunchContext context {};
    const int ret = TileXRMoonEpPreparePrefetchWeightLaunch(args, stream, &params, &context);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
#if defined(TILEXR_MOONEP_PREFETCH_UDMA_HOST_DEBUG)
    const bool debugUdma = PrefetchUdmaDebugEnabled();
    std::vector<Kernel::PrefetchWeightUdmaDebugRecord> debugRecords;
    if (debugUdma) {
        (void)aclrtSynchronizeStream(stream);
        DumpPrefetchUdmaQueues(context, "before");
        const uint64_t recordCount = static_cast<uint64_t>(context.layout.blockDim) *
            static_cast<uint64_t>(context.layout.rankSize);
        debugRecords.resize(static_cast<size_t>(recordCount));
        const size_t debugBytes = debugRecords.size() * sizeof(debugRecords[0]);
        void *debug = nullptr;
        if (aclrtMalloc(&debug, debugBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
            aclrtMemsetAsync(debug, debugBytes, 0, debugBytes, stream) == ACL_SUCCESS) {
            params.debug = debug;
            params.debugRecordCount = recordCount;
        } else {
            std::fprintf(stderr,
                "[TileXR Prefetch UDMA rank %d] debug_record_alloc_failed\n",
                context.layout.rank);
            if (debug != nullptr) {
                (void)aclrtFree(debug);
            }
        }
    }
#endif
    const int launchRet = TileXRMoonEpLaunchPrefetchWeightKernel(params, context);
#if defined(TILEXR_MOONEP_PREFETCH_UDMA_HOST_DEBUG)
    if (debugUdma && launchRet == TILEXR_MOONEP_SUCCESS) {
        const aclError syncRet = aclrtSynchronizeStream(stream);
        std::fprintf(stderr, "[TileXR Prefetch UDMA rank %d] after_sync=%d\n",
            context.layout.rank, static_cast<int>(syncRet));
        if (syncRet == ACL_SUCCESS && params.debug != nullptr &&
            aclrtMemcpy(debugRecords.data(),
                debugRecords.size() * sizeof(debugRecords[0]), params.debug,
                debugRecords.size() * sizeof(debugRecords[0]),
                ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
            DumpPrefetchUdmaRecords(context.layout.rank, debugRecords);
        }
        DumpPrefetchUdmaQueues(context, "after");
    }
    if (params.debug != nullptr) {
        (void)aclrtFree(params.debug);
    }
#endif
    return launchRet;
}

} // namespace TileXRMoonEp
