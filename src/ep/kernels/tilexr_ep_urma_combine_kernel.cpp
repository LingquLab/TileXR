#include "comm_args.h"
#include "ep_urma_combine.h"
#include "ep_urma_udma.h"
#include "ep_window.h"
#include "kernel_operator.h"
#include "tilexr_data_as_flag.h"
#include "ep_urma_perf_trace.h"
#include "tilexr_udma.h"

namespace {

#if !defined(TILEXR_EP_URMA_CACHELESS)
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define TILEXR_EP_URMA_CACHELESS 1
#else
#define TILEXR_EP_URMA_CACHELESS 0
#endif
#endif

constexpr uint32_t kCursorAlignment = 32;
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
constexpr uint32_t kRouteMetaPrefetchCapacityBytes = 16 * 1024;
static_assert(sizeof(TileXREp::EpAssistTuple) == 16,
    "URMA combine metadata prefetch requires the existing 16-byte tuple ABI");
#endif
#if !TILEXR_EP_URMA_CACHELESS
constexpr uint32_t kRouteMetaBufferBytes = 32;
#endif
constexpr int64_t kPipelineBufferCount = 2;
using PackInputQueue = AscendC::TQue<AscendC::QuePosition::VECIN, 2>;
constexpr int64_t kRxTokenScheduleWindow = 3;


#define TILEXR_EP_SEND_FUNCTION inline

#define TILEXR_EP_LOCAL_FUNCTION inline

using PerfStats = TileXR::TileXRPerfCoreStageStats;
using PerfStage = TileXREp::EpUrmaCombinePerfStage;

enum class DcciProfileCategory : uint32_t {
    TX_DATA = 0,
    RX_FLAG_POLL = 1,
    RX_DATA = 2,
    CONTROL_OTHER = 3,
};

static_assert(TileXREp::kEpUrmaCombinePerfStageCount <=
    TileXR::TILEXR_PERF_TRACE_LOCAL_MAX_STAGE_COUNT,
    "URMA combine profile stages must fit the reserved UB trace region");
static_assert(TileXR::TILEXR_PERF_TRACE_LOCAL_STATS_UB_OFFSET +
    TileXREp::kEpUrmaCombinePerfStageCount * sizeof(PerfStats) <=
    TileXR::TILEXR_PERF_TRACE_MIN_UB_BYTES,
    "URMA combine profile stats exceed the minimum supported AIV UB");

__aicore__ TILEXR_EP_LOCAL_FUNCTION uint32_t PerfStageId(PerfStage stage)
{
    return static_cast<uint32_t>(stage);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION uint64_t ProfileBegin(GM_ADDR perfTrace)
{
    return TileXR::TileXRPerfCycleNow(perfTrace);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void ProfileEnd(
    GM_ADDR perfTrace, __ubuf__ PerfStats *stats, PerfStage stage, uint64_t startCycle)
{
    TileXR::TileXRPerfLocalRecord(perfTrace, stats, TileXREp::kEpUrmaCombinePerfStageCount,
        PerfStageId(stage), startCycle, TileXR::TileXRPerfCycleNow(perfTrace));
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void ProfileAux(
    GM_ADDR perfTrace, __ubuf__ PerfStats *stats, PerfStage stage, uint32_t auxIndex, uint64_t value)
{
    TileXR::TileXRPerfLocalAddAux(perfTrace, stats, TileXREp::kEpUrmaCombinePerfStageCount,
        PerfStageId(stage), auxIndex, value);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void ProfileDcciEnd(
    GM_ADDR perfTrace, __ubuf__ PerfStats *stats,
    DcciProfileCategory category, uint64_t startCycle)
{
    const uint64_t endCycle = TileXR::TileXRPerfCycleNow(perfTrace);
    TileXR::TileXRPerfLocalRecord(perfTrace, stats, TileXREp::kEpUrmaCombinePerfStageCount,
        PerfStageId(PerfStage::DCCI_TOTAL), startCycle, endCycle);
    if (endCycle >= startCycle) {
        TileXR::TileXRPerfLocalAddAux(perfTrace, stats, TileXREp::kEpUrmaCombinePerfStageCount,
            PerfStageId(PerfStage::DCCI_TOTAL), static_cast<uint32_t>(category), endCycle - startCycle);
    }
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION uint64_t ProfileKernelTimingBegin(GM_ADDR perfTrace)
{
    if (TileXR::TileXRPerfTraceEnabled(perfTrace)) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return static_cast<uint64_t>(AscendC::GetSystemCycle());
    }
    return 0;
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void ProfileFinish(
    GM_ADDR perfTrace, uint32_t rank, uint32_t core,
    __ubuf__ PerfStats *stats, uint64_t kernelStart)
{
    if (TileXR::TileXRPerfTraceEnabled(perfTrace)) {
        AscendC::PipeBarrier<PIPE_ALL>();
        const uint64_t kernelEnd = static_cast<uint64_t>(AscendC::GetSystemCycle());
        TileXR::TileXRPerfLocalRecord(perfTrace, stats,
            TileXREp::kEpUrmaCombinePerfStageCount,
            PerfStageId(PerfStage::KERNEL_TOTAL), kernelStart, kernelEnd);
    }
    TileXR::TileXRPerfLocalStatsFlush(perfTrace, rank, core,
        static_cast<uint32_t>(TileXREp::kEpUrmaCombineAivCount),
        TileXREp::kEpUrmaCombinePerfStageCount, stats);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION uint64_t StrictKernelTimingBegin(
    GM_ADDR strictKernelCycles)
{
    if (strictKernelCycles == nullptr) {
        return 0;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return static_cast<uint64_t>(AscendC::GetSystemCycle());
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void StrictKernelTimingFinish(
    GM_ADDR strictKernelCycles, uint32_t core, uint64_t startCycle)
{
    if (strictKernelCycles == nullptr) {
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint64_t endCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    const uint64_t duration = endCycle >= startCycle ? endCycle - startCycle : 0;
#if TILEXR_EP_URMA_CACHELESS
    AscendC::WriteGmByPassDCache(
        reinterpret_cast<__gm__ uint64_t *>(strictKernelCycles) + core, duration);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#else
    reinterpret_cast<__gm__ uint64_t *>(strictKernelCycles)[core] = duration;
#endif
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION bool ProfileBufferValid(
    GM_ADDR perfTrace, int64_t perfTraceBytes, int32_t rank, int32_t rankSize)
{
    if (perfTrace == nullptr || perfTraceBytes <
        static_cast<int64_t>(sizeof(TileXR::TileXRPerfTraceHeader)) || rank < 0 || rankSize <= 0) {
        return false;
    }
    const uint64_t requiredStatsBytes = static_cast<uint64_t>(rankSize) *
        TileXREp::kEpUrmaCombineAivCount * TileXREp::kEpUrmaCombinePerfStageCount * sizeof(PerfStats);
    const uint64_t requiredBytes = TileXR::TILEXR_PERF_TRACE_STATS_OFFSET + requiredStatsBytes;
    if (requiredBytes > static_cast<uint64_t>(perfTraceBytes)) {
        return false;
    }
    const auto header = reinterpret_cast<__gm__ TileXR::TileXRPerfTraceHeader *>(perfTrace);
    return header->magic == TileXR::TILEXR_PERF_TRACE_MAGIC &&
        header->version == TileXR::TILEXR_PERF_TRACE_VERSION &&
        header->headerSize == sizeof(TileXR::TileXRPerfTraceHeader) &&
        header->coreStageStatsSize == sizeof(PerfStats) &&
        header->rank == static_cast<uint32_t>(rank) &&
        header->rankSize == static_cast<uint32_t>(rankSize) &&
        header->blockDim == TileXREp::kEpUrmaCombineAivCount &&
        header->maxCoreCount == TileXREp::kEpUrmaCombineAivCount &&
        header->stageCount == TileXREp::kEpUrmaCombinePerfStageCount &&
        header->flags <= 2 && header->cycleToUsDivisor != 0 &&
        header->statsOffset == TileXR::TILEXR_PERF_TRACE_STATS_OFFSET &&
        header->statsBytes == requiredStatsBytes;
}

static_assert(TileXREp::kEpUrmaCombineDataBlockBytes == TileXR::DATA_AS_FLAG_BLOCK_BYTES,
    "URMA combine and DataAsFlag block sizes must match");
static_assert(TileXREp::kEpUrmaCombinePayloadBytes == TileXR::DATA_AS_FLAG_PAYLOAD_BYTES,
    "URMA combine and DataAsFlag payload sizes must match");
static_assert(TileXREp::kEpUrmaCombineFlagBytes == TileXR::DATA_AS_FLAG_FLAG_BYTES,
    "URMA combine and DataAsFlag flag sizes must match");
constexpr int64_t kMaxInputBytes = TileXREp::kEpUrmaCombineMaxHidden * sizeof(half);
constexpr int64_t kMaxFloatBytes = TileXREp::kEpUrmaCombineMaxHidden * sizeof(float);
constexpr float kMaxFiniteHalf = 65504.0f;
constexpr int64_t kMaxLogicalBytes =
    ((TileXREp::kEpUrmaCombineQuantHeaderBytes + TileXREp::kEpUrmaCombineMaxHidden +
      TileXREp::kEpUrmaCombinePayloadBytes - 1) /
     TileXREp::kEpUrmaCombinePayloadBytes) * TileXREp::kEpUrmaCombinePayloadBytes;
constexpr int64_t kMaxBlockCount = kMaxLogicalBytes / TileXREp::kEpUrmaCombinePayloadBytes;
constexpr int64_t kMaxRouteStride = kMaxBlockCount * TileXREp::kEpUrmaCombineDataBlockBytes;
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
constexpr int64_t kMaxPackUbBytes = kPipelineBufferCount * kMaxInputBytes +
    kPipelineBufferCount * kMaxRouteStride +
    kMaxFloatBytes + 2 * kMaxInputBytes + kCursorAlignment + kMaxLogicalBytes;
#else
constexpr int64_t kMaxPackUbBytes = kPipelineBufferCount * kMaxInputBytes +
    kPipelineBufferCount * kMaxRouteStride +
    3 * kMaxFloatBytes + kCursorAlignment + kMaxLogicalBytes;
#endif
constexpr int64_t kMaxRouteFlagBytes =
    kMaxBlockCount * TileXREp::kEpUrmaCombineFlagBytes;
#if TILEXR_EP_URMA_RX_READY_BATCH_MTE2
constexpr int64_t kMaxReadyFlagBufferBytes =
    TileXREp::kEpUrmaCombineMaxTopK * kMaxRouteFlagBytes;
#else
constexpr int64_t kMaxReadyFlagBufferBytes = kMaxRouteFlagBytes;
#endif
#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
constexpr int64_t kMaxReadyVectorBufferBytes =
    TileXREp::kEpUrmaCombineMaxTopK * kMaxRouteFlagBytes + kMaxRouteFlagBytes +
    TileXREp::kEpUrmaCombineMaxTopK * kCursorAlignment;
#else
constexpr int64_t kMaxReadyVectorBufferBytes = 0;
#endif
constexpr int64_t kMaxReceiveUbBytes = kPipelineBufferCount * kMaxLogicalBytes +
    kMaxReadyFlagBufferBytes + kMaxRouteFlagBytes + kMaxReadyVectorBufferBytes +
    2 * kMaxFloatBytes + kMaxInputBytes +
    TileXREp::kEpUrmaCombineMaxTopK * static_cast<int64_t>(sizeof(float));
static_assert(kMaxPackUbBytes <= TileXR::TILEXR_PERF_TRACE_LOCAL_STATS_UB_OFFSET,
    "URMA combine Pack ping-pong buffers overlap the profiling UB region");
static_assert(kMaxReceiveUbBytes <= TileXR::TILEXR_PERF_TRACE_LOCAL_STATS_UB_OFFSET,
    "URMA combine Receive ping-pong buffers overlap the profiling UB region");
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
constexpr int64_t kMaxSendCursorBytes =
    ((TileXREp::kEpUrmaCombinePackLaneCount * static_cast<int64_t>(sizeof(uint32_t)) +
      kCursorAlignment - 1) / kCursorAlignment) * kCursorAlignment;
constexpr int64_t kMaxSendUsedPeerBytes =
    ((TileXR::TILEXR_MAX_RANK_SIZE + kCursorAlignment - 1) / kCursorAlignment) * kCursorAlignment;
constexpr int64_t kMaxSendSelfCopyBytes = kMaxBlockCount *
    (TileXREp::kEpUrmaCombinePayloadBytes + TileXREp::kEpUrmaCombineFlagBytes);
constexpr int64_t kMaxSendUbBytes = kMaxSendCursorBytes + kMaxSendUsedPeerBytes +
    kMaxSendSelfCopyBytes + kRouteMetaPrefetchCapacityBytes;
static_assert(kMaxSendUbBytes <= TileXR::TILEXR_PERF_TRACE_LOCAL_STATS_UB_OFFSET,
    "URMA combine Send metadata prefetch overlaps the profiling UB region");
#endif

__aicore__ TILEXR_EP_LOCAL_FUNCTION int64_t AlignUpInt64(int64_t value, int64_t alignment)
{
    const int64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

template <typename T>
__aicore__ TILEXR_EP_LOCAL_FUNCTION void SetNoCacheRead(AscendC::GlobalTensor<T> &tensor)
{
#if TILEXR_EP_URMA_CACHELESS
    tensor.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
        AscendC::CacheMode::CACHE_MODE_DISABLE);
#else
    (void)tensor;
#endif
}

template <typename T>
__aicore__ TILEXR_EP_LOCAL_FUNCTION void SetNoCacheWrite(AscendC::GlobalTensor<T> &tensor)
{
#if TILEXR_EP_URMA_CACHELESS
    tensor.template SetL2CacheHint<AscendC::CacheRwMode::WRITE>(
        AscendC::CacheMode::CACHE_MODE_DISABLE);
#else
    (void)tensor;
#endif
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void CachelessAcquireBarrier()
{
#if TILEXR_EP_URMA_CACHELESS
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION uint64_t EncodeControlValue(int64_t magic, uint32_t step)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(magic)) << 32) | static_cast<uint64_t>(step);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void StoreControlValue(
    GM_ADDR lineAddr, uint64_t value,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
#if TILEXR_EP_URMA_CACHELESS
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::WriteGmByPassDCache(
        reinterpret_cast<__gm__ uint64_t *>(lineAddr), value);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
#else
    auto line = reinterpret_cast<__gm__ uint64_t *>(lineAddr);
    line[0] = value;
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    TileXR::UDMACleanCacheLines(
        reinterpret_cast<__gm__ uint8_t *>(lineAddr), TileXREp::kEpUrmaCombineCacheLineBytes);
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::CONTROL_OTHER, dcciStart);
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}


__aicore__ TILEXR_EP_LOCAL_FUNCTION uint64_t LoadControlValue(
    GM_ADDR lineAddr, GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
#if TILEXR_EP_URMA_CACHELESS
    const uint64_t value = AscendC::ReadGmByPassDCache(
        reinterpret_cast<__gm__ uint64_t *>(lineAddr));
    return value;
#else
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    TileXR::UDMACleanCacheLines(
        reinterpret_cast<__gm__ uint8_t *>(lineAddr), TileXREp::kEpUrmaCombineCacheLineBytes);
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::CONTROL_OTHER, dcciStart);
    AscendC::PipeBarrier<PIPE_ALL>();
    return reinterpret_cast<__gm__ uint64_t *>(lineAddr)[0];
#endif
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void StoreError(
    GM_ADDR workspaceGM, int64_t errorStatusOffset, int64_t magic, uint64_t status,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    if (status != TileXREp::kEpUrmaCombineStatusOk) {
        StoreControlValue(workspaceGM + errorStatusOffset,
            EncodeControlValue(magic, static_cast<uint32_t>(status)), perfTrace, perfStats);
    }
}

#if TILEXR_EP_URMA_CACHELESS || TILEXR_EP_URMA_TX_META_PREFETCH_FULL
__aicore__ TILEXR_EP_LOCAL_FUNCTION TileXREp::EpAssistTuple LoadRouteMetaBypass(
    GM_ADDR assistInfoGM, int64_t index)
{
    __gm__ int32_t *src = reinterpret_cast<__gm__ int32_t *>(assistInfoGM) +
        index * TileXREp::kEpAssistTupleInts;
    TileXREp::EpAssistTuple tuple;
    tuple.srcRank = AscendC::ReadGmByPassDCache(src);
    tuple.tokenId = AscendC::ReadGmByPassDCache(src + 1);
    tuple.topKId = AscendC::ReadGmByPassDCache(src + 2);
    tuple.expertId = 0;
    return tuple;
}
#endif

#if TILEXR_EP_URMA_CACHELESS
__aicore__ TILEXR_EP_LOCAL_FUNCTION TileXREp::EpAssistTuple LoadRouteMeta(
    GM_ADDR assistInfoGM, int64_t index)
{
    return LoadRouteMetaBypass(assistInfoGM, index);
}
#else
__aicore__ TILEXR_EP_LOCAL_FUNCTION TileXREp::EpAssistTuple LoadRouteMeta(
    GM_ADDR assistInfoGM, int64_t index, AscendC::LocalTensor<int32_t> &local)
{
    AscendC::GlobalTensor<int32_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(assistInfoGM) +
        index * TileXREp::kEpAssistTupleInts, TileXREp::kEpAssistTupleInts);
    AscendC::DataCopyExtParams copyParams {
        1, static_cast<uint32_t>(sizeof(TileXREp::EpAssistTuple)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
    AscendC::DataCopyPad(local, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    TileXREp::EpAssistTuple tuple;
    tuple.srcRank = local.GetValue(0);
    tuple.tokenId = local.GetValue(1);
    tuple.topKId = local.GetValue(2);
    tuple.expertId = local.GetValue(3);
    return tuple;
}
#endif

#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
__aicore__ TILEXR_EP_LOCAL_FUNCTION void StartRouteMetaPrefetch(
    GM_ADDR assistInfoGM, int64_t tupleCount,
    AscendC::LocalTensor<int32_t> &local, event_t eventId)
{
    AscendC::GlobalTensor<int32_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(assistInfoGM),
        tupleCount * TileXREp::kEpAssistTupleInts);
    src.SetL2CacheHint<AscendC::CacheRwMode::READ>(
        AscendC::CacheMode::CACHE_MODE_DISABLE);
    AscendC::DataCopyExtParams copyParams {
        1, static_cast<uint32_t>(tupleCount * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple))),
        0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padParams {false, 0, 0, 0};
    AscendC::DataCopyPad(local, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION TileXREp::EpAssistTuple LoadPrefetchedRouteMeta(
    AscendC::LocalTensor<int32_t> &local, int64_t index)
{
    const uint32_t begin = static_cast<uint32_t>(index * TileXREp::kEpAssistTupleInts);
    TileXREp::EpAssistTuple tuple;
    tuple.srcRank = local.GetValue(begin);
    tuple.tokenId = local.GetValue(begin + 1);
    tuple.topKId = local.GetValue(begin + 2);
    tuple.expertId = 0;
    return tuple;
}
#endif

__aicore__ TILEXR_EP_LOCAL_FUNCTION bool RouteMetaValid(
    const TileXREp::EpAssistTuple &tuple, int32_t rankSize, int64_t bs, int64_t topK)
{
    return tuple.srcRank >= 0 && tuple.srcRank < rankSize && tuple.tokenId >= 0 && tuple.tokenId < bs &&
        tuple.topKId >= 0 && tuple.topKId < topK;
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void LoadTopKWeights(
    GM_ADDR weightsGM, int64_t begin, int64_t topK, AscendC::LocalTensor<float> &local)
{
    AscendC::GlobalTensor<float> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightsGM) + begin, topK);
    const uint32_t alignedElements = static_cast<uint32_t>(
        AlignUpInt64(topK * static_cast<int64_t>(sizeof(float)), kCursorAlignment) / sizeof(float));
    AscendC::DataCopyExtParams copyParams {
        1, static_cast<uint32_t>(topK * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padParams {
        true, 0, static_cast<uint8_t>(alignedElements - static_cast<uint32_t>(topK)), 0.0f};
    AscendC::DataCopyPad(local, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION event_t PipelineEvent(int64_t slot)
{
    return slot == 0 ? EVENT_ID0 : EVENT_ID1;
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void EnqueuePackInput(
    const AscendC::GlobalTensor<half> &expertOut, int64_t route,
    int64_t h, uint8_t inputPadding, PackInputQueue &inputQueue)
{
    AscendC::LocalTensor<half> input = inputQueue.AllocTensor<half>();
    AscendC::DataCopyExtParams inputParams {
        1, static_cast<uint32_t>(h * static_cast<int64_t>(sizeof(half))), 0, 0, 0};
    AscendC::DataCopyPadExtParams<half> inputPad {true, 0, inputPadding, static_cast<half>(0.0f)};
    AscendC::DataCopyPad(input, expertOut[route * h], inputParams, inputPad);
    inputQueue.EnQue(input);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void StartPackPublish(
    GM_ADDR txRouteAddr, int64_t routeStride,
    AscendC::LocalTensor<float> &packed, event_t eventId)
{
    AscendC::GlobalTensor<float> txRoute;
    txRoute.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(txRouteAddr), routeStride / sizeof(float));
    SetNoCacheWrite(txRoute);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
    AscendC::DataCopy(txRoute, packed, static_cast<uint32_t>(routeStride / sizeof(float)));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void FinishPackCopy(
    GM_ADDR txRouteAddr, int64_t routeStride, event_t eventId,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    const uint64_t waitStart = ProfileBegin(perfTrace);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    ProfileEnd(perfTrace, perfStats, PerfStage::PACK_MTE3_EXPOSED_WAIT, waitStart);
    ProfileAux(perfTrace, perfStats, PerfStage::PACK_MTE3_EXPOSED_WAIT, 0,
        static_cast<uint64_t>(routeStride));
#if !TILEXR_EP_URMA_CACHELESS
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(txRouteAddr), routeStride);
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::TX_DATA, dcciStart);
#endif
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void PublishTxReadyInData(
    GM_ADDR txRouteAddr, uint64_t readyValue,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
#if TILEXR_EP_URMA_CACHELESS
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::WriteGmByPassDCache(
        reinterpret_cast<__gm__ uint64_t *>(
            txRouteAddr + TileXREp::kEpUrmaCombineTxReadyHeaderOffset),
        readyValue);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
#else
    reinterpret_cast<__gm__ uint64_t *>(
        txRouteAddr + TileXREp::kEpUrmaCombineTxReadyHeaderOffset)[0] = readyValue;
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    TileXR::UDMACleanCacheLines(
        reinterpret_cast<__gm__ uint8_t *>(txRouteAddr),
        TileXREp::kEpUrmaCombineCacheLineBytes);
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::CONTROL_OTHER, dcciStart);
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION uint64_t LoadTxReadyInData(
    GM_ADDR txRouteAddr, GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
#if TILEXR_EP_URMA_CACHELESS
    const uint64_t value = AscendC::ReadGmByPassDCache(
        reinterpret_cast<__gm__ uint64_t *>(
            txRouteAddr + TileXREp::kEpUrmaCombineTxReadyHeaderOffset));
    return value;
#else
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    TileXR::UDMACleanCacheLines(
        reinterpret_cast<__gm__ uint8_t *>(txRouteAddr),
        TileXREp::kEpUrmaCombineCacheLineBytes);
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::CONTROL_OTHER, dcciStart);
    AscendC::PipeBarrier<PIPE_ALL>();
    return reinterpret_cast<__gm__ uint64_t *>(
        txRouteAddr + TileXREp::kEpUrmaCombineTxReadyHeaderOffset)[0];
#endif
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void PublishTxReadyBatch(
    GM_ADDR workspaceGM, int64_t txReadyOffset,
    int64_t txDataOffset, int64_t routeStride, int64_t firstRoute, int64_t routeCount,
    uint64_t readyValue, GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    if (routeCount <= 0) {
        return;
    }
#if TILEXR_EP_URMA_TX_READY_IN_DATA
    (void)txReadyOffset;
    for (int64_t index = 0; index < routeCount; ++index) {
        PublishTxReadyInData(workspaceGM + txDataOffset +
            (firstRoute + index) * routeStride, readyValue, perfTrace, perfStats);
    }
#else
    (void)txDataOffset;
    (void)routeStride;
    // In shared mode the one published line releases the whole lane-local batch. PackRoutes
    // reaches this call only after every route represented by that line has completed MTE3.
#if TILEXR_EP_URMA_TX_READY_SHARED_FLAG && TILEXR_EP_URMA_TX_READY_BATCH_SIZE > 1
    const int64_t publishedRouteCount = 1;
#else
    const int64_t publishedRouteCount = routeCount;
#endif
    GM_ADDR firstLineAddr = workspaceGM + txReadyOffset +
        firstRoute * TileXREp::kEpUrmaCombineCacheLineBytes;
#if TILEXR_EP_URMA_CACHELESS
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int64_t index = 0; index < publishedRouteCount; ++index) {
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint64_t *>(firstLineAddr +
                index * TileXREp::kEpUrmaCombineCacheLineBytes),
            readyValue);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
#else
    for (int64_t index = 0; index < publishedRouteCount; ++index) {
        reinterpret_cast<__gm__ uint64_t *>(firstLineAddr +
            index * TileXREp::kEpUrmaCombineCacheLineBytes)[0] = readyValue;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(firstLineAddr),
        publishedRouteCount * TileXREp::kEpUrmaCombineCacheLineBytes);
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::CONTROL_OTHER, dcciStart);
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
#endif
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void FinishAndPublishPackBatch(
    GM_ADDR workspaceGM, int64_t txReadyOffset,
    int64_t txDataOffset, int64_t routeStride, int64_t firstRoute, int64_t routeCount,
    event_t eventId, uint64_t readyValue, uint64_t firstReadyStart, bool &firstReadyPublished,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    GM_ADDR txRouteAddr = workspaceGM + txDataOffset + firstRoute * routeStride;
    FinishPackCopy(txRouteAddr, routeStride, eventId, perfTrace, perfStats);

    const uint64_t publishStart = ProfileBegin(perfTrace);
    PublishTxReadyBatch(workspaceGM, txReadyOffset, txDataOffset, routeStride,
        firstRoute, routeCount, readyValue, perfTrace, perfStats);
    ProfileEnd(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, publishStart);
#if TILEXR_EP_URMA_TX_READY_SHARED_FLAG && TILEXR_EP_URMA_TX_READY_BATCH_SIZE > 1
    constexpr uint64_t publishedLineCount = 1;
#else
    const uint64_t publishedLineCount = static_cast<uint64_t>(routeCount);
#endif
    ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 1, publishedLineCount);
    ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 2,
        static_cast<uint64_t>(routeCount));
    ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 3,
        publishedLineCount * TileXREp::kEpUrmaCombineCacheLineBytes);
    if (!firstReadyPublished) {
        ProfileEnd(perfTrace, perfStats, PerfStage::PACK_FIRST_TX_READY, firstReadyStart);
        ProfileAux(perfTrace, perfStats, PerfStage::PACK_FIRST_TX_READY, 0,
            static_cast<uint64_t>(firstRoute));
        firstReadyPublished = true;
    }
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION int64_t TxReadyPollRoute(
    int64_t route, int64_t laneBegin)
{
#if TILEXR_EP_URMA_TX_READY_SHARED_FLAG && TILEXR_EP_URMA_TX_READY_BATCH_SIZE > 1
    // Batches restart at each Pack lane boundary; global route/TXB rounding is incorrect when
    // selfSendCnt is not divisible by the number of Pack lanes.
    return laneBegin + ((route - laneBegin) /
        static_cast<int64_t>(TileXREp::kEpUrmaCombineTxReadyBatchSize)) *
        static_cast<int64_t>(TileXREp::kEpUrmaCombineTxReadyBatchSize);
#else
    (void)laneBegin;
    return route;
#endif
}

__aicore__ __attribute__((always_inline)) inline void StartPackQuantization(
    AscendC::LocalTensor<half> &routeInput,
    AscendC::LocalTensor<float> &routePacked,
    AscendC::LocalTensor<float> &quantFloat,
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
    AscendC::LocalTensor<half> &absHalf,
    AscendC::LocalTensor<half> &reduceOut,
    AscendC::LocalTensor<half> &reduceTmp,
#else
    AscendC::LocalTensor<float> &absFloat,
    AscendC::LocalTensor<float> &reduceOut,
    AscendC::LocalTensor<float> &reduceTmp,
#endif
    AscendC::LocalTensor<int8_t> &logical,
    int64_t h, int64_t logicalBytes, int64_t routeStride, AscendC::TEventID quantizeEvent)
{
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
    (void)routePacked;
    (void)logical;
    (void)logicalBytes;
    (void)routeStride;
    AscendC::Abs<half>(absHalf, routeInput, static_cast<uint32_t>(h));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax<half>(reduceOut, absHalf, reduceTmp, static_cast<int32_t>(h), false);
#if TILEXR_EP_URMA_QDC_VERSION != 3
    AscendC::Cast<float, half>(quantFloat, routeInput, AscendC::RoundMode::CAST_NONE,
        static_cast<uint32_t>(h));
#else
    (void)quantFloat;
#endif
#else
    AscendC::Duplicate<int8_t>(logical, static_cast<int8_t>(0), static_cast<uint32_t>(logicalBytes));
    AscendC::Duplicate<float>(routePacked, TileXR::DATA_AS_FLAG_READY_VALUE,
        static_cast<uint32_t>(routeStride / sizeof(float)));
    AscendC::Cast<float, half>(quantFloat, routeInput, AscendC::RoundMode::CAST_NONE,
        static_cast<uint32_t>(h));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs<float>(absFloat, quantFloat, static_cast<uint32_t>(h));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax<float>(reduceOut, absFloat, reduceTmp, static_cast<int32_t>(h), false);
#endif
    AscendC::SetFlag<AscendC::HardEvent::V_S>(quantizeEvent);
}

__aicore__ __attribute__((always_inline)) inline void FinishPackQuantization(
    AscendC::LocalTensor<half> &routeInput,
    AscendC::LocalTensor<float> &routePacked,
    AscendC::LocalTensor<float> &quantFloat,
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
    AscendC::LocalTensor<half> &reduceOut,
#else
    AscendC::LocalTensor<float> &reduceOut,
#endif
    AscendC::LocalTensor<int8_t> &logical,
    AscendC::LocalTensor<float> &logicalFloat,
    AscendC::LocalTensor<int32_t> &logicalInt,
    int64_t h, int64_t blockCount, uint8_t repeats,
    AscendC::TEventID quantizeEvent, AscendC::TEventID scalarToVectorEvent)
{
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(quantizeEvent);
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
    const float maxAbs = static_cast<float>(reduceOut(0));
#else
    const float maxAbs = reduceOut(0);
#endif
    const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
    const float inverseScale = maxAbs > 0.0f ? 127.0f / maxAbs : 1.0f;
    logicalFloat.SetValue(0, scale);
    logicalInt.SetValue(1, static_cast<int32_t>(TileXREp::kEpUrmaCombineQuantModeInt8PerRoute));
#if TILEXR_EP_URMA_TX_READY_IN_DATA
    constexpr uint32_t readyWordIndex = static_cast<uint32_t>(
        TileXREp::kEpUrmaCombineTxReadyHeaderOffset / sizeof(int32_t));
    logicalInt.SetValue(readyWordIndex, 0);
    logicalInt.SetValue(readyWordIndex + 1, 0);
#endif
#if TILEXR_EP_URMA_QDC_VERSION == 3
    const bool halfScaleSafe = maxAbs >= 0.0f && maxAbs <= kMaxFiniteHalf &&
        inverseScale > 0.0f && inverseScale <= kMaxFiniteHalf;
    if (halfScaleSafe) {
        AscendC::Muls<half>(routeInput, routeInput, static_cast<half>(inverseScale),
            static_cast<uint32_t>(h));
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::Cast<float, half>(quantFloat, routeInput, AscendC::RoundMode::CAST_NONE,
            static_cast<uint32_t>(h));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(quantFloat, quantFloat, inverseScale, static_cast<uint32_t>(h));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<half, float>(routeInput, quantFloat, AscendC::RoundMode::CAST_RINT,
            static_cast<uint32_t>(h));
        AscendC::PipeBarrier<PIPE_V>();
    }
#else
    AscendC::Muls<float>(quantFloat, quantFloat, inverseScale, static_cast<uint32_t>(h));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast<half, float>(routeInput, quantFloat, AscendC::RoundMode::CAST_RINT,
        static_cast<uint32_t>(h));
    AscendC::PipeBarrier<PIPE_V>();
#endif
    AscendC::Cast<int8_t, half>(logical[TileXREp::kEpUrmaCombineQuantHeaderBytes], routeInput,
        AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(h));
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVectorEvent);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVectorEvent);
    AscendC::Copy(routePacked, logicalFloat, static_cast<uint64_t>(64), repeats, {1, 1, 16, 15});
    AscendC::Copy(routePacked[64], logicalFloat[64], static_cast<uint64_t>(56), repeats, {1, 1, 16, 15});
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void PackRoutes(
    GM_ADDR expertOutGM, GM_ADDR workspaceGM, int64_t selfSendCnt, int64_t h,
    int64_t magic, int64_t blockCount, int64_t routeStride, int64_t txReadyOffset, int64_t txDataOffset,
    int64_t laneId, AscendC::TPipe &pipe, GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    const uint64_t firstReadyStart = ProfileBegin(perfTrace);
    const int64_t inputBytes = AlignUpInt64(h * static_cast<int64_t>(sizeof(half)), kCursorAlignment);
    const int64_t floatBytes = AlignUpInt64(h * static_cast<int64_t>(sizeof(float)), kCursorAlignment);
    const int64_t logicalBytes = blockCount * TileXREp::kEpUrmaCombinePayloadBytes;

    PackInputQueue inputQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> quantFloatBuf;
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
    AscendC::TBuf<AscendC::QuePosition::VECCALC> absHalfBuf;
#else
    AscendC::TBuf<AscendC::QuePosition::VECCALC> absFloatBuf;
#endif
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceOutBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> logicalBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> packedBuf;
    pipe.InitBuffer(inputQueue, static_cast<uint8_t>(kPipelineBufferCount), static_cast<uint32_t>(inputBytes));
    pipe.InitBuffer(quantFloatBuf, static_cast<uint32_t>(floatBytes));
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
    pipe.InitBuffer(absHalfBuf, static_cast<uint32_t>(inputBytes));
    pipe.InitBuffer(reduceOutBuf, kCursorAlignment);
    pipe.InitBuffer(reduceTmpBuf, static_cast<uint32_t>(inputBytes));
#else
    pipe.InitBuffer(absFloatBuf, static_cast<uint32_t>(floatBytes));
    pipe.InitBuffer(reduceOutBuf, kCursorAlignment);
    pipe.InitBuffer(reduceTmpBuf, static_cast<uint32_t>(floatBytes));
#endif
    pipe.InitBuffer(logicalBuf, static_cast<uint32_t>(logicalBytes));
    pipe.InitBuffer(packedBuf, static_cast<uint32_t>(kPipelineBufferCount * routeStride));
    AscendC::LocalTensor<float> packed0 = packedBuf.GetWithOffset<float>(
        static_cast<uint32_t>(routeStride / sizeof(float)), 0);
    AscendC::LocalTensor<float> packed1 = packedBuf.GetWithOffset<float>(
        static_cast<uint32_t>(routeStride / sizeof(float)), static_cast<uint32_t>(routeStride));
    AscendC::LocalTensor<float> quantFloat = quantFloatBuf.Get<float>();
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
    AscendC::LocalTensor<half> absHalf = absHalfBuf.Get<half>();
    AscendC::LocalTensor<half> reduceOutHalf = reduceOutBuf.Get<half>();
    AscendC::LocalTensor<half> reduceTmpHalf = reduceTmpBuf.Get<half>();
#else
    AscendC::LocalTensor<float> absFloat = absFloatBuf.Get<float>();
    AscendC::LocalTensor<float> reduceOut = reduceOutBuf.Get<float>();
    AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
#endif
    AscendC::LocalTensor<int8_t> logical = logicalBuf.Get<int8_t>();
    AscendC::LocalTensor<float> logicalFloat = logical.template ReinterpretCast<float>();
    AscendC::LocalTensor<int32_t> logicalInt = logical.template ReinterpretCast<int32_t>();
    AscendC::GlobalTensor<half> expertOut;
    expertOut.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(expertOutGM), selfSendCnt * h);

    const int64_t begin = selfSendCnt * laneId / TileXREp::kEpUrmaCombinePackLaneCount;
    const int64_t end = selfSendCnt * (laneId + 1) / TileXREp::kEpUrmaCombinePackLaneCount;
    const uint64_t readyValue = EncodeControlValue(magic, TileXREp::kEpUrmaCombineTxRouteReady);
    const uint32_t inputAlignedElements = static_cast<uint32_t>(inputBytes / sizeof(half));
    const uint8_t inputPadding = static_cast<uint8_t>(inputAlignedElements - static_cast<uint32_t>(h));
    const uint8_t repeats = static_cast<uint8_t>(blockCount);
    const AscendC::TEventID quantizeEvent = pipe.FetchEventID<AscendC::HardEvent::V_S>();
    const AscendC::TEventID scalarToVectorEvent = pipe.FetchEventID<AscendC::HardEvent::S_V>();
    bool copyPending = false;
    int64_t pendingRoute = 0;
    int64_t pendingSlot = 0;
#if !TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH
    int64_t readyBatchFirstRoute = 0;
    int64_t readyBatchCount = 0;
#endif
    bool firstReadyPublished = false;

#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
    AscendC::Duplicate<int8_t>(logical, static_cast<int8_t>(0), static_cast<uint32_t>(logicalBytes));
    AscendC::Duplicate<float>(packed0, TileXR::DATA_AS_FLAG_READY_VALUE,
        static_cast<uint32_t>(routeStride / sizeof(float)));
    AscendC::Duplicate<float>(packed1, TileXR::DATA_AS_FLAG_READY_VALUE,
        static_cast<uint32_t>(routeStride / sizeof(float)));
#endif

    for (int64_t route = begin; route < end; ++route) {
        const int64_t slot = (route - begin) % kPipelineBufferCount;
        const event_t eventId = PipelineEvent(slot);
        EnqueuePackInput(expertOut, route, h, inputPadding, inputQueue);
        const uint64_t inputWaitStart = ProfileBegin(perfTrace);
        AscendC::LocalTensor<half> input = inputQueue.DeQue<half>();
        ProfileEnd(perfTrace, perfStats, PerfStage::PACK_INPUT_WAIT, inputWaitStart);
        ProfileAux(perfTrace, perfStats, PerfStage::PACK_INPUT_WAIT, 0,
            static_cast<uint64_t>(h * static_cast<int64_t>(sizeof(half))));

        const uint64_t quantizeHeadStart = ProfileBegin(perfTrace);
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
        if (slot == 0) {
            StartPackQuantization(input, packed0, quantFloat, absHalf, reduceOutHalf, reduceTmpHalf,
                logical, h, logicalBytes, routeStride, quantizeEvent);
        } else {
            StartPackQuantization(input, packed1, quantFloat, absHalf, reduceOutHalf, reduceTmpHalf,
                logical, h, logicalBytes, routeStride, quantizeEvent);
        }
#else
        if (slot == 0) {
            StartPackQuantization(input, packed0, quantFloat, absFloat, reduceOut, reduceTmp,
                logical, h, logicalBytes, routeStride, quantizeEvent);
        } else {
            StartPackQuantization(input, packed1, quantFloat, absFloat, reduceOut, reduceTmp,
                logical, h, logicalBytes, routeStride, quantizeEvent);
        }
#endif
#if TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH
        ProfileEnd(perfTrace, perfStats, PerfStage::PACK_QUANTIZE, quantizeHeadStart);
        if (copyPending) {
            FinishAndPublishPackBatch(workspaceGM, txReadyOffset, txDataOffset, routeStride,
                pendingRoute, 1, PipelineEvent(pendingSlot), readyValue,
                firstReadyStart, firstReadyPublished, perfTrace, perfStats);
            copyPending = false;
        }
        const uint64_t quantizeTailStart = ProfileBegin(perfTrace);
#endif
#if TILEXR_EP_URMA_QDC_VERSION == 1 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
        if (slot == 0) {
            FinishPackQuantization(input, packed0, quantFloat, reduceOutHalf, logical, logicalFloat,
                logicalInt, h, blockCount, repeats, quantizeEvent, scalarToVectorEvent);
        } else {
            FinishPackQuantization(input, packed1, quantFloat, reduceOutHalf, logical, logicalFloat,
                logicalInt, h, blockCount, repeats, quantizeEvent, scalarToVectorEvent);
        }
#else
        if (slot == 0) {
            FinishPackQuantization(input, packed0, quantFloat, reduceOut, logical, logicalFloat,
                logicalInt, h, blockCount, repeats, quantizeEvent, scalarToVectorEvent);
        } else {
            FinishPackQuantization(input, packed1, quantFloat, reduceOut, logical, logicalFloat,
                logicalInt, h, blockCount, repeats, quantizeEvent, scalarToVectorEvent);
        }
#endif
        inputQueue.FreeTensor<half>(input);
#if TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH
        ProfileEnd(perfTrace, perfStats, PerfStage::PACK_QUANTIZE, quantizeTailStart);
#else
        ProfileEnd(perfTrace, perfStats, PerfStage::PACK_QUANTIZE, quantizeHeadStart);
#endif
        ProfileAux(perfTrace, perfStats, PerfStage::PACK_QUANTIZE, 0, static_cast<uint64_t>(h));
        ProfileAux(perfTrace, perfStats, PerfStage::PACK_QUANTIZE, 1, static_cast<uint64_t>(blockCount));

        GM_ADDR txRouteAddr = workspaceGM + txDataOffset + route * routeStride;
        const uint64_t dataSubmitStart = ProfileBegin(perfTrace);
        if (slot == 0) {
            StartPackPublish(txRouteAddr, routeStride, packed0, eventId);
        } else {
            StartPackPublish(txRouteAddr, routeStride, packed1, eventId);
        }
        ProfileEnd(perfTrace, perfStats, PerfStage::PACK_TX_DATA_SUBMIT, dataSubmitStart);
        ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_DATA_SUBMIT, 0,
            static_cast<uint64_t>(routeStride));
#if !TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH
        if (copyPending) {
            FinishPackCopy(workspaceGM + txDataOffset + pendingRoute * routeStride,
                routeStride, PipelineEvent(pendingSlot), perfTrace, perfStats);
            if (readyBatchCount == 0) {
                readyBatchFirstRoute = pendingRoute;
            }
            ++readyBatchCount;
            if (readyBatchCount == static_cast<int64_t>(TileXREp::kEpUrmaCombineTxReadyBatchSize)) {
                const uint64_t readyPublishStart = ProfileBegin(perfTrace);
                PublishTxReadyBatch(workspaceGM, txReadyOffset, txDataOffset, routeStride,
                    readyBatchFirstRoute, readyBatchCount, readyValue, perfTrace, perfStats);
                ProfileEnd(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, readyPublishStart);
#if TILEXR_EP_URMA_TX_READY_SHARED_FLAG && TILEXR_EP_URMA_TX_READY_BATCH_SIZE > 1
                constexpr uint64_t publishedLineCount = 1;
#else
                const uint64_t publishedLineCount = static_cast<uint64_t>(readyBatchCount);
#endif
                ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 1,
                    publishedLineCount);
                ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 2,
                    static_cast<uint64_t>(readyBatchCount));
                ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 3,
                    publishedLineCount * TileXREp::kEpUrmaCombineCacheLineBytes);
                if (!firstReadyPublished) {
                    ProfileEnd(perfTrace, perfStats, PerfStage::PACK_FIRST_TX_READY, firstReadyStart);
                    ProfileAux(perfTrace, perfStats, PerfStage::PACK_FIRST_TX_READY, 0,
                        static_cast<uint64_t>(readyBatchFirstRoute));
                    firstReadyPublished = true;
                }
                readyBatchCount = 0;
            }
        }
#endif
        copyPending = true;
        pendingRoute = route;
        pendingSlot = slot;
#if !TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH
        if (route + 1 == end) {
            FinishPackCopy(txRouteAddr, routeStride, eventId, perfTrace, perfStats);
            if (readyBatchCount == 0) {
                readyBatchFirstRoute = route;
            }
            ++readyBatchCount;
            copyPending = false;
            const uint64_t readyPublishStart = ProfileBegin(perfTrace);
            PublishTxReadyBatch(workspaceGM, txReadyOffset, txDataOffset, routeStride,
                readyBatchFirstRoute, readyBatchCount, readyValue, perfTrace, perfStats);
            ProfileEnd(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, readyPublishStart);
#if TILEXR_EP_URMA_TX_READY_SHARED_FLAG && TILEXR_EP_URMA_TX_READY_BATCH_SIZE > 1
            constexpr uint64_t publishedLineCount = 1;
#else
            const uint64_t publishedLineCount = static_cast<uint64_t>(readyBatchCount);
#endif
            ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 1,
                publishedLineCount);
            ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 2,
                static_cast<uint64_t>(readyBatchCount));
            ProfileAux(perfTrace, perfStats, PerfStage::PACK_TX_PUBLISH, 3,
                publishedLineCount * TileXREp::kEpUrmaCombineCacheLineBytes);
            if (!firstReadyPublished) {
                ProfileEnd(perfTrace, perfStats, PerfStage::PACK_FIRST_TX_READY, firstReadyStart);
                ProfileAux(perfTrace, perfStats, PerfStage::PACK_FIRST_TX_READY, 0,
                    static_cast<uint64_t>(readyBatchFirstRoute));
                firstReadyPublished = true;
            }
            readyBatchCount = 0;
        }
#endif
    }
#if TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH
    if (copyPending) {
        FinishAndPublishPackBatch(workspaceGM, txReadyOffset, txDataOffset, routeStride,
            pendingRoute, 1, PipelineEvent(pendingSlot), readyValue,
            firstReadyStart, firstReadyPublished, perfTrace, perfStats);
    }
#endif
    pipe.ReleaseEventID<AscendC::HardEvent::V_S>(quantizeEvent);
    pipe.ReleaseEventID<AscendC::HardEvent::S_V>(scalarToVectorEvent);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void StartRouteReadyCopy(
    GM_ADDR routeAddr, int64_t blockCount, AscendC::LocalTensor<float> &flags,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
#if !TILEXR_EP_URMA_CACHELESS
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    for (int64_t block = 0; block < blockCount; ++block) {
        TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(routeAddr +
            block * TileXREp::kEpUrmaCombineDataBlockBytes + TileXREp::kEpUrmaCombinePayloadBytes),
            TileXREp::kEpUrmaCombineFlagBytes);
    }
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::RX_FLAG_POLL, dcciStart);
#endif
    AscendC::GlobalTensor<float> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        routeAddr + TileXREp::kEpUrmaCombinePayloadBytes));
#if TILEXR_EP_URMA_CACHELESS
    SetNoCacheRead(src);
    AscendC::DataCopyExtParams copyParams {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(sizeof(float)),
        static_cast<uint32_t>(TileXREp::kEpUrmaCombineDataBlockBytes - sizeof(float)), 0, 0};
    AscendC::DataCopyPadExtParams<float> padParams {
        true, 0, static_cast<uint8_t>(TileXR::DATA_AS_FLAG_FLAG_FLOATS - 1U),
#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
        TileXR::DATA_AS_FLAG_READY_VALUE};
#else
        0.0f};
#endif
#else
    AscendC::DataCopyExtParams copyParams {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(TileXREp::kEpUrmaCombineFlagBytes),
        static_cast<uint32_t>(TileXREp::kEpUrmaCombinePayloadBytes), 0, 0};
    AscendC::DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
#endif
    AscendC::DataCopyPad(flags, src, copyParams, padParams);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION bool CheckRouteReady(
    GM_ADDR routeAddr, int64_t blockCount, AscendC::LocalTensor<float> &flags,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    StartRouteReadyCopy(routeAddr, blockCount, flags, perfTrace, perfStats);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
#if TILEXR_EP_URMA_CACHELESS
    for (int64_t block = 0; block < blockCount; ++block) {
        if (flags.GetValue(static_cast<uint32_t>(
                block * TileXR::DATA_AS_FLAG_FLAG_FLOATS)) != TileXR::DATA_AS_FLAG_READY_VALUE) {
            return false;
        }
    }
#else
    const int64_t flagCount = blockCount * TileXR::DATA_AS_FLAG_FLAG_FLOATS;
    for (int64_t flag = 0; flag < flagCount; ++flag) {
        if (flags.GetValue(static_cast<uint32_t>(flag)) != TileXR::DATA_AS_FLAG_READY_VALUE) {
            return false;
        }
    }
#endif
    return true;
}

#if TILEXR_EP_URMA_RX_READY_BATCH_MTE2
__aicore__ TILEXR_EP_LOCAL_FUNCTION uint32_t CheckRoutesReadyBatch(
    GM_ADDR workspaceGM, int64_t rxWindowOffset,
    int64_t token, int64_t topK, int64_t blockCount, int64_t routeStride, uint32_t pendingMask,
    AscendC::LocalTensor<float> &flags,
#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
    AscendC::LocalTensor<float> &delta, AscendC::LocalTensor<float> &reduceOut,
    AscendC::LocalTensor<float> &reduceTmp,
#endif
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    const uint32_t flagCount = static_cast<uint32_t>(
        blockCount * TileXR::DATA_AS_FLAG_FLAG_FLOATS);
    uint64_t pendingRouteCount = 0;
    for (int64_t topKId = 0; topKId < topK; ++topKId) {
        const uint32_t routeBit = 1U << static_cast<uint32_t>(topKId);
        if ((pendingMask & routeBit) == 0) {
            continue;
        }
        const int64_t routeIndex = token * topK + topKId;
        GM_ADDR routeAddr = workspaceGM + rxWindowOffset + routeIndex * routeStride;
        AscendC::LocalTensor<float> routeFlags = flags[static_cast<uint32_t>(topKId) * flagCount];
        StartRouteReadyCopy(routeAddr, blockCount, routeFlags, perfTrace, perfStats);
        ++pendingRouteCount;
    }
    if (pendingRouteCount == 0) {
        return 0;
    }

#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const uint64_t mte2WaitStart = ProfileBegin(perfTrace);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
#else
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    const uint64_t mte2WaitStart = ProfileBegin(perfTrace);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
#endif
    ProfileEnd(perfTrace, perfStats, PerfStage::RX_READY_MTE2_WAIT, mte2WaitStart);
    ProfileAux(perfTrace, perfStats, PerfStage::RX_READY_MTE2_WAIT, 0, pendingRouteCount);
    ProfileAux(perfTrace, perfStats, PerfStage::RX_READY_MTE2_WAIT, 1,
        pendingRouteCount * static_cast<uint64_t>(blockCount));
    ProfileAux(perfTrace, perfStats, PerfStage::RX_READY_MTE2_WAIT, 2,
        pendingRouteCount * static_cast<uint64_t>(blockCount) * sizeof(float));

    uint32_t readyMask = 0;
#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
    const uint64_t vectorStart = ProfileBegin(perfTrace);
    constexpr uint32_t reduceResultStride = kCursorAlignment / sizeof(float);
    for (int64_t topKId = 0; topKId < topK; ++topKId) {
        const uint32_t routeBit = 1U << static_cast<uint32_t>(topKId);
        if ((pendingMask & routeBit) == 0) {
            continue;
        }
        const uint32_t routeOffset = static_cast<uint32_t>(topKId) * flagCount;
        AscendC::LocalTensor<float> routeFlags = flags[routeOffset];
        AscendC::LocalTensor<float> routeDelta = delta[routeOffset];
        AscendC::LocalTensor<float> routeReduceOut =
            reduceOut[static_cast<uint32_t>(topKId) * reduceResultStride];
        AscendC::Adds<float>(routeDelta, routeFlags, -TileXR::DATA_AS_FLAG_READY_VALUE, flagCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs<float>(routeFlags, routeDelta, flagCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax<float>(
            routeReduceOut, routeFlags, reduceTmp, static_cast<int32_t>(flagCount), false);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    ProfileEnd(perfTrace, perfStats, PerfStage::RX_READY_VECTOR, vectorStart);
    ProfileAux(perfTrace, perfStats, PerfStage::RX_READY_VECTOR, 0, pendingRouteCount);
    ProfileAux(perfTrace, perfStats, PerfStage::RX_READY_VECTOR, 1,
        pendingRouteCount * static_cast<uint64_t>(flagCount));
    for (int64_t topKId = 0; topKId < topK; ++topKId) {
        const uint32_t routeBit = 1U << static_cast<uint32_t>(topKId);
        if ((pendingMask & routeBit) != 0 &&
            reduceOut.GetValue(static_cast<uint32_t>(topKId) * reduceResultStride) == 0.0f) {
            readyMask |= routeBit;
        }
    }
#else
    for (int64_t topKId = 0; topKId < topK; ++topKId) {
        const uint32_t routeBit = 1U << static_cast<uint32_t>(topKId);
        if ((pendingMask & routeBit) == 0) {
            continue;
        }
        const uint32_t routeOffset = static_cast<uint32_t>(topKId) * flagCount;
        bool routeReady = true;
#if TILEXR_EP_URMA_CACHELESS
        for (int64_t block = 0; block < blockCount; ++block) {
            if (flags.GetValue(routeOffset + static_cast<uint32_t>(
                    block * TileXR::DATA_AS_FLAG_FLAG_FLOATS)) !=
                TileXR::DATA_AS_FLAG_READY_VALUE) {
                routeReady = false;
                break;
            }
        }
#else
        for (uint32_t flag = 0; flag < flagCount; ++flag) {
            if (flags.GetValue(routeOffset + flag) != TileXR::DATA_AS_FLAG_READY_VALUE) {
                routeReady = false;
                break;
            }
        }
#endif
        if (routeReady) {
            readyMask |= routeBit;
        }
    }
#endif
    return readyMask;
}
#endif

__aicore__ TILEXR_EP_LOCAL_FUNCTION void StartUnpackRoute(
    GM_ADDR routeAddr, int64_t blockCount, int64_t routeStride,
    AscendC::LocalTensor<uint8_t> &logical, event_t eventId,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
#if !TILEXR_EP_URMA_CACHELESS
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(routeAddr), routeStride);
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::RX_DATA, dcciStart);
#endif
    AscendC::GlobalTensor<uint8_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(routeAddr));
    SetNoCacheRead(src);
    AscendC::DataCopyExtParams copyParams {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(TileXREp::kEpUrmaCombinePayloadBytes),
        static_cast<uint32_t>(TileXREp::kEpUrmaCombineFlagBytes), 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> padParams {false, 0, 0, 0};
    AscendC::DataCopyPad(logical, src, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void WaitUnpackRoute(event_t eventId)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void StartClearRouteFlags(
    GM_ADDR routeAddr, int64_t blockCount, AscendC::LocalTensor<float> &zeroFlags)
{
    AscendC::GlobalTensor<float> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        routeAddr + TileXREp::kEpUrmaCombinePayloadBytes));
    SetNoCacheWrite(dst);
    AscendC::DataCopyExtParams copyParams {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(TileXREp::kEpUrmaCombineFlagBytes), 0,
        static_cast<uint32_t>(TileXREp::kEpUrmaCombinePayloadBytes), 0};
    AscendC::DataCopyPad(dst, zeroFlags, copyParams);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void CleanTokenRouteFlags(
    GM_ADDR firstRouteAddr, int64_t topK, int64_t blockCount, int64_t routeStride,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
#if !TILEXR_EP_URMA_CACHELESS
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    for (int64_t topKId = 0; topKId < topK; ++topKId) {
        GM_ADDR routeAddr = firstRouteAddr + topKId * routeStride;
        for (int64_t block = 0; block < blockCount; ++block) {
            TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(routeAddr +
                block * TileXREp::kEpUrmaCombineDataBlockBytes + TileXREp::kEpUrmaCombinePayloadBytes),
                TileXREp::kEpUrmaCombineFlagBytes);
        }
    }
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::RX_DATA, dcciStart);
#else
    (void)firstRouteAddr;
    (void)topK;
    (void)blockCount;
    (void)routeStride;
    (void)perfTrace;
    (void)perfStats;
#endif
}

#if TILEXR_EP_URMA_QDC_VERSION == 0 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
__aicore__ TILEXR_EP_LOCAL_FUNCTION void FinishTokenRouteFlagClears(
    GM_ADDR firstRouteAddr, int64_t topK, int64_t blockCount, int64_t routeStride,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    CleanTokenRouteFlags(firstRouteAddr, topK, blockCount, routeStride, perfTrace, perfStats);
}
#endif

__aicore__ TILEXR_EP_LOCAL_FUNCTION void ReceiveTokens(
    GM_ADDR topKWeightsGM, GM_ADDR yOutGM, GM_ADDR workspaceGM, int64_t bs,
    int64_t h, int64_t topK, int64_t magic, int64_t rxWindowOffset, int64_t blockCount, int64_t routeStride,
    int64_t rxLaneDoneOffset, int64_t errorStatusOffset, int64_t laneId, AscendC::TPipe &pipe,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    const int64_t logicalBytes = blockCount * TileXREp::kEpUrmaCombinePayloadBytes;
    const int64_t flagBytes = blockCount * TileXREp::kEpUrmaCombineFlagBytes;
    const int64_t accumBytes = AlignUpInt64(h * static_cast<int64_t>(sizeof(float)), kCursorAlignment);
    const int64_t outputBytes = AlignUpInt64(h * static_cast<int64_t>(sizeof(half)), kCursorAlignment);
    const int64_t dequantBytes = AlignUpInt64(h * static_cast<int64_t>(sizeof(float)), kCursorAlignment);
    const int64_t weightBytes = AlignUpInt64(
        topK * static_cast<int64_t>(sizeof(float)), kCursorAlignment);

    AscendC::TBuf<AscendC::QuePosition::VECCALC> logicalBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> flagBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> clearFlagBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accumBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outputBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dequantBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> weightBuf;
#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
    AscendC::TBuf<AscendC::QuePosition::VECCALC> readyDeltaBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> readyReduceOutBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> readyReduceTmpBuf;
#endif
    pipe.InitBuffer(logicalBuf, static_cast<uint32_t>(kPipelineBufferCount * logicalBytes));
#if TILEXR_EP_URMA_RX_READY_BATCH_MTE2
    pipe.InitBuffer(flagBuf, static_cast<uint32_t>(topK * flagBytes));
#else
    pipe.InitBuffer(flagBuf, static_cast<uint32_t>(flagBytes));
#endif
    pipe.InitBuffer(clearFlagBuf, static_cast<uint32_t>(flagBytes));
    pipe.InitBuffer(accumBuf, static_cast<uint32_t>(accumBytes));
    pipe.InitBuffer(outputBuf, static_cast<uint32_t>(outputBytes));
    pipe.InitBuffer(dequantBuf, static_cast<uint32_t>(dequantBytes));
    pipe.InitBuffer(weightBuf, static_cast<uint32_t>(weightBytes));
#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
    pipe.InitBuffer(readyDeltaBuf, static_cast<uint32_t>(topK * flagBytes));
    pipe.InitBuffer(readyReduceOutBuf, static_cast<uint32_t>(topK * kCursorAlignment));
    pipe.InitBuffer(readyReduceTmpBuf, static_cast<uint32_t>(flagBytes));
#endif
    AscendC::LocalTensor<uint8_t> logical0 = logicalBuf.GetWithOffset<uint8_t>(
        static_cast<uint32_t>(logicalBytes), 0);
    AscendC::LocalTensor<uint8_t> logical1 = logicalBuf.GetWithOffset<uint8_t>(
        static_cast<uint32_t>(logicalBytes), static_cast<uint32_t>(logicalBytes));
    AscendC::LocalTensor<float> logicalFloat0 = logical0.template ReinterpretCast<float>();
    AscendC::LocalTensor<float> logicalFloat1 = logical1.template ReinterpretCast<float>();
    AscendC::LocalTensor<int32_t> logicalInt0 = logical0.template ReinterpretCast<int32_t>();
    AscendC::LocalTensor<int32_t> logicalInt1 = logical1.template ReinterpretCast<int32_t>();
    AscendC::LocalTensor<int8_t> logicalInt80 = logical0.template ReinterpretCast<int8_t>();
    AscendC::LocalTensor<int8_t> logicalInt81 = logical1.template ReinterpretCast<int8_t>();
    AscendC::LocalTensor<float> flags = flagBuf.Get<float>();
    AscendC::LocalTensor<float> zeroFlags = clearFlagBuf.Get<float>();
    AscendC::LocalTensor<float> accum = accumBuf.Get<float>();
    AscendC::LocalTensor<half> output = outputBuf.Get<half>();
    AscendC::LocalTensor<float> dequant = dequantBuf.Get<float>();
    AscendC::LocalTensor<float> weight = weightBuf.Get<float>();
#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
    AscendC::LocalTensor<float> readyDelta = readyDeltaBuf.Get<float>();
    AscendC::LocalTensor<float> readyReduceOut = readyReduceOutBuf.Get<float>();
    AscendC::LocalTensor<float> readyReduceTmp = readyReduceTmpBuf.Get<float>();
#endif
    AscendC::GlobalTensor<half> yOut;
    yOut.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yOutGM), bs * h);

    const uint32_t flagCount = static_cast<uint32_t>(
        blockCount * TileXR::DATA_AS_FLAG_FLAG_FLOATS);
    AscendC::Duplicate<float>(zeroFlags, 0.0f, flagCount);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    const int64_t tokenBegin = bs * laneId / TileXREp::kEpUrmaCombinePackLaneCount;
    const int64_t tokenEnd = bs * (laneId + 1) / TileXREp::kEpUrmaCombinePackLaneCount;
    const bool profileFine = TileXR::TileXRPerfTraceEnabled(perfTrace);
    int64_t tokensRemaining = tokenEnd - tokenBegin;
#if TILEXR_EP_URMA_RX_SCHEDULER == 0
    int64_t nextSequentialToken = tokenBegin;
#else
    int64_t activeToken[kRxTokenScheduleWindow] = {};
    bool tokenActive[kRxTokenScheduleWindow] = {};
    uint32_t routeReadyMask[kRxTokenScheduleWindow] = {};
    int64_t nextToken = tokenBegin;
    for (int64_t slot = 0; slot < kRxTokenScheduleWindow && nextToken < tokenEnd; ++slot) {
        activeToken[slot] = nextToken++;
        routeReadyMask[slot] = 0;
        tokenActive[slot] = true;
    }
    int64_t nextCandidate = 0;
    const uint32_t allRoutesReadyMask = (1U << static_cast<uint32_t>(topK)) - 1U;
#endif
    while (tokensRemaining > 0) {
#if TILEXR_EP_URMA_RX_SCHEDULER == 0
        int64_t token = nextSequentialToken;
#else
        int64_t token = 0;
        int64_t selectedSlot = -1;
#endif
        uint64_t bypassedTokens = 0;
        const uint64_t flagPollStart = ProfileBegin(perfTrace);
        uint64_t pollPasses = 0;
        uint64_t routeChecks = 0;
        uint64_t readyMisses = 0;
#if TILEXR_EP_URMA_RX_SCHEDULER > 0
            while (selectedSlot < 0) {
                for (int64_t offset = 0; offset < kRxTokenScheduleWindow; ++offset) {
                    int64_t candidate = nextCandidate + offset;
                    if (candidate >= kRxTokenScheduleWindow) {
                        candidate -= kRxTokenScheduleWindow;
                    }
                    if (!tokenActive[candidate]) {
                        continue;
                    }
                    if (profileFine) {
                        ++pollPasses;
                    }
                    const int64_t candidateToken = activeToken[candidate];
                    bool candidateReady = true;
                    uint32_t readyMask = routeReadyMask[candidate];
#if TILEXR_EP_URMA_RX_READY_BATCH_MTE2
                    uint32_t pendingMask = allRoutesReadyMask;
                    if constexpr (TileXREp::kEpUrmaCombineRxStickyReady) {
                        pendingMask &= ~readyMask;
                    }
                    if (profileFine) {
                        for (int64_t topKId = 0; topKId < topK; ++topKId) {
                            if ((pendingMask & (1U << static_cast<uint32_t>(topKId))) != 0) {
                                ++routeChecks;
                            }
                        }
                    }
                    const uint32_t observedReadyMask = CheckRoutesReadyBatch(
                        workspaceGM, rxWindowOffset, candidateToken, topK, blockCount, routeStride,
                        pendingMask, flags,
#if TILEXR_EP_URMA_RX_READY_BATCH_VECTOR
                        readyDelta, readyReduceOut, readyReduceTmp,
#endif
                        perfTrace, perfStats);
                    if (profileFine) {
                        for (int64_t topKId = 0; topKId < topK; ++topKId) {
                            const uint32_t routeBit = 1U << static_cast<uint32_t>(topKId);
                            if ((pendingMask & routeBit) != 0 &&
                                (observedReadyMask & routeBit) == 0) {
                                ++readyMisses;
                            }
                        }
                    }
                    if constexpr (TileXREp::kEpUrmaCombineRxStickyReady) {
                        readyMask |= observedReadyMask;
                    } else {
                        readyMask = observedReadyMask;
                    }
                    candidateReady = readyMask == allRoutesReadyMask;
#else
                    for (int64_t topKId = 0; topKId < topK; ++topKId) {
                        const uint32_t routeBit = 1U << static_cast<uint32_t>(topKId);
                        if constexpr (TileXREp::kEpUrmaCombineRxStickyReady) {
                            if ((readyMask & routeBit) != 0) {
                                continue;
                            }
                        }
                        const int64_t routeIndex = candidateToken * topK + topKId;
                        GM_ADDR routeAddr = workspaceGM + rxWindowOffset + routeIndex * routeStride;
                        if (profileFine) {
                            ++routeChecks;
                        }
                        if (CheckRouteReady(routeAddr, blockCount, flags, perfTrace, perfStats)) {
                            if constexpr (TileXREp::kEpUrmaCombineRxStickyReady) {
                                readyMask |= routeBit;
                            }
                        } else {
                            if (profileFine) {
                                ++readyMisses;
                            }
                            candidateReady = false;
                            if constexpr (!TileXREp::kEpUrmaCombineRxStickyReady) {
                                break;
                            }
                        }
                    }
#endif
                    if constexpr (TileXREp::kEpUrmaCombineRxStickyReady) {
                        routeReadyMask[candidate] = readyMask;
                        candidateReady = readyMask == allRoutesReadyMask;
                    }
                    if (candidateReady) {
                        selectedSlot = candidate;
                        break;
                    }
                }
            }
            token = activeToken[selectedSlot];
            for (int64_t slot = 0; slot < kRxTokenScheduleWindow; ++slot) {
                if (tokenActive[slot] && activeToken[slot] < token) {
                    ++bypassedTokens;
                }
            }
            nextCandidate = selectedSlot + 1;
            if (nextCandidate >= kRxTokenScheduleWindow) {
                nextCandidate = 0;
            }
#else
            ++nextSequentialToken;
            bool tokenReady = false;
            while (!tokenReady) {
                if (profileFine) {
                    ++pollPasses;
                }
                tokenReady = true;
                for (int64_t topKId = 0; topKId < topK; ++topKId) {
                    const int64_t routeIndex = token * topK + topKId;
                    GM_ADDR routeAddr = workspaceGM + rxWindowOffset + routeIndex * routeStride;
                    if (profileFine) {
                        ++routeChecks;
                    }
                    if (!CheckRouteReady(routeAddr, blockCount, flags, perfTrace, perfStats)) {
                        if (profileFine) {
                            ++readyMisses;
                        }
                        tokenReady = false;
                        break;
                    }
                }
            }
#endif
        CachelessAcquireBarrier();
        ProfileEnd(perfTrace, perfStats, PerfStage::RX_FLAG_POLL_WAIT, flagPollStart);
        ProfileAux(perfTrace, perfStats, PerfStage::RX_FLAG_POLL_WAIT, 0, pollPasses);
        ProfileAux(perfTrace, perfStats, PerfStage::RX_FLAG_POLL_WAIT, 1, routeChecks);
        ProfileAux(perfTrace, perfStats, PerfStage::RX_FLAG_POLL_WAIT, 2, readyMisses);
        ProfileAux(perfTrace, perfStats, PerfStage::RX_FLAG_POLL_WAIT, 3,
            routeChecks * static_cast<uint64_t>(blockCount));

        LoadTopKWeights(topKWeightsGM, token * topK, topK, weight);
        if (topK > 0) {
            GM_ADDR firstRouteAddr = workspaceGM + rxWindowOffset + token * topK * routeStride;
            StartUnpackRoute(firstRouteAddr, blockCount, routeStride, logical0, PipelineEvent(0),
                perfTrace, perfStats);
        }
        for (int64_t topKId = 0; topKId < topK; ++topKId) {
            const int64_t slot = topKId % kPipelineBufferCount;
            const event_t eventId = PipelineEvent(slot);
            AscendC::LocalTensor<float> &routeLogicalFloat = slot == 0 ? logicalFloat0 : logicalFloat1;
            AscendC::LocalTensor<int32_t> &routeLogicalInt = slot == 0 ? logicalInt0 : logicalInt1;
            AscendC::LocalTensor<int8_t> &routeLogicalInt8 = slot == 0 ? logicalInt80 : logicalInt81;
            const int64_t routeIndex = token * topK + topKId;
            GM_ADDR routeAddr = workspaceGM + rxWindowOffset + routeIndex * routeStride;
            const uint64_t unpackWaitStart = ProfileBegin(perfTrace);
            WaitUnpackRoute(eventId);
            ProfileEnd(perfTrace, perfStats, PerfStage::RX_UNPACK_WAIT, unpackWaitStart);
            ProfileAux(perfTrace, perfStats, PerfStage::RX_UNPACK_WAIT, 0,
                static_cast<uint64_t>(blockCount * TileXREp::kEpUrmaCombinePayloadBytes));

            const uint64_t dequantStart = ProfileBegin(perfTrace);
            const float scale = routeLogicalFloat.GetValue(0);
            const int32_t quantMode = routeLogicalInt.GetValue(1);
            if (!(scale > 0.0f) || quantMode != TileXREp::kEpUrmaCombineQuantModeInt8PerRoute) {
                StoreError(workspaceGM, errorStatusOffset, magic,
                    TileXREp::kEpUrmaCombineStatusInvalidQuantHeader, perfTrace, perfStats);
            }
            const float effectiveScale = scale > 0.0f ? scale : 1.0f;
            const float topKWeight = weight.GetValue(static_cast<uint32_t>(topKId));
            AscendC::Cast<half, int8_t>(output,
                routeLogicalInt8[TileXREp::kEpUrmaCombineQuantHeaderBytes], AscendC::RoundMode::CAST_NONE,
                static_cast<uint32_t>(h));
            AscendC::PipeBarrier<PIPE_V>();
            if (topKId + 1 < topK) {
                const int64_t nextSlot = (slot + 1) % kPipelineBufferCount;
                GM_ADDR nextRouteAddr = routeAddr + routeStride;
                AscendC::LocalTensor<uint8_t> &nextLogical = nextSlot == 0 ? logical0 : logical1;
                StartUnpackRoute(nextRouteAddr, blockCount, routeStride,
                    nextLogical, PipelineEvent(nextSlot), perfTrace, perfStats);
            }
            AscendC::Cast<float, half>(dequant, output, AscendC::RoundMode::CAST_NONE,
                static_cast<uint32_t>(h));
            AscendC::PipeBarrier<PIPE_V>();
            if (topKId == 0) {
                AscendC::Muls<float>(accum, dequant, effectiveScale * topKWeight,
                    static_cast<uint32_t>(h));
            } else {
                AscendC::Axpy<float, float>(accum, dequant, effectiveScale * topKWeight,
                    static_cast<int32_t>(h));
            }
            AscendC::PipeBarrier<PIPE_V>();
            StartClearRouteFlags(routeAddr, blockCount, zeroFlags);
            ProfileEnd(perfTrace, perfStats, PerfStage::RX_UNPACK_DEQUANT_CLEAR, dequantStart);
            ProfileAux(perfTrace, perfStats, PerfStage::RX_UNPACK_DEQUANT_CLEAR, 0,
                static_cast<uint64_t>(h));
            ProfileAux(perfTrace, perfStats, PerfStage::RX_UNPACK_DEQUANT_CLEAR, 1,
                static_cast<uint64_t>(blockCount));
        }

#if TILEXR_EP_URMA_QDC_VERSION == 0 || TILEXR_EP_URMA_QDC_VERSION == 2 || \
    TILEXR_EP_URMA_QDC_VERSION == 3
        const uint64_t clearFenceStart = ProfileBegin(perfTrace);
        GM_ADDR firstRouteAddr = workspaceGM + rxWindowOffset + token * topK * routeStride;
        FinishTokenRouteFlagClears(firstRouteAddr, topK, blockCount, routeStride, perfTrace, perfStats);
        ProfileEnd(perfTrace, perfStats, PerfStage::RX_UNPACK_DEQUANT_CLEAR, clearFenceStart);
#endif

        const uint64_t outputStart = ProfileBegin(perfTrace);
        AscendC::Cast<half, float>(output, accum, AscendC::RoundMode::CAST_ROUND,
            static_cast<uint32_t>(h));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams outputParams {
            1, static_cast<uint32_t>(h * static_cast<int64_t>(sizeof(half))), 0, 0, 0};
        AscendC::DataCopyPad(yOut[token * h], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        ProfileEnd(perfTrace, perfStats, PerfStage::RX_OUTPUT, outputStart);
        ProfileAux(perfTrace, perfStats, PerfStage::RX_OUTPUT, 0,
            static_cast<uint64_t>(h * static_cast<int64_t>(sizeof(half))));
#if TILEXR_EP_URMA_QDC_VERSION == 1
        ProfileAux(perfTrace, perfStats, PerfStage::RX_UNPACK_DEQUANT_CLEAR, 2,
            static_cast<uint64_t>(topK * blockCount));
        GM_ADDR firstRouteAddr = workspaceGM + rxWindowOffset + token * topK * routeStride;
        CleanTokenRouteFlags(firstRouteAddr, topK, blockCount, routeStride, perfTrace, perfStats);
#endif
#if TILEXR_EP_URMA_RX_SCHEDULER > 0
        ProfileAux(perfTrace, perfStats, PerfStage::RX_OUTPUT, 1, bypassedTokens);
        ProfileAux(perfTrace, perfStats, PerfStage::RX_OUTPUT, 2, bypassedTokens == 0 ? 0U : 1U);
#endif

        --tokensRemaining;
#if TILEXR_EP_URMA_RX_SCHEDULER > 0
        routeReadyMask[selectedSlot] = 0;
        if (nextToken < tokenEnd) {
            activeToken[selectedSlot] = nextToken++;
            tokenActive[selectedSlot] = true;
        } else {
            tokenActive[selectedSlot] = false;
        }
#endif
    }

    StoreControlValue(workspaceGM + rxLaneDoneOffset +
        laneId * TileXREp::kEpUrmaCombineCacheLineBytes,
        EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxLaneDone), perfTrace, perfStats);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION void CopySelfRoute(
    GM_ADDR srcAddr, GM_ADDR dstAddr, int64_t blockCount, AscendC::LocalTensor<uint8_t> &payload,
    AscendC::LocalTensor<float> &readyFlags,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    AscendC::GlobalTensor<uint8_t> src;
    AscendC::GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(srcAddr));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dstAddr));
#if TILEXR_EP_URMA_CACHELESS
    SetNoCacheRead(src);
    SetNoCacheWrite(dst);
    AscendC::DataCopyExtParams payloadIn {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(TileXREp::kEpUrmaCombinePayloadBytes),
        static_cast<uint32_t>(TileXREp::kEpUrmaCombineFlagBytes), 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> noPad {false, 0, 0, 0};
    AscendC::DataCopyPad(payload, src, payloadIn, noPad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

    AscendC::DataCopyExtParams payloadOut {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(TileXREp::kEpUrmaCombinePayloadBytes),
        0, static_cast<uint32_t>(TileXREp::kEpUrmaCombineFlagBytes), 0};
    AscendC::DataCopyPad(dst, payload, payloadOut);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

    AscendC::GlobalTensor<float> dstFlags;
    dstFlags.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        dstAddr + TileXREp::kEpUrmaCombinePayloadBytes));
    SetNoCacheWrite(dstFlags);
    AscendC::DataCopyExtParams flagOut {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(TileXREp::kEpUrmaCombineFlagBytes),
        0, static_cast<uint32_t>(TileXREp::kEpUrmaCombinePayloadBytes), 0};
    AscendC::DataCopyPad(dstFlags, readyFlags, flagOut);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
#else
    for (int64_t block = 0; block < blockCount; ++block) {
        const int64_t offset = block * TileXREp::kEpUrmaCombineDataBlockBytes;
        AscendC::DataCopy(payload, src[offset],
            static_cast<uint32_t>(TileXREp::kEpUrmaCombineDataBlockBytes));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopy(dst[offset], payload,
            static_cast<uint32_t>(TileXREp::kEpUrmaCombineDataBlockBytes));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    const uint64_t dcciStart = ProfileBegin(perfTrace);
    TileXR::UDMACleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(dstAddr),
        blockCount * TileXREp::kEpUrmaCombineDataBlockBytes);
    ProfileDcciEnd(perfTrace, perfStats, DcciProfileCategory::TX_DATA, dcciStart);
#endif
}

__aicore__ TILEXR_EP_SEND_FUNCTION bool SendRoutes(const __gm__ TileXR::CommArgs *args, GM_ADDR assistInfoGM,
    GM_ADDR workspaceGM, int64_t selfSendCnt, int64_t bs, int64_t topK, int64_t magic, int64_t rxWindowOffset,
    int64_t blockCount, int64_t routeStride, int64_t txReadyOffset, int64_t txDataOffset,
    int64_t senderDoneOffset, int64_t errorStatusOffset, int64_t senderId, AscendC::TPipe &pipe,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    const int64_t cursorBytes = AlignUpInt64(
        TileXREp::kEpUrmaCombinePackLaneCount * static_cast<int64_t>(sizeof(uint32_t)), kCursorAlignment);
    const int64_t usedPeerBytes = AlignUpInt64(args->rankSize, kCursorAlignment);
    const int64_t selfCopyPayloadBytes = blockCount * TileXREp::kEpUrmaCombinePayloadBytes;
    const int64_t selfCopyFlagBytes = blockCount * TileXREp::kEpUrmaCombineFlagBytes;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> cursorBuf;
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL || !TILEXR_EP_URMA_CACHELESS
    AscendC::TBuf<AscendC::QuePosition::VECCALC> metaBuf;
#endif
    AscendC::TBuf<AscendC::QuePosition::VECCALC> usedPeerBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> selfCopyBuf;
    pipe.InitBuffer(cursorBuf, static_cast<uint32_t>(cursorBytes));
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
    pipe.InitBuffer(metaBuf, kRouteMetaPrefetchCapacityBytes);
#elif !TILEXR_EP_URMA_CACHELESS
    pipe.InitBuffer(metaBuf, kRouteMetaBufferBytes);
#endif
    pipe.InitBuffer(usedPeerBuf, static_cast<uint32_t>(usedPeerBytes));
    pipe.InitBuffer(selfCopyBuf,
        static_cast<uint32_t>(selfCopyPayloadBytes + selfCopyFlagBytes));
    AscendC::LocalTensor<uint32_t> cursor = cursorBuf.Get<uint32_t>();
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL || !TILEXR_EP_URMA_CACHELESS
    AscendC::LocalTensor<int32_t> meta = metaBuf.Get<int32_t>();
#endif
    AscendC::LocalTensor<uint8_t> usedPeer = usedPeerBuf.Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> selfCopyPayload = selfCopyBuf.GetWithOffset<uint8_t>(
        static_cast<uint32_t>(selfCopyPayloadBytes), 0);
    AscendC::LocalTensor<float> selfCopyReady = selfCopyBuf.GetWithOffset<float>(
        static_cast<uint32_t>(selfCopyFlagBytes / sizeof(float)),
        static_cast<uint32_t>(selfCopyPayloadBytes));
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
    const bool useRouteMetaPrefetch = selfSendCnt > 0 && selfSendCnt <=
        static_cast<int64_t>(kRouteMetaPrefetchCapacityBytes / sizeof(TileXREp::EpAssistTuple));
    const int64_t routeMetaBytes = useRouteMetaPrefetch ?
        selfSendCnt * static_cast<int64_t>(sizeof(TileXREp::EpAssistTuple)) : 0;
    bool routeMetaPrefetchPending = false;
    if (useRouteMetaPrefetch) {
        StartRouteMetaPrefetch(assistInfoGM, selfSendCnt, meta, EVENT_ID1);
        routeMetaPrefetchPending = true;
        ProfileAux(perfTrace, perfStats, PerfStage::TX_META_SCAN, 3,
            static_cast<uint64_t>(routeMetaBytes));
    }
#endif
#if TILEXR_EP_URMA_CACHELESS
    AscendC::Duplicate<float>(selfCopyReady, TileXR::DATA_AS_FLAG_READY_VALUE,
        static_cast<uint32_t>(selfCopyFlagBytes / sizeof(float)));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
#endif

    for (int64_t lane = 0; lane < TileXREp::kEpUrmaCombinePackLaneCount; ++lane) {
        const int64_t begin = selfSendCnt * lane / TileXREp::kEpUrmaCombinePackLaneCount;
        const int64_t end = selfSendCnt * (lane + 1) / TileXREp::kEpUrmaCombinePackLaneCount;
        const int64_t laneLength = end - begin;
        const int64_t remainder = laneLength % TileXREp::kEpUrmaCombineSendLaneCount;
        const bool reverse = (lane & 1) != 0;

        // Preserve the globally balanced residue set, but reverse odd lanes so no sender
        // consistently waits for the last route produced by every Pack core.
        int64_t rotation = lane % TileXREp::kEpUrmaCombineSendLaneCount;
        if (remainder != 0) {
            rotation = begin % TileXREp::kEpUrmaCombineSendLaneCount;
            if (reverse) {
                rotation = (rotation + remainder - 1) % TileXREp::kEpUrmaCombineSendLaneCount;
            }
        }
        const int64_t signedDelta = reverse ? rotation - senderId : senderId - rotation;
        const int64_t delta = (signedDelta + TileXREp::kEpUrmaCombineSendLaneCount) %
            TileXREp::kEpUrmaCombineSendLaneCount;
        cursor.SetValue(static_cast<uint32_t>(lane), static_cast<uint32_t>(begin + delta));
    }

    const int32_t rank = args->rank;
    const int32_t rankSize = args->rankSize;
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        usedPeer.SetValue(static_cast<uint32_t>(peer), 0);
    }
    const uint64_t readyValue = EncodeControlValue(magic, TileXREp::kEpUrmaCombineTxRouteReady);
#if TILEXR_EP_URMA_DOORBELL_BATCH_SIZE > 1
    TileXR::UDMADoorbellBatchGroup doorbellBatch = {};
#else
    uint64_t remotePutCount = 0;
#endif
    bool allScanned = false;
    while (!allScanned) {
        allScanned = true;
        for (int64_t lane = 0; lane < TileXREp::kEpUrmaCombinePackLaneCount; ++lane) {
            const int64_t begin = selfSendCnt * lane / TileXREp::kEpUrmaCombinePackLaneCount;
            const int64_t end = selfSendCnt * (lane + 1) / TileXREp::kEpUrmaCombinePackLaneCount;
            const int64_t route = cursor.GetValue(static_cast<uint32_t>(lane));
            if (route >= end) {
                continue;
            }
            allScanned = false;
            const int64_t readyRoute = TxReadyPollRoute(route, begin);
#if TILEXR_EP_URMA_TX_READY_IN_DATA
            GM_ADDR readyAddr = workspaceGM + txDataOffset + readyRoute * routeStride;
#else
            GM_ADDR readyAddr = workspaceGM + txReadyOffset +
                readyRoute * TileXREp::kEpUrmaCombineCacheLineBytes;
#endif
            const uint64_t readyPollStart = ProfileBegin(perfTrace);
#if TILEXR_EP_URMA_TX_READY_IN_DATA
            const bool routeReady = LoadTxReadyInData(
                readyAddr, perfTrace, perfStats) == readyValue;
#else
            const bool routeReady = LoadControlValue(readyAddr, perfTrace, perfStats) == readyValue;
#endif
            ProfileEnd(perfTrace, perfStats, PerfStage::TX_READY_POLL, readyPollStart);
            if (!routeReady) {
                ProfileAux(perfTrace, perfStats, PerfStage::TX_READY_POLL, 0, 1);
                continue;
            }
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
            if (!useRouteMetaPrefetch) {
                CachelessAcquireBarrier();
            }
#else
            CachelessAcquireBarrier();
#endif
            ProfileAux(perfTrace, perfStats, PerfStage::TX_READY_POLL, 1, 1);

            const uint64_t metaScanStart = ProfileBegin(perfTrace);
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
            TileXREp::EpAssistTuple tuple;
            if (useRouteMetaPrefetch) {
                if (routeMetaPrefetchPending) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
                    routeMetaPrefetchPending = false;
                }
                tuple = LoadPrefetchedRouteMeta(meta, route);
            } else {
                tuple = LoadRouteMetaBypass(assistInfoGM, route);
            }
#elif TILEXR_EP_URMA_CACHELESS
            const TileXREp::EpAssistTuple tuple = LoadRouteMeta(assistInfoGM, route);
#else
            const TileXREp::EpAssistTuple tuple = LoadRouteMeta(assistInfoGM, route, meta);
#endif
            if (!RouteMetaValid(tuple, rankSize, bs, topK)) {
                ProfileEnd(perfTrace, perfStats, PerfStage::TX_META_SCAN, metaScanStart);
                ProfileAux(perfTrace, perfStats, PerfStage::TX_META_SCAN, 0, 1);
                StoreError(workspaceGM, errorStatusOffset, magic,
                    TileXREp::kEpUrmaCombineStatusInvalidRoute, perfTrace, perfStats);
                cursor.SetValue(static_cast<uint32_t>(lane), static_cast<uint32_t>(
                    route + TileXREp::kEpUrmaCombineSendLaneCount));
                continue;
            }
            ProfileEnd(perfTrace, perfStats, PerfStage::TX_META_SCAN, metaScanStart);
            ProfileAux(perfTrace, perfStats, PerfStage::TX_META_SCAN,
                tuple.srcRank == rank ? 1U : 2U, 1);
#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
            if (useRouteMetaPrefetch) {
                CachelessAcquireBarrier();
            }
#endif

            const int64_t routeIndex = static_cast<int64_t>(tuple.tokenId) * topK + tuple.topKId;
            const int64_t rxRouteOffset = rxWindowOffset + routeIndex * routeStride;
            GM_ADDR txRouteAddr = workspaceGM + txDataOffset + route * routeStride;
            if (tuple.srcRank == rank) {
                const uint64_t selfCopyStart = ProfileBegin(perfTrace);
                CopySelfRoute(txRouteAddr, workspaceGM + rxRouteOffset, blockCount, selfCopyPayload,
                    selfCopyReady,
                    perfTrace, perfStats);
                ProfileEnd(perfTrace, perfStats, PerfStage::SELF_COPY, selfCopyStart);
                ProfileAux(perfTrace, perfStats, PerfStage::SELF_COPY, 0,
                    static_cast<uint64_t>(routeStride));
            } else {
                const uint64_t postStart = ProfileBegin(perfTrace);
#if TILEXR_EP_URMA_DOORBELL_BATCH_SIZE == 1
                    TileXREp::EpUrmaUDMAPutNbi<uint8_t>(args, tuple.srcRank,
                        reinterpret_cast<__gm__ uint8_t *>(txRouteAddr),
                        static_cast<uint64_t>(rxRouteOffset), static_cast<uint32_t>(routeStride),
                        static_cast<uint32_t>(senderId));
                    ++remotePutCount;
#else
                    TileXR::UDMADoorbellBatchState *batchState = TileXR::UDMAGetDoorbellBatchState(
                        args, static_cast<uint32_t>(tuple.srcRank), static_cast<uint32_t>(senderId),
                        &doorbellBatch);
                    if (batchState == nullptr) {
                        TileXREp::EpUrmaUDMAPutNbi<uint8_t>(args, tuple.srcRank,
                            reinterpret_cast<__gm__ uint8_t *>(txRouteAddr),
                            static_cast<uint64_t>(rxRouteOffset), static_cast<uint32_t>(routeStride),
                            static_cast<uint32_t>(senderId));
                        ++doorbellBatch.fallbackDoorbellCount;
                    } else {
                        TileXR::UDMAPutNbiDoorbellBatched<uint8_t>(args, tuple.srcRank,
                            reinterpret_cast<__gm__ uint8_t *>(txRouteAddr),
                            static_cast<uint64_t>(rxRouteOffset), static_cast<uint32_t>(routeStride),
                            static_cast<uint32_t>(senderId), TileXREp::kEpUrmaCombineDoorbellBatchSize,
                            batchState);
                    }
#endif
                usedPeer.SetValue(static_cast<uint32_t>(tuple.srcRank), 1);
                ProfileEnd(perfTrace, perfStats, PerfStage::UDMA_POST, postStart);
                ProfileAux(perfTrace, perfStats, PerfStage::UDMA_POST, 0,
                    static_cast<uint64_t>(routeStride));
            }
            cursor.SetValue(static_cast<uint32_t>(lane), static_cast<uint32_t>(
                route + TileXREp::kEpUrmaCombineSendLaneCount));
        }
    }

#if TILEXR_EP_URMA_TX_META_PREFETCH_FULL
    if (routeMetaPrefetchPending) {
        const uint64_t metaDrainStart = ProfileBegin(perfTrace);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        ProfileEnd(perfTrace, perfStats, PerfStage::TX_META_SCAN, metaDrainStart);
        routeMetaPrefetchPending = false;
    }
#endif

#if TILEXR_EP_URMA_DOORBELL_BATCH_SIZE > 1
    TileXR::UDMAFlushDoorbellBatchGroup(&doorbellBatch);
    const uint64_t doorbellCommitCount = static_cast<uint64_t>(
        TileXR::UDMADoorbellBatchGroupCommitCount(&doorbellBatch));
    const uint64_t activeSendSqCount = static_cast<uint64_t>(doorbellBatch.activeQueueCount);
#else
    const uint64_t doorbellCommitCount = remotePutCount;
    const uint64_t activeSendSqCount = 0;
#endif
    ProfileAux(perfTrace, perfStats, PerfStage::UDMA_POST, 1,
        doorbellCommitCount);
    ProfileAux(perfTrace, perfStats, PerfStage::UDMA_POST, 2,
        static_cast<uint64_t>(TileXREp::kEpUrmaCombineDoorbellBatchSize));
    ProfileAux(perfTrace, perfStats, PerfStage::UDMA_POST, 3,
        activeSendSqCount);

    __gm__ TileXR::UDMAInfo *udmaInfo = TileXR::GetUDMAInfo(args);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (usedPeer.GetValue(static_cast<uint32_t>(peer)) == 0) {
            continue;
        }
        const uint64_t wqeCntAddr = TileXR::UDMAGetWQCtx(
            udmaInfo, static_cast<uint32_t>(peer), static_cast<uint32_t>(senderId))->wqeCntAddr;
        bool queueAlreadyDrained = false;
        for (int32_t previous = 0; previous < peer; ++previous) {
            if (usedPeer.GetValue(static_cast<uint32_t>(previous)) != 0 &&
                TileXR::UDMAGetWQCtx(udmaInfo, static_cast<uint32_t>(previous),
                    static_cast<uint32_t>(senderId))->wqeCntAddr == wqeCntAddr) {
                queueAlreadyDrained = true;
                break;
            }
        }
        if (queueAlreadyDrained) {
            continue;
        }
        const uint64_t quietStart = ProfileBegin(perfTrace);
        (void)TileXREp::EpUrmaUDMAQuiet(args, peer, static_cast<uint32_t>(senderId));
        ProfileEnd(perfTrace, perfStats, PerfStage::UDMA_QUIET, quietStart);
        ProfileAux(perfTrace, perfStats, PerfStage::UDMA_QUIET, 0, 1);
    }
    StoreControlValue(workspaceGM + senderDoneOffset +
        senderId * TileXREp::kEpUrmaCombineCacheLineBytes,
        EncodeControlValue(magic, TileXREp::kEpUrmaCombineSenderDone), perfTrace, perfStats);
    return true;
}

#undef TILEXR_EP_SEND_FUNCTION

__aicore__ TILEXR_EP_LOCAL_FUNCTION bool WaitLocalLines(
    GM_ADDR base, int64_t lineCount, uint64_t expected,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    for (int64_t line = 0; line < lineCount; ++line) {
        GM_ADDR lineAddr = base + line * TileXREp::kEpUrmaCombineCacheLineBytes;
        while (LoadControlValue(lineAddr, perfTrace, perfStats) != expected) {
        }
    }
    CachelessAcquireBarrier();
    return true;
}

#if TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
__aicore__ TILEXR_EP_LOCAL_FUNCTION bool ReleaseGenerationReached(
    uint64_t observed, uint32_t expectedGeneration)
{
    const uint32_t observedStep = static_cast<uint32_t>(observed);
    const uint32_t observedGeneration = static_cast<uint32_t>(observed >> 32U);
    return observedStep == TileXREp::kEpUrmaCombineRxBufferReleased &&
        static_cast<int32_t>(observedGeneration - expectedGeneration) >= 0;
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION bool WaitDeferredRoundCredit(
    const __gm__ TileXR::CommArgs *args, GM_ADDR workspaceGM, int64_t magic,
    int64_t roundDoneOffset, int64_t senderDoneOffset, int64_t roundCreditOffset,
    int64_t senderId,
    GM_ADDR perfTrace, GM_ADDR finePerfTrace, __ubuf__ PerfStats *perfStats)
{
    const uint64_t globalWaitStart = ProfileBegin(perfTrace);
    const int32_t rank = args->rank;
    const int32_t rankSize = args->rankSize;
    const int64_t creditLaneCount = rankSize < TileXREp::kEpUrmaCombineSendLaneCount ?
        rankSize : TileXREp::kEpUrmaCombineSendLaneCount;
    uint64_t pollLoads = 0;
    uint32_t expectedGeneration = 0;
    if (senderId < creditLaneCount) {
        const uint64_t previousRelease = LoadControlValue(workspaceGM + roundDoneOffset +
            rank * TileXREp::kEpUrmaCombineCacheLineBytes, finePerfTrace, perfStats);
        const uint32_t previousStep = static_cast<uint32_t>(previousRelease);
        expectedGeneration = static_cast<uint32_t>(previousRelease >> 32U);
        if (previousRelease != 0 && previousStep != TileXREp::kEpUrmaCombineRxBufferReleased) {
            return false;
        }
        for (int32_t peer = static_cast<int32_t>(senderId); peer < rankSize;
             peer += static_cast<int32_t>(creditLaneCount)) {
            if (peer == rank || previousRelease == 0) {
                continue;
            }
            // The same-parity roundDone line is the previous release source. Complete its
            // QP before this launch can eventually overwrite that line with the next release.
            (void)TileXREp::EpUrmaUDMAQuiet(
                args, peer, static_cast<uint32_t>(senderId));
            GM_ADDR peerRelease = workspaceGM + roundDoneOffset +
                peer * TileXREp::kEpUrmaCombineCacheLineBytes;
            while (true) {
                ++pollLoads;
                if (ReleaseGenerationReached(
                        LoadControlValue(peerRelease, finePerfTrace, perfStats), expectedGeneration)) {
                    break;
                }
            }
        }
        StoreControlValue(workspaceGM + senderDoneOffset +
            senderId * TileXREp::kEpUrmaCombineCacheLineBytes,
            EncodeControlValue(magic, TileXREp::kEpUrmaCombineCreditShardDone),
            finePerfTrace, perfStats);
    }
    if (senderId == 0) {
        if (!WaitLocalLines(workspaceGM + senderDoneOffset, creditLaneCount,
                EncodeControlValue(magic, TileXREp::kEpUrmaCombineCreditShardDone),
                finePerfTrace, perfStats)) {
            return false;
        }
        StoreControlValue(workspaceGM + roundCreditOffset,
            EncodeControlValue(magic, TileXREp::kEpUrmaCombineCreditRun),
            finePerfTrace, perfStats);
    }
    if (!WaitLocalLines(workspaceGM + roundCreditOffset, 1,
            EncodeControlValue(magic, TileXREp::kEpUrmaCombineCreditRun),
            finePerfTrace, perfStats)) {
        return false;
    }
    ProfileEnd(perfTrace, perfStats, PerfStage::GLOBAL_ROUND_WAIT, globalWaitStart);
    ProfileAux(perfTrace, perfStats, PerfStage::GLOBAL_ROUND_WAIT, 0, pollLoads);
    ProfileAux(perfTrace, perfStats, PerfStage::GLOBAL_ROUND_WAIT, 1, expectedGeneration);
    return true;
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION bool WaitLocalLineShard(
    GM_ADDR base, int64_t lineCount, int64_t firstLine, int64_t lineStride,
    uint64_t expected, GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    for (int64_t line = firstLine; line < lineCount; line += lineStride) {
        GM_ADDR lineAddr = base + line * TileXREp::kEpUrmaCombineCacheLineBytes;
        while (LoadControlValue(lineAddr, perfTrace, perfStats) != expected) {
        }
    }
    CachelessAcquireBarrier();
    return true;
}
#endif

#if TILEXR_EP_URMA_START_GATE
__aicore__ TILEXR_EP_LOCAL_FUNCTION bool WaitForSynchronizedStart(
    const __gm__ TileXR::CommArgs *args,
    GM_ADDR workspaceGM, int64_t magic, int64_t rxLaneDoneOffset, int64_t senderDoneOffset,
    int64_t roundPublishOffset, int64_t startGateOffset, int64_t blockIdx,
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats)
{
    const uint64_t gateStart = ProfileBegin(perfTrace);
    const uint64_t localReady = EncodeControlValue(
        magic, TileXREp::kEpUrmaCombineStartLocalReady);
    const bool isPackReceive = blockIdx < TileXREp::kEpUrmaCombinePackLaneCount;
    if (isPackReceive) {
        StoreControlValue(workspaceGM + rxLaneDoneOffset +
            blockIdx * TileXREp::kEpUrmaCombineCacheLineBytes,
            localReady, nullptr, perfStats);
    } else {
        const int64_t senderId = blockIdx - TileXREp::kEpUrmaCombinePackLaneCount;
        StoreControlValue(workspaceGM + senderDoneOffset +
            senderId * TileXREp::kEpUrmaCombineCacheLineBytes,
            localReady, nullptr, perfStats);
    }

    const uint64_t startRun = EncodeControlValue(magic, TileXREp::kEpUrmaCombineStartRun);
    if (isPackReceive) {
        if (!WaitLocalLines(workspaceGM + roundPublishOffset, 1, startRun,
                nullptr, perfStats)) {
            return false;
        }
        ProfileEnd(perfTrace, perfStats, PerfStage::START_GATE, gateStart);
        return true;
    }

    const int64_t senderId = blockIdx - TileXREp::kEpUrmaCombinePackLaneCount;
    const int32_t rank = args->rank;
    const int32_t rankSize = args->rankSize;
    const uint64_t rankReady = EncodeControlValue(
        magic, TileXREp::kEpUrmaCombineStartRankReady);
    GM_ADDR rankReadyAddr = workspaceGM + startGateOffset +
        rank * TileXREp::kEpUrmaCombineCacheLineBytes;
    if (senderId == 0) {
        if (!WaitLocalLines(workspaceGM + rxLaneDoneOffset, TileXREp::kEpUrmaCombinePackLaneCount,
                localReady, nullptr, perfStats) ||
            !WaitLocalLines(workspaceGM + senderDoneOffset, TileXREp::kEpUrmaCombineSendLaneCount,
                localReady, nullptr, perfStats)) {
            return false;
        }
        StoreControlValue(rankReadyAddr, rankReady, nullptr, perfStats);
        StoreControlValue(workspaceGM + roundPublishOffset, rankReady, nullptr, perfStats);
    }
    if (!WaitLocalLines(workspaceGM + roundPublishOffset, 1, rankReady, nullptr, perfStats)) {
        return false;
    }

    const uint32_t qpIdx = static_cast<uint32_t>(senderId);
    const uint64_t remoteOffset = static_cast<uint64_t>(startGateOffset +
        rank * TileXREp::kEpUrmaCombineCacheLineBytes);
    uint64_t publishCount = 0;
    for (int32_t peer = static_cast<int32_t>(senderId); peer < rankSize;
         peer += static_cast<int32_t>(TileXREp::kEpUrmaCombineSendLaneCount)) {
        if (peer == rank) {
            continue;
        }
        TileXREp::EpUrmaUDMAPutNbi<uint8_t>(args, peer,
            reinterpret_cast<__gm__ uint8_t *>(rankReadyAddr), remoteOffset,
            static_cast<uint32_t>(TileXREp::kEpUrmaCombineCacheLineBytes), qpIdx);
        ++publishCount;
    }
    for (int32_t peer = static_cast<int32_t>(senderId); peer < rankSize;
         peer += static_cast<int32_t>(TileXREp::kEpUrmaCombineSendLaneCount)) {
        if (peer == rank) {
            continue;
        }
        (void)TileXREp::EpUrmaUDMAQuiet(args, peer, qpIdx);
    }
    ProfileAux(perfTrace, perfStats, PerfStage::START_GATE, 0, publishCount);
    ProfileAux(perfTrace, perfStats, PerfStage::START_GATE, 1,
        publishCount * TileXREp::kEpUrmaCombineCacheLineBytes);
    ProfileAux(perfTrace, perfStats, PerfStage::START_GATE, 2, publishCount);
    ProfileAux(perfTrace, perfStats, PerfStage::START_GATE, 3, publishCount);
    StoreControlValue(workspaceGM + senderDoneOffset +
        senderId * TileXREp::kEpUrmaCombineCacheLineBytes,
        EncodeControlValue(magic, TileXREp::kEpUrmaCombineStartPublishDone), nullptr, perfStats);

    if (senderId == 0) {
        if (!WaitLocalLines(workspaceGM + senderDoneOffset, TileXREp::kEpUrmaCombineSendLaneCount,
                EncodeControlValue(magic, TileXREp::kEpUrmaCombineStartPublishDone), nullptr, perfStats) ||
            !WaitLocalLines(workspaceGM + startGateOffset, rankSize, rankReady,
                nullptr, perfStats)) {
            return false;
        }
        StoreControlValue(workspaceGM + roundPublishOffset, startRun, nullptr, perfStats);
    }
    if (!WaitLocalLines(workspaceGM + roundPublishOffset, 1, startRun, nullptr, perfStats)) {
        return false;
    }
    ProfileEnd(perfTrace, perfStats, PerfStage::START_GATE, gateStart);
    return true;
}
#endif

__aicore__ TILEXR_EP_LOCAL_FUNCTION void RecordRoundPublishCounters(
    GM_ADDR perfTrace, __ubuf__ PerfStats *perfStats, uint64_t publishCount)
{
    ProfileAux(perfTrace, perfStats, PerfStage::ROUND_PUBLISH, 0, publishCount);
    ProfileAux(perfTrace, perfStats, PerfStage::ROUND_PUBLISH, 1,
        publishCount * TileXREp::kEpUrmaCombineCacheLineBytes);
    ProfileAux(perfTrace, perfStats, PerfStage::ROUND_PUBLISH, 2, publishCount);
    ProfileAux(perfTrace, perfStats, PerfStage::ROUND_PUBLISH, 3, publishCount);
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION bool WaitGlobalRoundDone(
    GM_ADDR workspaceGM, int32_t rankSize,
    int64_t magic, int64_t roundDoneOffset,
    GM_ADDR perfTrace, GM_ADDR finePerfTrace, __ubuf__ PerfStats *perfStats)
{
    const uint64_t globalWaitStart = ProfileBegin(perfTrace);
    uint64_t pollLoads = 0;
    const uint64_t expected = EncodeControlValue(
        magic, TileXREp::kEpUrmaCombineRxBufferReleased);
    for (int32_t peer = 0; peer < rankSize; ++peer) {
        GM_ADDR lineAddr = workspaceGM + roundDoneOffset +
            peer * TileXREp::kEpUrmaCombineCacheLineBytes;
        while (true) {
            ++pollLoads;
            if (LoadControlValue(lineAddr, finePerfTrace, perfStats) == expected) {
                break;
            }
        }
    }
    CachelessAcquireBarrier();
    ProfileEnd(perfTrace, perfStats, PerfStage::GLOBAL_ROUND_WAIT, globalWaitStart);
    ProfileAux(perfTrace, perfStats, PerfStage::GLOBAL_ROUND_WAIT, 0, pollLoads);
    return true;
}

#if TILEXR_EP_URMA_PARALLEL_ROUND_PUBLISH
__aicore__ TILEXR_EP_LOCAL_FUNCTION uint64_t StartParallelRoundPublish(
    const __gm__ TileXR::CommArgs *args,
    GM_ADDR workspaceGM, int64_t magic, int64_t roundDoneOffset, int64_t roundPublishOffset,
    GM_ADDR perfTrace, GM_ADDR finePerfTrace, __ubuf__ PerfStats *perfStats)
{
    const uint64_t publishStart = ProfileBegin(perfTrace);
    const uint64_t roundValue = EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxBufferReleased);
    StoreControlValue(workspaceGM + roundDoneOffset +
        args->rank * TileXREp::kEpUrmaCombineCacheLineBytes,
        roundValue, finePerfTrace, perfStats);
#if !TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
    // This release store is also the shared 64-byte UDMA source and the local shard start flag.
    StoreControlValue(workspaceGM + roundPublishOffset, roundValue, finePerfTrace, perfStats);
#else
    (void)roundPublishOffset;
#endif
    return publishStart;
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION uint64_t PublishRoundShard(
    const __gm__ TileXR::CommArgs *args,
    GM_ADDR workspaceGM, int64_t magic, int64_t roundDoneOffset, int64_t roundPublishOffset,
    int64_t senderDoneOffset, int64_t senderId,
    GM_ADDR perfTrace, GM_ADDR finePerfTrace, __ubuf__ PerfStats *perfStats)
{
    const int32_t rank = args->rank;
    const int32_t rankSize = args->rankSize;
    const uint64_t roundValue = EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxBufferReleased);
    GM_ADDR publishAddr = workspaceGM + roundPublishOffset;
#if TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
    publishAddr = workspaceGM + roundDoneOffset +
        rank * TileXREp::kEpUrmaCombineCacheLineBytes;
#endif
    if (!WaitLocalLines(publishAddr, 1, roundValue, finePerfTrace, perfStats)) {
        return ~0ULL;
    }

    const uint64_t shardStart = senderId == 0 ? 0 : ProfileBegin(perfTrace);
    const uint32_t qpIdx = static_cast<uint32_t>(senderId);
    const uint64_t remoteOffset = static_cast<uint64_t>(roundDoneOffset +
        rank * TileXREp::kEpUrmaCombineCacheLineBytes);
    uint64_t publishCount = 0;
    for (int32_t peer = static_cast<int32_t>(senderId); peer < rankSize;
         peer += static_cast<int32_t>(TileXREp::kEpUrmaCombineSendLaneCount)) {
        if (peer == rank) {
            continue;
        }
        TileXREp::EpUrmaUDMAPutNbi<uint8_t>(args, peer,
            reinterpret_cast<__gm__ uint8_t *>(publishAddr), remoteOffset,
            static_cast<uint32_t>(TileXREp::kEpUrmaCombineCacheLineBytes), qpIdx);
        ++publishCount;
    }
#if !TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
    for (int32_t peer = static_cast<int32_t>(senderId); peer < rankSize;
         peer += static_cast<int32_t>(TileXREp::kEpUrmaCombineSendLaneCount)) {
        if (peer == rank) {
            continue;
        }
        (void)TileXREp::EpUrmaUDMAQuiet(args, peer, qpIdx);
    }
    StoreControlValue(workspaceGM + senderDoneOffset +
        senderId * TileXREp::kEpUrmaCombineCacheLineBytes,
        EncodeControlValue(magic, TileXREp::kEpUrmaCombinePublishDone), finePerfTrace, perfStats);
#endif
    if (senderId != 0) {
        ProfileEnd(perfTrace, perfStats, PerfStage::ROUND_PUBLISH, shardStart);
        RecordRoundPublishCounters(perfTrace, perfStats, publishCount);
    }
    return publishCount;
}

__aicore__ TILEXR_EP_LOCAL_FUNCTION bool FinishParallelRoundPublish(
    GM_ADDR workspaceGM,
    int64_t magic, int64_t senderDoneOffset, int64_t publisherCount,
    uint64_t publishStart, uint64_t sender0PublishCount,
    GM_ADDR perfTrace, GM_ADDR finePerfTrace, __ubuf__ PerfStats *perfStats)
{
#if TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
    (void)workspaceGM;
    (void)magic;
    (void)senderDoneOffset;
    (void)publisherCount;
    (void)finePerfTrace;
#else
    if (!WaitLocalLines(workspaceGM + senderDoneOffset, publisherCount,
            EncodeControlValue(magic, TileXREp::kEpUrmaCombinePublishDone), finePerfTrace, perfStats)) {
        return false;
    }
#endif
    ProfileEnd(perfTrace, perfStats, PerfStage::ROUND_PUBLISH, publishStart);
    RecordRoundPublishCounters(perfTrace, perfStats, sender0PublishCount);
    return true;
}
#else
__aicore__ TILEXR_EP_LOCAL_FUNCTION void PublishAndWaitRoundDone(
    const __gm__ TileXR::CommArgs *args, GM_ADDR workspaceGM,
    int64_t magic, int64_t roundDoneOffset, int64_t roundPublishOffset,
    GM_ADDR perfTrace, GM_ADDR finePerfTrace, __ubuf__ PerfStats *perfStats)
{
    const uint64_t publishStart = ProfileBegin(perfTrace);
    const int32_t rank = args->rank;
    const int32_t rankSize = args->rankSize;
    const uint64_t roundValue = EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxBufferReleased);
    GM_ADDR publishAddr = workspaceGM + roundPublishOffset;
    StoreControlValue(publishAddr, roundValue, finePerfTrace, perfStats);
    StoreControlValue(workspaceGM + roundDoneOffset +
        rank * TileXREp::kEpUrmaCombineCacheLineBytes, roundValue, finePerfTrace, perfStats);

    for (int32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rank) {
            continue;
        }
        const uint64_t remoteOffset = static_cast<uint64_t>(roundDoneOffset +
            rank * TileXREp::kEpUrmaCombineCacheLineBytes);
        TileXREp::EpUrmaUDMAPutNbi<uint8_t>(args, peer,
            reinterpret_cast<__gm__ uint8_t *>(publishAddr), remoteOffset,
            static_cast<uint32_t>(TileXREp::kEpUrmaCombineCacheLineBytes));
        TileXREp::EpUrmaUDMAQuiet(args, peer);
    }
    ProfileEnd(perfTrace, perfStats, PerfStage::ROUND_PUBLISH, publishStart);
    RecordRoundPublishCounters(perfTrace, perfStats, static_cast<uint64_t>(rankSize - 1));
    WaitGlobalRoundDone(workspaceGM, rankSize, magic, roundDoneOffset,
        perfTrace, finePerfTrace, perfStats);
}
#endif

} // namespace

extern "C" __global__ __aicore__ void tilexr_ep_urma_combine_kernel(GM_ADDR commArgsGM,
    GM_ADDR expertOutGM, GM_ADDR assistInfoForCombineGM, GM_ADDR topKWeightsGM, GM_ADDR yOutGM,
    GM_ADDR workspaceGM, int64_t selfSendCnt, int64_t bs, int64_t h, int64_t topK,
    int64_t workspaceBytes, int64_t magic, int64_t commBytes, int64_t blockCount, int64_t routeStride,
    int64_t rxWindowBytes, int64_t rxWindowOffset0, int64_t rxWindowOffset1, int64_t roundDoneOffset0,
    int64_t roundDoneOffset1, int64_t rxLaneDoneOffset, int64_t senderDoneOffset,
    int64_t roundPublishOffset, int64_t roundCreditOffset, int64_t startGateOffset,
    int64_t runStartGate, int64_t errorStatusOffset,
    int64_t txReadyOffset, int64_t txDataOffset, GM_ADDR perfTrace, int64_t perfTraceBytes,
    GM_ADDR strictKernelCycles)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (commArgsGM == nullptr || topKWeightsGM == nullptr || yOutGM == nullptr || workspaceGM == nullptr ||
            (selfSendCnt > 0 && (expertOutGM == nullptr || assistInfoForCombineGM == nullptr)) ||
            selfSendCnt < 0 || bs <= 0 || h <= 0 || h > TileXREp::kEpUrmaCombineMaxHidden || topK <= 0 ||
            topK > TileXREp::kEpUrmaCombineMaxTopK || magic <= 0 || commBytes !=
                TileXREp::kEpUrmaCombineQuantHeaderBytes + AlignUpInt64(h, kCursorAlignment) || blockCount <= 0 ||
            blockCount != (commBytes + TileXREp::kEpUrmaCombinePayloadBytes - 1) /
                TileXREp::kEpUrmaCombinePayloadBytes ||
            blockCount > TileXREp::kEpUrmaCombineMaxBlocksPerRoute || routeStride <= 0 ||
            routeStride != blockCount * TileXREp::kEpUrmaCombineDataBlockBytes ||
            (runStartGate != 0 && runStartGate != 1) ||
#if TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
            roundCreditOffset <= 0 ||
            roundCreditOffset > workspaceBytes - TileXREp::kEpUrmaCombineCacheLineBytes ||
#endif
            rxWindowBytes <= 0 || txReadyOffset < 0 || txDataOffset < txReadyOffset ||
            txDataOffset > workspaceBytes || selfSendCnt >
                (txDataOffset - txReadyOffset) / TileXREp::kEpUrmaCombineCacheLineBytes ||
            selfSendCnt > (workspaceBytes - txDataOffset) / routeStride) {
            return;
        }
        auto args = reinterpret_cast<__gm__ TileXR::CommArgs *>(commArgsGM);
        if (args->rankSize <= 0 || args->rankSize > TileXR::TILEXR_MAX_RANK_SIZE || args->rank < 0 ||
            args->rank >= args->rankSize || (args->rankSize > 1 && !TileXR::UDMARegistryEnabled(args))) {
            return;
        }
        if (args->rankSize > 1) {
            __gm__ TileXR::UDMAInfo *udmaInfo = TileXR::GetUDMAInfo(args);
            if (udmaInfo == nullptr || udmaInfo->qpNum <
                static_cast<uint32_t>(TileXREp::kEpUrmaCombineRequiredQpCount)) {
                return;
            }
        }
#if TILEXR_EP_URMA_START_GATE
        if (startGateOffset < 0 || startGateOffset > workspaceBytes ||
            args->rankSize > (workspaceBytes - startGateOffset) /
                TileXREp::kEpUrmaCombineCacheLineBytes) {
            return;
        }
#else
        (void)startGateOffset;
#endif
#if !TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
        (void)roundCreditOffset;
#endif
        if (strictKernelCycles != nullptr && perfTrace != nullptr) {
            return;
        }
        if (!ProfileBufferValid(perfTrace, perfTraceBytes, args->rank, args->rankSize)) {
            perfTrace = nullptr;
        }
        GM_ADDR kernelPerfTrace = perfTrace;

        const int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < 0 || blockIdx >= TileXREp::kEpUrmaCombineAivCount) {
            return;
        }
        const uint32_t perfRank = static_cast<uint32_t>(args->rank);
        const uint32_t perfCore = static_cast<uint32_t>(blockIdx);
        __ubuf__ PerfStats *perfStats = reinterpret_cast<__ubuf__ PerfStats *>(
            TileXR::TILEXR_PERF_TRACE_LOCAL_STATS_UB_OFFSET);
        TileXR::TileXRPerfLocalStatsInit(kernelPerfTrace, perfStats, perfRank, perfCore,
            TileXREp::kEpUrmaCombinePerfStageCount);
        GM_ADDR finePerfTrace = nullptr;
        if (kernelPerfTrace != nullptr) {
            const uint32_t profileDetail =
                reinterpret_cast<__gm__ TileXR::TileXRPerfTraceHeader *>(kernelPerfTrace)->flags;
            if (profileDetail >= 2) {
                finePerfTrace = kernelPerfTrace;
            }
            if (profileDetail == 0) {
                perfTrace = nullptr;
            }
        }
#if TILEXR_EP_URMA_START_GATE
        if (runStartGate != 0) {
            if (!WaitForSynchronizedStart(args, workspaceGM, magic, rxLaneDoneOffset, senderDoneOffset,
                    roundPublishOffset, startGateOffset, blockIdx, perfTrace, perfStats)) {
                return;
            }
        }
#else
        (void)runStartGate;
#endif
        const uint64_t kernelStart = ProfileKernelTimingBegin(kernelPerfTrace);
        const uint64_t strictKernelStart = StrictKernelTimingBegin(strictKernelCycles);
        const int64_t rxBufferIndex = static_cast<uint32_t>(magic) & 1U;
        const int64_t rxWindowOffset = rxBufferIndex == 0 ? rxWindowOffset0 : rxWindowOffset1;
        const int64_t roundDoneOffset = rxBufferIndex == 0 ? roundDoneOffset0 : roundDoneOffset1;
        AscendC::TPipe pipe;
        if (blockIdx < TileXREp::kEpUrmaCombinePackLaneCount) {
            const uint64_t packStart = ProfileBegin(perfTrace);
            PackRoutes(expertOutGM, workspaceGM, selfSendCnt, h, magic, blockCount, routeStride,
                txReadyOffset, txDataOffset, blockIdx, pipe, finePerfTrace, perfStats);
            ProfileEnd(perfTrace, perfStats, PerfStage::PACK_TOTAL, packStart);
            pipe.Reset();
            const uint64_t receiveStart = ProfileBegin(perfTrace);
            ReceiveTokens(topKWeightsGM, yOutGM, workspaceGM, bs, h, topK, magic, rxWindowOffset,
                blockCount, routeStride, rxLaneDoneOffset, errorStatusOffset, blockIdx, pipe,
                finePerfTrace, perfStats);
            ProfileEnd(perfTrace, perfStats, PerfStage::RECEIVE_TOTAL, receiveStart);
            StrictKernelTimingFinish(strictKernelCycles, perfCore, strictKernelStart);
            ProfileFinish(kernelPerfTrace, perfRank, perfCore, perfStats, kernelStart);
            return;
        }

        const int64_t senderId = blockIdx - TileXREp::kEpUrmaCombinePackLaneCount;
        if (senderId >= TileXREp::kEpUrmaCombineSendLaneCount) {
            return;
        }
#if TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
        if (!WaitDeferredRoundCredit(args, workspaceGM, magic, roundDoneOffset,
                senderDoneOffset, roundCreditOffset, senderId,
                perfTrace, finePerfTrace, perfStats)) {
            return;
        }
#endif
        const uint64_t sendStart = ProfileBegin(perfTrace);
        if (!SendRoutes(args, assistInfoForCombineGM, workspaceGM, selfSendCnt, bs, topK, magic,
                rxWindowOffset, blockCount, routeStride, txReadyOffset, txDataOffset, senderDoneOffset,
                errorStatusOffset, senderId, pipe, finePerfTrace, perfStats)) {
            return;
        }
        ProfileEnd(perfTrace, perfStats, PerfStage::SEND_TOTAL, sendStart);
#if TILEXR_EP_URMA_PARALLEL_ROUND_PUBLISH
#if TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT
        const int64_t publisherCount = args->rankSize < TileXREp::kEpUrmaCombineSendLaneCount ?
            args->rankSize : TileXREp::kEpUrmaCombineSendLaneCount;
        uint64_t publishStart = 0;
        if (senderId < publisherCount) {
            // Only release publishers participate, reducing sender0's serial fan-in while
            // preserving complete coverage of every RX lane.
            const uint64_t receiveWaitStart = ProfileBegin(perfTrace);
            if (!WaitLocalLineShard(workspaceGM + rxLaneDoneOffset,
                    TileXREp::kEpUrmaCombinePackLaneCount, senderId, publisherCount,
                    EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxLaneDone),
                    finePerfTrace, perfStats)) {
                return;
            }
            StoreControlValue(workspaceGM + senderDoneOffset +
                senderId * TileXREp::kEpUrmaCombineCacheLineBytes,
                EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxReleaseShardDone),
                finePerfTrace, perfStats);
            if (senderId == 0) {
                if (!WaitLocalLines(workspaceGM + senderDoneOffset, publisherCount,
                        EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxReleaseShardDone),
                        finePerfTrace, perfStats)) {
                    return;
                }
                ProfileEnd(perfTrace, perfStats, PerfStage::LOCAL_RX_WAIT, receiveWaitStart);
                ProfileAux(perfTrace, perfStats, PerfStage::LOCAL_RX_WAIT, 0,
                    static_cast<uint64_t>(TileXREp::kEpUrmaCombinePackLaneCount));
                publishStart = StartParallelRoundPublish(args, workspaceGM, magic,
                    roundDoneOffset, roundPublishOffset, perfTrace, finePerfTrace, perfStats);
            } else {
                ProfileEnd(perfTrace, perfStats, PerfStage::LOCAL_RX_WAIT, receiveWaitStart);
            }
        }

        uint64_t publishCount = 0;
        if (senderId < publisherCount) {
            publishCount = PublishRoundShard(args, workspaceGM, magic,
                roundDoneOffset, roundPublishOffset, senderDoneOffset, senderId,
                perfTrace, finePerfTrace, perfStats);
            if (publishCount == ~0ULL) {
                return;
            }
        }
        if (senderId == 0 &&
            !FinishParallelRoundPublish(workspaceGM, magic, senderDoneOffset,
                publisherCount, publishStart, publishCount,
                perfTrace, finePerfTrace, perfStats)) {
            return;
        }
#else
        uint64_t publishStart = 0;
        if (senderId == 0) {
            const uint64_t senderWaitStart = ProfileBegin(perfTrace);
            if (!WaitLocalLines(workspaceGM + senderDoneOffset, TileXREp::kEpUrmaCombineSendLaneCount,
                    EncodeControlValue(magic, TileXREp::kEpUrmaCombineSenderDone), finePerfTrace, perfStats)) {
                return;
            }
            ProfileEnd(perfTrace, perfStats, PerfStage::LOCAL_SENDER_WAIT, senderWaitStart);
            ProfileAux(perfTrace, perfStats, PerfStage::LOCAL_SENDER_WAIT, 0,
                static_cast<uint64_t>(TileXREp::kEpUrmaCombineSendLaneCount));
            const uint64_t receiveWaitStart = ProfileBegin(perfTrace);
            if (!WaitLocalLines(workspaceGM + rxLaneDoneOffset, TileXREp::kEpUrmaCombinePackLaneCount,
                    EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxLaneDone), finePerfTrace, perfStats)) {
                return;
            }
            ProfileEnd(perfTrace, perfStats, PerfStage::LOCAL_RX_WAIT, receiveWaitStart);
            ProfileAux(perfTrace, perfStats, PerfStage::LOCAL_RX_WAIT, 0,
                static_cast<uint64_t>(TileXREp::kEpUrmaCombinePackLaneCount));
            publishStart = StartParallelRoundPublish(args, workspaceGM, magic,
                roundDoneOffset, roundPublishOffset, perfTrace, finePerfTrace, perfStats);
        }
        const uint64_t publishCount = PublishRoundShard(args, workspaceGM, magic,
            roundDoneOffset, roundPublishOffset, senderDoneOffset, senderId,
            perfTrace, finePerfTrace, perfStats);
        if (publishCount == ~0ULL) {
            return;
        }
        if (senderId == 0) {
            if (!FinishParallelRoundPublish(workspaceGM, magic, senderDoneOffset,
                    TileXREp::kEpUrmaCombineSendLaneCount, publishStart, publishCount,
                    perfTrace, finePerfTrace, perfStats)) {
                return;
            }
            if (!WaitGlobalRoundDone(workspaceGM, args->rankSize, magic, roundDoneOffset,
                    perfTrace, finePerfTrace, perfStats)) {
                return;
            }
        }
#endif
#else
        if (senderId == 0) {
            const uint64_t senderWaitStart = ProfileBegin(perfTrace);
            WaitLocalLines(workspaceGM + senderDoneOffset, TileXREp::kEpUrmaCombineSendLaneCount,
                EncodeControlValue(magic, TileXREp::kEpUrmaCombineSenderDone), finePerfTrace, perfStats);
            ProfileEnd(perfTrace, perfStats, PerfStage::LOCAL_SENDER_WAIT, senderWaitStart);
            ProfileAux(perfTrace, perfStats, PerfStage::LOCAL_SENDER_WAIT, 0,
                static_cast<uint64_t>(TileXREp::kEpUrmaCombineSendLaneCount));
            const uint64_t receiveWaitStart = ProfileBegin(perfTrace);
            WaitLocalLines(workspaceGM + rxLaneDoneOffset, TileXREp::kEpUrmaCombinePackLaneCount,
                EncodeControlValue(magic, TileXREp::kEpUrmaCombineRxLaneDone), finePerfTrace, perfStats);
            ProfileEnd(perfTrace, perfStats, PerfStage::LOCAL_RX_WAIT, receiveWaitStart);
            ProfileAux(perfTrace, perfStats, PerfStage::LOCAL_RX_WAIT, 0,
                static_cast<uint64_t>(TileXREp::kEpUrmaCombinePackLaneCount));
            PublishAndWaitRoundDone(args, workspaceGM, magic, roundDoneOffset, roundPublishOffset,
                perfTrace, finePerfTrace, perfStats);
        }
#endif
        StrictKernelTimingFinish(strictKernelCycles, perfCore, strictKernelStart);
        ProfileFinish(kernelPerfTrace, perfRank, perfCore, perfStats, kernelStart);
    }
}

#undef TILEXR_EP_LOCAL_FUNCTION

void launch_tilexr_ep_urma_combine_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs,
    GM_ADDR expertOut, GM_ADDR assistInfoForCombine, GM_ADDR topKWeights, GM_ADDR yOut, GM_ADDR workspace,
    int64_t selfSendCnt, int64_t bs, int64_t h, int64_t topK, int64_t workspaceBytes, int64_t magic,
    int64_t commBytes, int64_t blockCount, int64_t routeStride, int64_t rxWindowBytes, int64_t rxWindowOffset0,
    int64_t rxWindowOffset1, int64_t roundDoneOffset0, int64_t roundDoneOffset1, int64_t rxLaneDoneOffset,
    int64_t senderDoneOffset, int64_t roundPublishOffset, int64_t roundCreditOffset,
    int64_t startGateOffset, int64_t runStartGate,
    int64_t errorStatusOffset, int64_t txReadyOffset, int64_t txDataOffset, GM_ADDR perfTrace, int64_t perfTraceBytes,
    GM_ADDR strictKernelCycles)
{
    tilexr_ep_urma_combine_kernel<<<blockDim, nullptr, stream>>>(commArgs, expertOut, assistInfoForCombine,
        topKWeights, yOut, workspace, selfSendCnt, bs, h, topK, workspaceBytes, magic, commBytes,
        blockCount, routeStride, rxWindowBytes, rxWindowOffset0, rxWindowOffset1,
        roundDoneOffset0, roundDoneOffset1, rxLaneDoneOffset, senderDoneOffset, roundPublishOffset,
        roundCreditOffset, startGateOffset, runStartGate, errorStatusOffset, txReadyOffset, txDataOffset,
        perfTrace, perfTraceBytes,
        strictKernelCycles);
}

#undef TILEXR_EP_URMA_CACHELESS
