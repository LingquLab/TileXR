#ifndef TILEXR_EP_COMMON_EP_URMA_COMBINE_CONFIG_H
#define TILEXR_EP_COMMON_EP_URMA_COMBINE_CONFIG_H

#include <cstdint>

namespace TileXREp {

// Fixed Ascend 950 S22 production configuration.
#define TILEXR_EP_URMA_COMBINE_SEND_CORE_COUNT 22
#define TILEXR_EP_URMA_DOORBELL_BATCH_SIZE 1
#define TILEXR_EP_URMA_TX_READY_BATCH_SIZE 1
#define TILEXR_EP_URMA_TX_READY_SHARED_FLAG 0
#define TILEXR_EP_URMA_TX_READY_IN_DATA 0
#define TILEXR_EP_URMA_RX_SCHEDULER 1
#define TILEXR_EP_URMA_PARALLEL_ROUND_PUBLISH 1
#define TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT 1
#define TILEXR_EP_URMA_START_GATE 1
#define TILEXR_EP_URMA_QDC_VERSION 3
#define TILEXR_EP_URMA_TX_META_PREFETCH_FULL 0
#define TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH 0
#define TILEXR_EP_URMA_RX_READY_STICKY_MASK 1
#define TILEXR_EP_URMA_RX_READY_BATCH_MTE2 0
#define TILEXR_EP_URMA_RX_READY_BATCH_VECTOR 0

constexpr uint32_t kEpUrmaCombineProfileCoreCount = 64;
constexpr uint32_t kEpUrmaCombineProfileSendCoreCount = TILEXR_EP_URMA_COMBINE_SEND_CORE_COUNT;
constexpr uint32_t kEpUrmaCombineProfilePackReceiveCoreCount =
    kEpUrmaCombineProfileCoreCount - kEpUrmaCombineProfileSendCoreCount;
constexpr uint32_t kEpUrmaCombineDoorbellBatchSize = TILEXR_EP_URMA_DOORBELL_BATCH_SIZE;
constexpr uint32_t kEpUrmaCombineTxReadyBatchSize = TILEXR_EP_URMA_TX_READY_BATCH_SIZE;
constexpr bool kEpUrmaCombineTxReadySharedFlag = TILEXR_EP_URMA_TX_READY_SHARED_FLAG != 0;
constexpr bool kEpUrmaCombineTxReadyInData = TILEXR_EP_URMA_TX_READY_IN_DATA != 0;
constexpr uint32_t kEpUrmaCombineRxScheduler = TILEXR_EP_URMA_RX_SCHEDULER;
constexpr bool kEpUrmaCombineRxRoundRobin = kEpUrmaCombineRxScheduler >= 1;
constexpr bool kEpUrmaCombineRxStickyReady =
    kEpUrmaCombineRxScheduler >= 2 || TILEXR_EP_URMA_RX_READY_STICKY_MASK != 0;
constexpr bool kEpUrmaCombineParallelRoundPublish = TILEXR_EP_URMA_PARALLEL_ROUND_PUBLISH != 0;
constexpr bool kEpUrmaCombineDeferredRoundCredit = TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT != 0;
constexpr bool kEpUrmaCombineStartGate = TILEXR_EP_URMA_START_GATE != 0;
constexpr uint32_t kEpUrmaCombineQdcVersion = TILEXR_EP_URMA_QDC_VERSION;
constexpr bool kEpUrmaCombineTxMetaPrefetchFull = TILEXR_EP_URMA_TX_META_PREFETCH_FULL != 0;
constexpr bool kEpUrmaCombineTxReadyEarlyPublish = TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH != 0;
constexpr bool kEpUrmaCombineRxReadyStickyMask = TILEXR_EP_URMA_RX_READY_STICKY_MASK != 0;
constexpr bool kEpUrmaCombineRxReadyBatchMte2 = TILEXR_EP_URMA_RX_READY_BATCH_MTE2 != 0;
constexpr bool kEpUrmaCombineRxReadyBatchVector = TILEXR_EP_URMA_RX_READY_BATCH_VECTOR != 0;
constexpr bool kEpUrmaCombineBalancedSendRoutes = true;

enum class EpUrmaCombinePerfStage : uint32_t {
    KERNEL_TOTAL = 0,
    PACK_TOTAL = 1,
    PACK_INPUT_WAIT = 2,
    PACK_QUANTIZE = 3,
    PACK_TX_PUBLISH = 4,
    RECEIVE_TOTAL = 5,
    RX_FLAG_POLL_WAIT = 6,
    RX_UNPACK_WAIT = 7,
    RX_UNPACK_DEQUANT_CLEAR = 8,
    RX_OUTPUT = 9,
    SEND_TOTAL = 10,
    TX_META_SCAN = 11,
    TX_READY_POLL = 12,
    SELF_COPY = 13,
    UDMA_POST = 14,
    UDMA_QUIET = 15,
    LOCAL_SENDER_WAIT = 16,
    LOCAL_RX_WAIT = 17,
    ROUND_PUBLISH = 18,
    GLOBAL_ROUND_WAIT = 19,
    DCCI_TOTAL = 20,
    START_GATE = 21,
    PACK_TX_DATA_SUBMIT = 22,
    PACK_FIRST_TX_READY = 23,
    PACK_MTE3_EXPOSED_WAIT = 24,
    RX_READY_MTE2_WAIT = 25,
    RX_READY_VECTOR = 26,
};

constexpr uint32_t kEpUrmaCombinePerfStageCount = 27;

#if !defined(__CCE__) || !defined(__CCE_IS_AICORE__)
constexpr const char *const kEpUrmaCombinePerfStageNames[kEpUrmaCombinePerfStageCount] = {
    "kernel_total",
    "pack_total",
    "pack_input_wait",
    "pack_quantize",
    "pack_tx_publish",
    "receive_total",
    "rx_flag_poll_wait",
    "rx_unpack_wait",
    "rx_unpack_dequant_clear",
    "rx_output",
    "send_total",
    "tx_meta_scan",
    "tx_ready_poll",
    "self_copy",
    "udma_post",
    "udma_quiet",
    "local_sender_wait",
    "local_rx_wait",
    "round_publish",
    "global_round_wait",
    "dcci_total",
    "start_gate",
    "pack_tx_data_submit",
    "pack_first_tx_ready",
    "pack_mte3_exposed_wait",
    "rx_ready_mte2_wait",
    "rx_ready_vector",
};
#endif

static_assert(kEpUrmaCombineProfilePackReceiveCoreCount + kEpUrmaCombineProfileSendCoreCount ==
    kEpUrmaCombineProfileCoreCount,
    "URMA combine profile core roles must cover all AIV cores");
static_assert(kEpUrmaCombineProfileSendCoreCount > 0 &&
    kEpUrmaCombineProfileSendCoreCount < kEpUrmaCombineProfileCoreCount,
    "URMA combine Send core count must leave at least one Pack/Receive core");
static_assert(kEpUrmaCombineDoorbellBatchSize == 1 || kEpUrmaCombineDoorbellBatchSize == 2 ||
    kEpUrmaCombineDoorbellBatchSize == 4 || kEpUrmaCombineDoorbellBatchSize == 8,
    "URMA combine doorbell batch size must be 1, 2, 4, or 8");
static_assert(kEpUrmaCombineTxReadyBatchSize == 1 || kEpUrmaCombineTxReadyBatchSize == 2 ||
    kEpUrmaCombineTxReadyBatchSize == 4,
    "URMA combine TX-ready batch size must be 1, 2, or 4");
static_assert(TILEXR_EP_URMA_TX_READY_SHARED_FLAG == 0 ||
    TILEXR_EP_URMA_TX_READY_SHARED_FLAG == 1,
    "URMA combine shared TX-ready flag must be disabled or enabled");
static_assert(TILEXR_EP_URMA_TX_READY_IN_DATA == 0 || TILEXR_EP_URMA_TX_READY_IN_DATA == 1,
    "URMA combine in-data TX-ready flag must be disabled or enabled");
static_assert(TILEXR_EP_URMA_TX_READY_IN_DATA == 0 ||
    (TILEXR_EP_URMA_TX_READY_BATCH_SIZE == 1 && TILEXR_EP_URMA_TX_READY_SHARED_FLAG == 0),
    "URMA combine in-data TX-ready requires TX batch 1 and shared-ready disabled");
static_assert(TILEXR_EP_URMA_RX_SCHEDULER >= 0 && TILEXR_EP_URMA_RX_SCHEDULER <= 2,
    "URMA combine Receive scheduler must be sequential, round-robin, or round-robin+sticky");
static_assert(TILEXR_EP_URMA_PARALLEL_ROUND_PUBLISH == 0 ||
    TILEXR_EP_URMA_PARALLEL_ROUND_PUBLISH == 1,
    "URMA combine parallel round publish must be disabled or enabled");
static_assert(TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT == 0 ||
    TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT == 1,
    "URMA combine deferred round credit must be disabled or enabled");
static_assert(TILEXR_EP_URMA_DEFERRED_ROUND_CREDIT == 0 ||
    TILEXR_EP_URMA_PARALLEL_ROUND_PUBLISH == 1,
    "URMA combine deferred round credit requires parallel round publish");
static_assert(TILEXR_EP_URMA_START_GATE == 0 || TILEXR_EP_URMA_START_GATE == 1,
    "URMA combine start gate must be disabled or enabled");
static_assert(TILEXR_EP_URMA_QDC_VERSION == 0 || TILEXR_EP_URMA_QDC_VERSION == 1 ||
    TILEXR_EP_URMA_QDC_VERSION == 2 || TILEXR_EP_URMA_QDC_VERSION == 3,
    "URMA combine Quant/Dequant implementation version must be in [0, 3]");
static_assert(TILEXR_EP_URMA_TX_META_PREFETCH_FULL == 0 ||
    TILEXR_EP_URMA_TX_META_PREFETCH_FULL == 1,
    "URMA combine full metadata prefetch must be disabled or enabled");
static_assert(TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH == 0 ||
    TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH == 1,
    "URMA combine early TX-ready publish must be disabled or enabled");
static_assert(TILEXR_EP_URMA_TX_READY_EARLY_PUBLISH == 0 ||
    (TILEXR_EP_URMA_TX_READY_IN_DATA == 0 && TILEXR_EP_URMA_TX_READY_BATCH_SIZE == 1 &&
     TILEXR_EP_URMA_TX_READY_SHARED_FLAG == 0),
    "URMA combine early TX-ready publish requires separate per-route ready lines");
static_assert(TILEXR_EP_URMA_RX_READY_STICKY_MASK == 0 ||
    TILEXR_EP_URMA_RX_READY_STICKY_MASK == 1,
    "URMA combine RX sticky ready mask must be disabled or enabled");
static_assert(TILEXR_EP_URMA_RX_READY_BATCH_MTE2 == 0 ||
    TILEXR_EP_URMA_RX_READY_BATCH_MTE2 == 1,
    "URMA combine RX ready MTE2 batching must be disabled or enabled");
static_assert(TILEXR_EP_URMA_RX_READY_BATCH_VECTOR == 0 ||
    TILEXR_EP_URMA_RX_READY_BATCH_VECTOR == 1,
    "URMA combine RX ready vector reduction must be disabled or enabled");
static_assert(TILEXR_EP_URMA_RX_READY_STICKY_MASK == 0 || TILEXR_EP_URMA_RX_SCHEDULER > 0,
    "URMA combine RX sticky ready mask requires a round-robin scheduler");
static_assert(TILEXR_EP_URMA_RX_READY_BATCH_MTE2 == 0 || TILEXR_EP_URMA_RX_SCHEDULER > 0,
    "URMA combine RX ready MTE2 batching requires a round-robin scheduler");
static_assert(TILEXR_EP_URMA_RX_READY_BATCH_VECTOR == 0 ||
    TILEXR_EP_URMA_RX_READY_BATCH_MTE2 == 1,
    "URMA combine RX ready vector reduction requires batched MTE2 reads");

} // namespace TileXREp

#endif // TILEXR_EP_COMMON_EP_URMA_COMBINE_CONFIG_H
