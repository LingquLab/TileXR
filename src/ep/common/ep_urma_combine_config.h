#ifndef TILEXR_EP_COMMON_EP_URMA_COMBINE_CONFIG_H
#define TILEXR_EP_COMMON_EP_URMA_COMBINE_CONFIG_H

#include <cstdint>

namespace TileXREp {

// Validated Ascend 950 production protocol. CMake may override only the Send AIV count.
#ifndef TILEXR_EP_URMA_COMBINE_SEND_CORE_COUNT
#define TILEXR_EP_URMA_COMBINE_SEND_CORE_COUNT 22
#endif

constexpr uint32_t kEpUrmaCombineProfileCoreCount = 64;
constexpr uint32_t kEpUrmaCombineProfileSendCoreCount = TILEXR_EP_URMA_COMBINE_SEND_CORE_COUNT;
constexpr uint32_t kEpUrmaCombineProfilePackReceiveCoreCount =
    kEpUrmaCombineProfileCoreCount - kEpUrmaCombineProfileSendCoreCount;

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

} // namespace TileXREp

#endif // TILEXR_EP_COMMON_EP_URMA_COMBINE_CONFIG_H
