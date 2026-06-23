#ifndef TILEXR_UDMA_P2P_PERF_CONFIG_H
#define TILEXR_UDMA_P2P_PERF_CONFIG_H

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace TileXR {
namespace Demo {

constexpr uint64_t kP2PMemoryMaxBytes = 100ULL * 1024ULL * 1024ULL;

enum class P2PTransport {
    DirectUrma,
    DirectUrmaMultiWqe,
    DirectUrmaMultiJetty,
    DirectUrmaMultiJettyParallel,
    DirectUrmaMultiJettyParallelFixedWqe,
    Memory,
    DataAsFlag,
    Invalid,
};

enum class P2PTraffic {
    UniDir,
    BiDir,
    Invalid,
};

inline const char* P2PTransportName(P2PTransport transport)
{
    switch (transport) {
        case P2PTransport::DirectUrma:
            return "direct_urma";
        case P2PTransport::DirectUrmaMultiWqe:
            return "direct_urma_multi_wqe";
        case P2PTransport::DirectUrmaMultiJetty:
            return "direct_urma_multi_jetty";
        case P2PTransport::DirectUrmaMultiJettyParallel:
            return "direct_urma_multi_jetty_parallel";
        case P2PTransport::DirectUrmaMultiJettyParallelFixedWqe:
            return "direct_urma_multi_jetty_parallel_fixed_wqe";
        case P2PTransport::Memory:
            return "memory";
        case P2PTransport::DataAsFlag:
            return "data_as_flag";
        default:
            return "invalid";
    }
}

inline const char* P2PTrafficName(P2PTraffic traffic)
{
    switch (traffic) {
        case P2PTraffic::UniDir:
            return "unidir";
        case P2PTraffic::BiDir:
            return "bidir";
        default:
            return "invalid";
    }
}

inline P2PTransport ParseP2PTransport(const std::string& name)
{
    if (name == "direct_urma" || name == "udma") {
        return P2PTransport::DirectUrma;
    }
    if (name == "direct_urma_multi_wqe" || name == "direct-urma-multi-wqe" || name == "udma_multi_wqe") {
        return P2PTransport::DirectUrmaMultiWqe;
    }
    if (name == "direct_urma_multi_jetty" || name == "direct-urma-multi-jetty" || name == "udma_multi_jetty") {
        return P2PTransport::DirectUrmaMultiJetty;
    }
    if (name == "direct_urma_multi_jetty_parallel" || name == "direct-urma-multi-jetty-parallel" ||
        name == "udma_multi_jetty_parallel") {
        return P2PTransport::DirectUrmaMultiJettyParallel;
    }
    if (name == "direct_urma_multi_jetty_parallel_fixed_wqe" ||
        name == "direct-urma-multi-jetty-parallel-fixed-wqe" ||
        name == "udma_multi_jetty_parallel_fixed_wqe" || name == "fixed_wqe_multi_jetty_parallel") {
        return P2PTransport::DirectUrmaMultiJettyParallelFixedWqe;
    }
    if (name == "memory" || name == "ipc" || name == "datacopy") {
        return P2PTransport::Memory;
    }
    if (name == "data_as_flag" || name == "data-as-flag" || name == "daf") {
        return P2PTransport::DataAsFlag;
    }
    return P2PTransport::Invalid;
}

inline P2PTraffic ParseP2PTraffic(const std::string& name)
{
    if (name == "unidir" || name == "uni" || name == "single") {
        return P2PTraffic::UniDir;
    }
    if (name == "bidir" || name == "bi" || name == "duplex" || name == "full_duplex") {
        return P2PTraffic::BiDir;
    }
    return P2PTraffic::Invalid;
}

struct P2PPerfOptions {
    int srcRank = 0;
    int dstRank = 1;
    uint64_t minBytes = 4096;
    uint64_t maxBytes = 4096;
    uint64_t stepFactor = 2;
    int iters = 100;
    int warmupIters = 10;
    bool check = true;
    std::string csvPath;
    std::string logDir;
    P2PTransport transport = P2PTransport::DirectUrma;
    P2PTraffic traffic = P2PTraffic::UniDir;
    uint32_t blockDim = 1;
};

struct P2PPerfRow {
    P2PTransport transport = P2PTransport::DirectUrma;
    P2PTraffic traffic = P2PTraffic::UniDir;
    uint32_t blockDim = 1;
    int srcRank = 0;
    int dstRank = 1;
    int rankSize = 2;
    uint64_t bytes = 0;
    int iters = 0;
    double avgUs = 0.0;
    double minUs = 0.0;
    double maxUs = 0.0;
    uint32_t status = 0;
    uint64_t errors = 0;
    std::string logDir;
};

struct P2PRankStatus {
    uint32_t status = 0;
    uint64_t errors = 0;
    float elapsedMs = 0.0f;
};

inline std::string DirectionName(int srcRank, int dstRank)
{
    return std::to_string(srcRank) + "to" + std::to_string(dstRank);
}

inline std::string TrafficDirectionName(int srcRank, int dstRank, P2PTraffic traffic)
{
    if (traffic == P2PTraffic::BiDir) {
        return DirectionName(srcRank, dstRank) + "+" + DirectionName(dstRank, srcRank);
    }
    return DirectionName(srcRank, dstRank);
}

inline uint64_t DataAsFlagWindowBytes(uint64_t payloadBytes)
{
    return ((payloadBytes + 479ULL) / 480ULL) * 512ULL;
}

inline uint64_t P2PAlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0ULL) {
        return value;
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0ULL ? value : value + alignment - remainder;
}

inline uint64_t P2PFixedWqeStrideBytes(uint64_t payloadBytes)
{
    constexpr uint64_t kFixedWqeAddressAlignment = 64ULL;
    return P2PAlignUp(payloadBytes, kFixedWqeAddressAlignment);
}

inline uint64_t P2PFixedWqeWindowBytes(uint64_t payloadBytes, uint32_t blockDim)
{
    if (blockDim == 0U) {
        return 0;
    }
    return P2PFixedWqeStrideBytes(payloadBytes) * static_cast<uint64_t>(blockDim - 1U) + payloadBytes;
}

inline uint64_t P2PTransportWindowBytes(P2PTransport transport, uint64_t payloadBytes)
{
    return transport == P2PTransport::DataAsFlag ? DataAsFlagWindowBytes(payloadBytes) : payloadBytes;
}

inline uint64_t P2PTransportWindowBytes(P2PTransport transport, uint64_t payloadBytes, uint32_t blockDim)
{
    if (transport == P2PTransport::DirectUrmaMultiJettyParallelFixedWqe) {
        return P2PFixedWqeWindowBytes(payloadBytes, blockDim);
    }
    return P2PTransportWindowBytes(transport, payloadBytes);
}

inline int ActiveP2PFlowCount(P2PTraffic traffic)
{
    return traffic == P2PTraffic::BiDir ? 2 : 1;
}

inline uint64_t P2PEffectiveTransferBytes(P2PTransport transport, uint64_t payloadBytes, uint32_t blockDim)
{
    if (transport == P2PTransport::DirectUrmaMultiJettyParallelFixedWqe) {
        return payloadBytes * static_cast<uint64_t>(blockDim);
    }
    return payloadBytes;
}

inline bool ValidateP2PPerfOptions(const P2PPerfOptions& options, int rankSize, std::string* error)
{
    auto fail = [error](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (rankSize != 2) {
        return fail("test_type=4 requires rank_size=2");
    }
    if (options.srcRank < 0 || options.srcRank >= rankSize ||
        options.dstRank < 0 || options.dstRank >= rankSize) {
        return fail("src_rank and dst_rank must be in [0, rank_size)");
    }
    if (options.srcRank == options.dstRank) {
        return fail("src_rank and dst_rank must be different");
    }
    if (options.minBytes == 0 || options.maxBytes == 0 || options.minBytes > options.maxBytes) {
        return fail("byte range must be nonzero and min_bytes <= max_bytes");
    }
    if (options.stepFactor < 2) {
        return fail("step_factor must be at least 2");
    }
    if (options.iters <= 0 || options.warmupIters < 0) {
        return fail("iters must be positive and warmup_iters must be nonnegative");
    }
    if (options.blockDim == 0U || options.blockDim > 64U) {
        return fail("block_dim must be in [1, 64]");
    }
    if (options.transport == P2PTransport::Invalid) {
        return fail("transport must be direct_urma, direct_urma_multi_wqe, direct_urma_multi_jetty, "
                    "direct_urma_multi_jetty_parallel, direct_urma_multi_jetty_parallel_fixed_wqe, "
                    "memory, or data_as_flag");
    }
    if (options.traffic == P2PTraffic::Invalid) {
        return fail("traffic must be unidir or bidir");
    }
    if ((options.transport == P2PTransport::Memory || options.transport == P2PTransport::DataAsFlag) &&
        P2PTransportWindowBytes(options.transport, options.maxBytes, options.blockDim) > kP2PMemoryMaxBytes) {
        return fail("memory/data_as_flag transport max_bytes must fit in the TileXR IPC data window");
    }
    return true;
}

inline std::vector<uint64_t> BuildP2PPerfSizeSweep(const P2PPerfOptions& options)
{
    std::vector<uint64_t> sizes;
    for (uint64_t bytes = options.minBytes; bytes <= options.maxBytes;) {
        sizes.push_back(bytes);
        if (bytes > options.maxBytes / options.stepFactor) {
            break;
        }
        bytes *= options.stepFactor;
    }
    if (sizes.empty() || sizes.back() != options.maxBytes) {
        sizes.push_back(options.maxBytes);
    }
    return sizes;
}

inline uint32_t P2PPattern(int srcRank, int dstRank, uint64_t bytes)
{
    uint32_t pattern = 0x5a000000u;
    pattern ^= static_cast<uint32_t>((srcRank & 0xff) << 16);
    pattern ^= static_cast<uint32_t>((dstRank & 0xff) << 8);
    pattern ^= static_cast<uint32_t>(bytes & 0xffu);
    return pattern;
}

inline uint8_t P2PPatternByte(uint32_t pattern, uint64_t index)
{
    return static_cast<uint8_t>((pattern >> ((index & 3u) * 8u)) & 0xffu);
}

inline void FillP2PPattern(std::vector<uint8_t>& data, uint32_t pattern)
{
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[static_cast<size_t>(i)] = P2PPatternByte(pattern, i);
    }
}

inline uint64_t CountP2PMismatches(const std::vector<uint8_t>& data, uint32_t pattern, uint64_t bytes)
{
    uint64_t errors = 0;
    const uint64_t limit = bytes < data.size() ? bytes : data.size();
    for (uint64_t i = 0; i < limit; ++i) {
        if (data[static_cast<size_t>(i)] != P2PPatternByte(pattern, i)) {
            ++errors;
        }
    }
    return errors;
}

inline uint64_t CountP2PTransportMismatches(
    const std::vector<uint8_t>& data, uint32_t pattern, uint64_t payloadBytes,
    P2PTransport transport, uint32_t blockDim)
{
    if (transport != P2PTransport::DirectUrmaMultiJettyParallelFixedWqe) {
        return CountP2PMismatches(data, pattern, payloadBytes);
    }

    uint64_t errors = 0;
    const uint64_t strideBytes = P2PFixedWqeStrideBytes(payloadBytes);
    for (uint32_t qpIdx = 0; qpIdx < blockDim; ++qpIdx) {
        const uint64_t offset = strideBytes * static_cast<uint64_t>(qpIdx);
        for (uint64_t i = 0; i < payloadBytes && offset + i < data.size(); ++i) {
            if (data[static_cast<size_t>(offset + i)] != P2PPatternByte(pattern, offset + i)) {
                ++errors;
            }
        }
    }
    return errors;
}

inline std::string P2PPerfCsvHeader()
{
    return "transport,traffic,block_dim,direction,src,dst,ranks,bytes,iters,avg_us,min_us,max_us,"
        "bw_GBps,per_flow_bw_GBps,status,errors,log_dir\n";
}

inline std::string FormatP2PPerfCsvRow(const P2PPerfRow& row)
{
    const uint64_t effectiveBytes = P2PEffectiveTransferBytes(row.transport, row.bytes, row.blockDim);
    const double perFlowBwGBps = row.avgUs > 0.0 ? static_cast<double>(effectiveBytes) / row.avgUs / 1000.0 : 0.0;
    const double bwGBps = perFlowBwGBps * static_cast<double>(ActiveP2PFlowCount(row.traffic));
    std::ostringstream out;
    out << P2PTransportName(row.transport) << ','
        << P2PTrafficName(row.traffic) << ','
        << row.blockDim << ','
        << TrafficDirectionName(row.srcRank, row.dstRank, row.traffic) << ','
        << row.srcRank << ','
        << row.dstRank << ','
        << row.rankSize << ','
        << row.bytes << ','
        << row.iters << ','
        << std::fixed << std::setprecision(3)
        << row.avgUs << ','
        << row.minUs << ','
        << row.maxUs << ','
        << bwGBps << ','
        << perFlowBwGBps << ','
        << row.status << ','
        << row.errors << ','
        << row.logDir << '\n';
    return out.str();
}

inline P2PPerfRow BuildP2PPerfRow(
    const P2PPerfOptions& options,
    int rankSize,
    uint64_t bytes,
    const P2PRankStatus& srcStatus,
    const P2PRankStatus& dstStatus)
{
    P2PPerfRow row;
    row.transport = options.transport;
    row.traffic = options.traffic;
    row.blockDim = options.blockDim;
    row.srcRank = options.srcRank;
    row.dstRank = options.dstRank;
    row.rankSize = rankSize;
    row.bytes = bytes;
    row.iters = options.iters;
    const bool bothRanksActive =
        options.traffic == P2PTraffic::BiDir || options.transport == P2PTransport::DataAsFlag;
    const float elapsedMs = bothRanksActive && dstStatus.elapsedMs > srcStatus.elapsedMs ?
        dstStatus.elapsedMs : srcStatus.elapsedMs;
    row.avgUs = options.iters > 0 ? static_cast<double>(elapsedMs) * 1000.0 / static_cast<double>(options.iters) : 0.0;
    row.status = bothRanksActive ? (srcStatus.status | dstStatus.status) : srcStatus.status;
    row.errors = bothRanksActive ? (srcStatus.errors + dstStatus.errors) : dstStatus.errors;
    row.logDir = options.logDir;
    return row;
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_P2P_PERF_CONFIG_H
