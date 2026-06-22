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
    Memory,
    Invalid,
};

inline const char* P2PTransportName(P2PTransport transport)
{
    switch (transport) {
        case P2PTransport::DirectUrma:
            return "direct_urma";
        case P2PTransport::Memory:
            return "memory";
        default:
            return "invalid";
    }
}

inline P2PTransport ParseP2PTransport(const std::string& name)
{
    if (name == "direct_urma" || name == "udma") {
        return P2PTransport::DirectUrma;
    }
    if (name == "memory" || name == "ipc" || name == "datacopy") {
        return P2PTransport::Memory;
    }
    return P2PTransport::Invalid;
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
};

struct P2PPerfRow {
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
    if (options.transport == P2PTransport::Invalid) {
        return fail("transport must be direct_urma or memory");
    }
    if (options.transport == P2PTransport::Memory && options.maxBytes > kP2PMemoryMaxBytes) {
        return fail("memory transport max_bytes must fit in the TileXR IPC data window");
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

inline std::string P2PPerfCsvHeader()
{
    return "direction,src,dst,ranks,bytes,iters,avg_us,min_us,max_us,bw_GBps,status,errors,log_dir\n";
}

inline std::string FormatP2PPerfCsvRow(const P2PPerfRow& row)
{
    const double bwGBps = row.avgUs > 0.0 ? static_cast<double>(row.bytes) / row.avgUs / 1000.0 : 0.0;
    std::ostringstream out;
    out << DirectionName(row.srcRank, row.dstRank) << ','
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
    row.srcRank = options.srcRank;
    row.dstRank = options.dstRank;
    row.rankSize = rankSize;
    row.bytes = bytes;
    row.iters = options.iters;
    row.avgUs = options.iters > 0 ?
        static_cast<double>(srcStatus.elapsedMs) * 1000.0 / static_cast<double>(options.iters) : 0.0;
    row.status = srcStatus.status;
    row.errors = dstStatus.errors;
    row.logDir = options.logDir;
    return row;
}

} // namespace Demo
} // namespace TileXR

#endif // TILEXR_UDMA_P2P_PERF_CONFIG_H
