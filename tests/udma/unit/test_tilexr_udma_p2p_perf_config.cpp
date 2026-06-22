#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "demo/tilexr_udma_p2p_perf_config.h"

namespace {

void Require(bool condition, const char* message)
{
    if (!condition) {
        throw message;
    }
}

} // namespace

int main()
{
    TileXR::Demo::P2PPerfOptions options;
    options.srcRank = 1;
    options.dstRank = 0;
    options.minBytes = 4096;
    options.maxBytes = 16384;
    options.stepFactor = 2;
    options.iters = 20;
    options.warmupIters = 5;
    options.csvPath = "logs/p2p_perf.csv";

    std::string error;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error), "valid options rejected");
    Require(TileXR::Demo::DirectionName(0, 1) == "0to1", "direction name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DirectUrma) == "direct_urma",
        "direct transport name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::Memory) == "memory",
        "memory transport name mismatch");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma") == TileXR::Demo::P2PTransport::DirectUrma,
        "direct transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("memory") == TileXR::Demo::P2PTransport::Memory,
        "memory transport parse mismatch");
    options.transport = TileXR::Demo::P2PTransport::Memory;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error), "memory transport options rejected");
    options.maxBytes = TileXR::Demo::kP2PMemoryMaxBytes + 1;
    Require(!TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error), "oversized memory transport accepted");
    options.maxBytes = 16384;

    const std::vector<uint64_t> sizes = TileXR::Demo::BuildP2PPerfSizeSweep(options);
    Require(sizes.size() == 3, "size sweep count mismatch");
    Require(sizes[0] == 4096 && sizes[1] == 8192 && sizes[2] == 16384, "size sweep values mismatch");

    const uint32_t pattern = TileXR::Demo::P2PPattern(1, 0, 4096);
    std::vector<uint8_t> bytes(4096);
    TileXR::Demo::FillP2PPattern(bytes, pattern);
    Require(TileXR::Demo::CountP2PMismatches(bytes, pattern, 4096) == 0, "pattern validation failed");
    bytes[17] ^= 0xff;
    Require(TileXR::Demo::CountP2PMismatches(bytes, pattern, 4096) == 1, "mismatch count failed");

    TileXR::Demo::P2PPerfRow row;
    row.srcRank = 1;
    row.dstRank = 0;
    row.rankSize = 2;
    row.bytes = 4096;
    row.iters = 20;
    row.avgUs = 8.0;
    row.status = 0;
    row.errors = 0;
    row.logDir = "logs/run";
    const std::string csv = TileXR::Demo::FormatP2PPerfCsvRow(row);
    Require(csv == "1to0,1,0,2,4096,20,8.000,0.000,0.000,0.512,0,0,logs/run\n",
        "csv row mismatch");

    TileXR::Demo::P2PRankStatus srcSample;
    srcSample.status = 0;
    srcSample.errors = 0;
    srcSample.elapsedMs = 6.4f;
    TileXR::Demo::P2PRankStatus dstSample;
    dstSample.status = 0xffffffffu;
    dstSample.errors = 0;
    dstSample.elapsedMs = 0.1f;

    const TileXR::Demo::P2PPerfRow aggregated =
        TileXR::Demo::BuildP2PPerfRow(options, 2, 4096, srcSample, dstSample);
    Require(std::fabs(aggregated.avgUs - 320.0) < 0.001,
        "p2p row must use src rank elapsed time");
    Require(aggregated.status == 0, "p2p row must use src rank status");
    Require(aggregated.errors == 0, "p2p row must use dst rank errors");

    options.dstRank = 1;
    Require(!TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error), "same src/dst accepted");
    return 0;
}
