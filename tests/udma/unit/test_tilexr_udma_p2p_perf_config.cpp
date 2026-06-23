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
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DirectUrmaMultiWqe) ==
            "direct_urma_multi_wqe",
        "direct_urma_multi_wqe transport name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DirectUrmaMultiJetty) ==
            "direct_urma_multi_jetty",
        "direct_urma_multi_jetty transport name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallel) ==
            "direct_urma_multi_jetty_parallel",
        "direct_urma_multi_jetty_parallel transport name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe) ==
            "direct_urma_multi_jetty_parallel_fixed_wqe",
        "direct_urma_multi_jetty_parallel_fixed_wqe transport name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::Memory) == "memory",
        "memory transport name mismatch");
    Require(TileXR::Demo::P2PTransportName(TileXR::Demo::P2PTransport::DataAsFlag) == "data_as_flag",
        "data_as_flag transport name mismatch");
    Require(TileXR::Demo::P2PTrafficName(TileXR::Demo::P2PTraffic::UniDir) == "unidir",
        "unidir traffic name mismatch");
    Require(TileXR::Demo::P2PTrafficName(TileXR::Demo::P2PTraffic::BiDir) == "bidir",
        "bidir traffic name mismatch");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma") == TileXR::Demo::P2PTransport::DirectUrma,
        "direct transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma_multi_wqe") ==
            TileXR::Demo::P2PTransport::DirectUrmaMultiWqe,
        "direct_urma_multi_wqe transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma_multi_jetty") ==
            TileXR::Demo::P2PTransport::DirectUrmaMultiJetty,
        "direct_urma_multi_jetty transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma_multi_jetty_parallel") ==
            TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallel,
        "direct_urma_multi_jetty_parallel transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("direct_urma_multi_jetty_parallel_fixed_wqe") ==
            TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe,
        "direct_urma_multi_jetty_parallel_fixed_wqe transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("memory") == TileXR::Demo::P2PTransport::Memory,
        "memory transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTransport("data_as_flag") == TileXR::Demo::P2PTransport::DataAsFlag,
        "data_as_flag transport parse mismatch");
    Require(TileXR::Demo::ParseP2PTraffic("unidir") == TileXR::Demo::P2PTraffic::UniDir,
        "unidir traffic parse mismatch");
    Require(TileXR::Demo::ParseP2PTraffic("bidir") == TileXR::Demo::P2PTraffic::BiDir,
        "bidir traffic parse mismatch");
    Require(TileXR::Demo::TrafficDirectionName(0, 1, TileXR::Demo::P2PTraffic::BiDir) == "0to1+1to0",
        "bidir direction name mismatch");
    Require(TileXR::Demo::P2PTransportWindowBytes(TileXR::Demo::P2PTransport::DataAsFlag, 0) == 0,
        "data_as_flag zero layout mismatch");
    Require(TileXR::Demo::P2PTransportWindowBytes(TileXR::Demo::P2PTransport::DataAsFlag, 480) == 512,
        "data_as_flag 480B layout mismatch");
    Require(TileXR::Demo::P2PTransportWindowBytes(TileXR::Demo::P2PTransport::DataAsFlag, 481) == 1024,
        "data_as_flag 481B layout mismatch");
    Require(TileXR::Demo::P2PFixedWqeStrideBytes(8) == 64, "fixed-wqe small payload stride mismatch");
    Require(TileXR::Demo::P2PFixedWqeWindowBytes(8, 4) == 200,
        "fixed-wqe small payload window layout mismatch");
    Require(TileXR::Demo::P2PTransportWindowBytes(
                TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe, 4096, 4) == 16384,
        "fixed-wqe large payload window layout mismatch");
    Require(TileXR::Demo::ActiveP2PFlowCount(TileXR::Demo::P2PTraffic::UniDir) == 1,
        "unidir active flow count mismatch");
    Require(TileXR::Demo::ActiveP2PFlowCount(TileXR::Demo::P2PTraffic::BiDir) == 2,
        "bidir active flow count mismatch");
    options.transport = TileXR::Demo::P2PTransport::Memory;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error), "memory transport options rejected");
    options.transport = TileXR::Demo::P2PTransport::DirectUrmaMultiWqe;
    options.traffic = TileXR::Demo::P2PTraffic::BiDir;
    options.blockDim = 8;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error),
        "valid direct_urma_multi_wqe options rejected");
    options.transport = TileXR::Demo::P2PTransport::DirectUrmaMultiJetty;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error),
        "valid direct_urma_multi_jetty options rejected");
    options.transport = TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallel;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error),
        "valid direct_urma_multi_jetty_parallel options rejected");
    options.transport = TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error),
        "valid direct_urma_multi_jetty_parallel_fixed_wqe options rejected");
    options.transport = TileXR::Demo::P2PTransport::DataAsFlag;
    options.traffic = TileXR::Demo::P2PTraffic::BiDir;
    options.blockDim = 4;
    Require(TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error),
        "valid data_as_flag bidir options rejected");
    options.blockDim = 0;
    Require(!TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error), "block_dim=0 accepted");
    options.blockDim = 4;
    options.maxBytes = TileXR::Demo::kP2PMemoryMaxBytes + 1;
    Require(!TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error), "oversized ipc transport accepted");
    options.maxBytes = 16384;
    options.transport = TileXR::Demo::P2PTransport::Memory;
    options.traffic = TileXR::Demo::P2PTraffic::UniDir;
    options.blockDim = 4;

    const std::vector<uint64_t> sizes = TileXR::Demo::BuildP2PPerfSizeSweep(options);
    Require(sizes.size() == 3, "size sweep count mismatch");
    Require(sizes[0] == 4096 && sizes[1] == 8192 && sizes[2] == 16384, "size sweep values mismatch");

    const uint32_t pattern = TileXR::Demo::P2PPattern(1, 0, 4096);
    std::vector<uint8_t> bytes(4096);
    TileXR::Demo::FillP2PPattern(bytes, pattern);
    Require(TileXR::Demo::CountP2PMismatches(bytes, pattern, 4096) == 0, "pattern validation failed");
    bytes[17] ^= 0xff;
    Require(TileXR::Demo::CountP2PMismatches(bytes, pattern, 4096) == 1, "mismatch count failed");
    std::vector<uint8_t> fixedWqeBytes(TileXR::Demo::P2PFixedWqeWindowBytes(8, 4), 0);
    TileXR::Demo::FillP2PPattern(fixedWqeBytes, pattern);
    fixedWqeBytes[12] = 0;
    fixedWqeBytes[32] ^= 0xff;
    Require(TileXR::Demo::CountP2PTransportMismatches(fixedWqeBytes, pattern, 8,
                TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe, 4) == 0,
        "fixed-wqe mismatch checker must ignore padding");
    fixedWqeBytes[64] ^= 0xff;
    Require(TileXR::Demo::CountP2PTransportMismatches(fixedWqeBytes, pattern, 8,
                TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe, 4) == 1,
        "fixed-wqe mismatch checker must validate qp payload");

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
    Require(csv == "direct_urma,unidir,1,1to0,1,0,2,4096,20,8.000,0.000,0.000,0.512,0.512,0,0,logs/run\n",
        "csv row mismatch");
    row.traffic = TileXR::Demo::P2PTraffic::BiDir;
    row.blockDim = 4;
    row.transport = TileXR::Demo::P2PTransport::DataAsFlag;
    const std::string bidirCsv = TileXR::Demo::FormatP2PPerfCsvRow(row);
    Require(bidirCsv ==
            "data_as_flag,bidir,4,1to0+0to1,1,0,2,4096,20,8.000,0.000,0.000,1.024,0.512,0,0,logs/run\n",
        "bidir csv row mismatch");
    row.transport = TileXR::Demo::P2PTransport::DirectUrmaMultiWqe;
    row.traffic = TileXR::Demo::P2PTraffic::UniDir;
    row.blockDim = 8;
    const std::string multiWqeCsv = TileXR::Demo::FormatP2PPerfCsvRow(row);
    Require(multiWqeCsv ==
            "direct_urma_multi_wqe,unidir,8,1to0,1,0,2,4096,20,8.000,0.000,0.000,0.512,0.512,0,0,logs/run\n",
        "direct_urma_multi_wqe csv row mismatch");
    row.transport = TileXR::Demo::P2PTransport::DirectUrmaMultiJetty;
    const std::string multiJettyCsv = TileXR::Demo::FormatP2PPerfCsvRow(row);
    Require(multiJettyCsv ==
            "direct_urma_multi_jetty,unidir,8,1to0,1,0,2,4096,20,8.000,0.000,0.000,0.512,0.512,0,0,logs/run\n",
        "direct_urma_multi_jetty csv row mismatch");
    row.transport = TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallel;
    const std::string multiJettyParallelCsv = TileXR::Demo::FormatP2PPerfCsvRow(row);
    Require(multiJettyParallelCsv ==
            "direct_urma_multi_jetty_parallel,unidir,8,1to0,1,0,2,4096,20,8.000,0.000,0.000,0.512,0.512,0,0,logs/run\n",
        "direct_urma_multi_jetty_parallel csv row mismatch");
    row.transport = TileXR::Demo::P2PTransport::DirectUrmaMultiJettyParallelFixedWqe;
    const std::string multiJettyParallelFixedWqeCsv = TileXR::Demo::FormatP2PPerfCsvRow(row);
    Require(multiJettyParallelFixedWqeCsv ==
            "direct_urma_multi_jetty_parallel_fixed_wqe,unidir,8,1to0,1,0,2,4096,20,8.000,0.000,0.000,4.096,4.096,0,0,logs/run\n",
        "direct_urma_multi_jetty_parallel_fixed_wqe csv row mismatch");

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
    Require(aggregated.transport == TileXR::Demo::P2PTransport::Memory,
        "p2p row must preserve transport");
    Require(aggregated.traffic == TileXR::Demo::P2PTraffic::UniDir,
        "p2p row must preserve traffic");
    Require(aggregated.blockDim == 4, "p2p row must preserve block_dim");

    options.traffic = TileXR::Demo::P2PTraffic::BiDir;
    dstSample.status = 4;
    dstSample.errors = 7;
    dstSample.elapsedMs = 10.0f;
    const TileXR::Demo::P2PPerfRow bidirAggregated =
        TileXR::Demo::BuildP2PPerfRow(options, 2, 4096, srcSample, dstSample);
    Require(std::fabs(bidirAggregated.avgUs - 500.0) < 0.001,
        "bidir row must use max rank elapsed time");
    Require(bidirAggregated.status == 4, "bidir row must combine rank status");
    Require(bidirAggregated.errors == 7, "bidir row must sum rank errors");

    options.dstRank = 1;
    Require(!TileXR::Demo::ValidateP2PPerfOptions(options, 2, &error), "same src/dst accepted");
    return 0;
}
