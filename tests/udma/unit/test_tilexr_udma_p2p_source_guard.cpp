#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string& path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string& path)
{
    const std::string fullPath = RepoPath(path);
    std::ifstream input(fullPath.c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << fullPath << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << path << " does not contain required text: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string& path, const std::string& text, const std::string& needle)
{
    if (text.find(needle) != std::string::npos) {
        std::cerr << path << " contains forbidden text: " << needle << std::endl;
        ++g_failures;
    }
}

void TestMemoryConsumeKernelUsesSyncCollectives()
{
    const std::string path = "tests/udma/demo/tilexr_udma_demo_kernel.cpp";
    const std::string text = ReadFile(path);
    CheckContains(path, text, "#include \"tilexr_sync.h\"");
    CheckContains(path, text, "tilexr_memory_consume_p2p_perf_kernel");
    CheckContains(path, text, "launch_tilexr_memory_consume_p2p_perf");
    CheckContains(path, text, "SyncCollectives sync");
    CheckContains(path, text, "int32_t magic, int32_t step");
    CheckContains(path, text, "sync.SetOuterFlag(magic, step);");
    CheckContains(path, text, "sync.WaitOuterFlag(magic, step, peer, blockIdx);");
}

void TestMemorySegmentedDiagnosticTransport()
{
    const std::string kernelPath = "tests/udma/demo/tilexr_udma_demo_kernel.cpp";
    const std::string kernelText = ReadFile(kernelPath);
    CheckContains(kernelPath, kernelText, "tilexr_memory_segmented_p2p_perf_kernel");
    CheckContains(kernelPath, kernelText, "launch_tilexr_memory_segmented_p2p_perf");
    CheckContains(kernelPath, kernelText, "uint32_t segmentBytes, int32_t rotateWindow");
    CheckContains(kernelPath, kernelText, "dstInWindow = srcOffset % segmentBytes");
    CheckContains(kernelPath, kernelText, "uint32_t traceSegments");
    CheckContains(kernelPath, kernelText, "AscendC::GetSystemCycle()");
    CheckContains(kernelPath, kernelText, "TileXRUdmaDemoAddCycleSum(debug, traceLowIdx");

    const std::string configPath = "tests/udma/demo/tilexr_udma_p2p_perf_config.h";
    const std::string configText = ReadFile(configPath);
    CheckContains(configPath, configText, "kP2PMemorySegmentBytes = 16ULL * 1024ULL * 1024ULL");
    CheckContains(configPath, configText, "kP2PMemoryTraceSegmentBytes = 8ULL * 1024ULL * 1024ULL");
    CheckContains(configPath, configText, "MemorySegmented");
    CheckContains(configPath, configText, "MemorySegmentedRotate");
    CheckContains(configPath, configText, "MemorySegmentedTrace");
    CheckContains(configPath, configText, "MemorySegmentedRotateTrace");
    CheckContains(configPath, configText, "memory_segmented_rotate");
    CheckContains(configPath, configText, "memory_segmented_rotate_trace");

    const std::string hostPath = "tests/udma/demo/tilexr_udma_demo.cpp";
    const std::string hostText = ReadFile(hostPath);
    CheckContains(hostPath, hostText, "launch_tilexr_memory_segmented_p2p_perf");
    CheckContains(hostPath, hostText, "P2PTransport::MemorySegmentedRotate");
    CheckContains(hostPath, hostText, "P2PTransport::MemorySegmentedRotateTrace");
    CheckContains(hostPath, hostText, "traceSegments ? 8U : 0U");
    CheckContains(hostPath, hostText, "skipPayloadCheck");
}

void TestMemoryConsumeHostWiring()
{
    const std::string path = "tests/udma/demo/tilexr_udma_demo.cpp";
    const std::string text = ReadFile(path);
    CheckContains(path, text, "launch_tilexr_memory_consume_p2p_perf");
    CheckContains(path, text, "P2PTransport::MemoryConsume");
    CheckContains(path, text, "P2PTransportUsesIpc");
    CheckContains(path, text, "P2PTransportBothRanksActive");
    CheckContains(path, text, "constexpr int32_t kP2PMagicBase = 0x5444554d");
    CheckContains(path, text, "int32_t launchMagic = kP2PMagicBase");
    CheckContains(path, text, "const int32_t warmupMagic = ++launchMagic");
    CheckContains(path, text, "const int32_t measuredMagic = ++launchMagic");
    CheckContains(path, text, "warmupMagic, i + 1");
    CheckContains(path, text, "measuredMagic, i + 1");
}

void TestMemoryConsumeSweepDefault()
{
    const std::string path = "tests/udma/demo/run_tilexr_udma_p2p_concurrency_sweep.sh";
    const std::string text = ReadFile(path);
    CheckContains(path, text, "direct_urma,memory,memory_consume,data_as_flag");
}

void TestAscend950ChipAliases()
{
    const std::string path = "src/comm/tilexr_internal.cpp";
    const std::string text = ReadFile(path);
    CheckContains(path, text, "\"Ascend950DT_9581\", ChipName::CHIP_950");
    CheckContains(path, text, "\"Ascend950DT_9582\", ChipName::CHIP_950");
    CheckContains(path, text, "\"Ascend950DT_9584\", ChipName::CHIP_950");
}

void TestAscend950IpcMemoryUsesP2PMalloc()
{
    const std::string path = "src/comm/tilexr_comm.cpp";
    const std::string text = ReadFile(path);
    CheckContains(path, text, "TileXRIpcMemUsesP2PMalloc(GetChipName())");
    CheckContains(path, text, "chipName == ChipName::CHIP_950");
    CheckContains(path, text, "ACL_MEM_MALLOC_HUGE_FIRST_P2P");
}

void TestDataAsFlagEpochOrderedSource()
{
    const std::string kernelPath = "tests/udma/demo/tilexr_udma_demo_kernel.cpp";
    const std::string kernelText = ReadFile(kernelPath);
    CheckContains(kernelPath, kernelText, "tilexr_data_as_flag_epoch_ordered_p2p_perf_kernel");
    CheckContains(kernelPath, kernelText, "launch_tilexr_data_as_flag_epoch_ordered_p2p_perf");
    CheckContains(kernelPath, kernelText, "DataAsFlagEpoch(magic, step)");
    CheckContains(kernelPath, kernelText, "DataAsFlagSendEpochOrdered");
    CheckContains(kernelPath, kernelText, "DataAsFlagCheckAndRecvEpochOrdered");
    CheckContains(kernelPath, kernelText, "int32_t magic, int32_t step");
    CheckContains(kernelPath, kernelText, "int32_t strict");
    CheckContains(kernelPath, kernelText, "strict != 0");

    const std::string headerPath = "src/include/tilexr_data_as_flag.h";
    const std::string headerText = ReadFile(headerPath);
    CheckContains(headerPath, headerText, "DataAsFlagEpochReady");
    CheckContains(headerPath, headerText, "DataAsFlagCommitEpoch");
    CheckContains(headerPath, headerText, "DataAsFlagWriteBatchCommitFlag");
    CheckContains(headerPath, headerText, "DATA_AS_FLAG_COMMIT_BIT");
    CheckContains(headerPath, headerText, "const uint64_t commitEpoch = DataAsFlagCommitEpoch(epoch)");
    CheckContains(headerPath, headerText, "DataAsFlagEpochReady(DataAsFlagLoadEpochFlag(dataAsFlagGM, lastBlock, recvScratch), commitEpoch)");
    CheckNotContains(headerPath, headerText, "DataAsFlagCopyEpochFlagsToGM");
    CheckNotContains(headerPath, headerText, "return DataAsFlagMaxRecvBlocks(scratchBytes);");

    const std::string hostPath = "tests/udma/demo/tilexr_udma_demo.cpp";
    const std::string hostText = ReadFile(hostPath);
    CheckContains(hostPath, hostText, "launch_tilexr_data_as_flag_epoch_ordered_p2p_perf");
    CheckContains(hostPath, hostText, "P2PTransport::DataAsFlagEpochOrdered");
    CheckContains(hostPath, hostText, "magic, step");
    CheckContains(hostPath, hostText, "useLegacyDataAsFlagTransport");
    CheckContains(hostPath, hostText, "TILEXR_DATA_AS_FLAG_STRICT");
}

} // namespace

int main()
{
    TestMemoryConsumeKernelUsesSyncCollectives();
    TestMemorySegmentedDiagnosticTransport();
    TestMemoryConsumeHostWiring();
    TestMemoryConsumeSweepDefault();
    TestAscend950ChipAliases();
    TestAscend950IpcMemoryUsesP2PMalloc();
    TestDataAsFlagEpochOrderedSource();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR UDMA P2P source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA P2P source guard checks passed" << std::endl;
    return 0;
}
