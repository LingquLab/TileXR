#include <dirent.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string &path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string &path)
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

std::string ExtractInitializer(const std::string &path, const std::string &text, const std::string &symbol)
{
    const auto symbolPos = text.find(symbol);
    if (symbolPos == std::string::npos) {
        std::cerr << "expected " << path << " to contain initializer symbol: " << symbol << std::endl;
        ++g_failures;
        return {};
    }

    const auto openBrace = text.find('{', symbolPos);
    if (openBrace == std::string::npos) {
        std::cerr << "expected " << path << " to contain initializer body for: " << symbol << std::endl;
        ++g_failures;
        return {};
    }

    int depth = 0;
    for (std::size_t pos = openBrace; pos < text.size(); ++pos) {
        if (text[pos] == '{') {
            ++depth;
            continue;
        }
        if (text[pos] == '}') {
            --depth;
            if (depth == 0) {
                return text.substr(openBrace + 1, pos - openBrace - 1);
            }
        }
    }

    std::cerr << "expected " << path << " to contain matching '}' for initializer: " << symbol << std::endl;
    ++g_failures;
    return {};
}

bool DirectoryExists(const std::string &path)
{
    DIR *dir = opendir(RepoPath(path).c_str());
    if (dir == nullptr) {
        return false;
    }
    closedir(dir);
    return true;
}

std::vector<std::string> CollectFiles(const std::string &path)
{
    std::vector<std::string> files;
    DIR *dir = opendir(RepoPath(path).c_str());
    if (dir == nullptr) {
        std::cerr << "failed to open directory " << RepoPath(path) << std::endl;
        ++g_failures;
        return files;
    }

    while (dirent *entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        const std::string child = path + "/" + name;
        if (entry->d_type == DT_DIR) {
            const auto childFiles = CollectFiles(child);
            files.insert(files.end(), childFiles.begin(), childFiles.end());
        } else if (entry->d_type == DT_REG) {
            files.push_back(child);
        }
    }
    closedir(dir);
    return files;
}

void CheckTrue(bool condition, const std::string &message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

void CheckContains(const std::string &path, const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "expected " << path << " to contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckDoesNotContain(const std::string &path, const std::string &text, const std::string &needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "expected " << path << " not to contain: " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

void TestCollectivesOwnsCceBuild()
{
    CheckTrue(DirectoryExists("src/collectives/kernels"),
              "expected src/collectives/kernels to exist");

    const std::string collectivesCmakePath = "src/collectives/CMakeLists.txt";
    const auto collectivesCmake = ReadFile(collectivesCmakePath);
    CheckContains(collectivesCmakePath, collectivesCmake, "add_subdirectory(kernels)");
    CheckContains(collectivesCmakePath, collectivesCmake, "tilexr_collectives_kernel_embed");
    CheckContains(collectivesCmakePath, collectivesCmake, "tilexr_collectives_op");
    CheckContains(collectivesCmakePath, collectivesCmake, "OBJECT_DEPENDS");
    CheckContains(collectivesCmakePath, collectivesCmake, "${TILEXR_COLLECTIVES_OP}");

    const std::string kernelsCmakePath = "src/collectives/kernels/CMakeLists.txt";
    const auto kernelsCmake = ReadFile(kernelsCmakePath);
    CheckContains(kernelsCmakePath, kernelsCmake, "enable_language(CCE)");
    CheckContains(kernelsCmakePath, kernelsCmake, "tilexr_lccl_op.cpp");
    CheckContains(kernelsCmakePath, kernelsCmake, "TILEXR_COLLECTIVES_KERNEL_DEPENDS");
    CheckContains(kernelsCmakePath, kernelsCmake, "CONFIGURE_DEPENDS");
    CheckContains(kernelsCmakePath, kernelsCmake, "OBJECT_DEPENDS");
    CheckContains(kernelsCmakePath, kernelsCmake, "tilexr_collectives_op.o");
    CheckContains(kernelsCmakePath, kernelsCmake, "tilexr_collectives_op");
    CheckContains(kernelsCmakePath, kernelsCmake, "TILEXR_COLLECTIVES_ENABLE_PROFILING");
    CheckContains(kernelsCmakePath, kernelsCmake, "TILEXR_COLLECTIVES_1OP_BIN_SIZE 10485760");
    CheckDoesNotContain(kernelsCmakePath, kernelsCmake, "src/comm");
}

void TestCollectivesKernelSourcesAreScoped()
{
    const std::string kernelTuPath = "src/collectives/kernels/tilexr_lccl_op.cpp";
    const auto kernelTu = ReadFile(kernelTuPath);
    CheckContains(kernelTuPath, kernelTu, "LCCL_TYPE_AIV_FUNC(LCCL_ALLGATHER_FUNC_AUTO_DEF)");
    CheckContains(kernelTuPath, kernelTu, "LCCL_TYPE_AIV_FUNC(LCCL_ALL2ALL_FUNC_AUTO_DEF)");
    CheckContains(kernelTuPath, kernelTu, "LCCL_TYPE_AIV_FUNC(LCCL_ALL_REDUCE_FUNC_AUTO_DEF)");
    CheckContains(kernelTuPath, kernelTu, "LCCL_TYPE_AIV_FUNC(LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF)");
    CheckContains(kernelTuPath, kernelTu, "LCCL_BROADCAST_FUNC_AUTO_DEF()");
    CheckContains(kernelTuPath, kernelTu, "LCCL_PROFILE_PROBE_FUNC_AUTO_DEF()");

    const std::string perfTraceKernelPath = "src/collectives/kernels/perf_trace_kernel.h";
    const auto perfTraceKernel = ReadFile(perfTraceKernelPath);
    CheckContains(perfTraceKernelPath, perfTraceKernel, "TILEXR_COLLECTIVES_ENABLE_PROFILING");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "TileXRPerfStageBegin");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "TileXRPerfStageEnd");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "TileXRPerfAccumulateDuration");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "TileXRPerfTraceEnabled");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "#include \"comm_args.h\"");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "#include \"datacopy_gm2gm.h\"");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "TILEXR_PERF_TRACE_STATS_UB_OFFSET");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "TILEXR_PERF_TRACE_STATS_OFFSET");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "CpGM2UB");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "CpUB2GM");
    CheckContains(perfTraceKernelPath, perfTraceKernel, "GetBlockNum()");
    CheckDoesNotContain(perfTraceKernelPath, perfTraceKernel, "header->statsOffset");
    CheckDoesNotContain(perfTraceKernelPath, perfTraceKernel, "header->maxCoreCount");
    CheckDoesNotContain(perfTraceKernelPath, perfTraceKernel, "header->stageCount");
    CheckDoesNotContain(perfTraceKernelPath, perfTraceKernel, "slot->count += 1");

    const std::string profileProbePath = "src/collectives/kernels/kernels/lcal_profile_probe.cce";
    const auto profileProbe = ReadFile(profileProbePath);
    CheckContains(profileProbePath, profileProbe, "TileXRProfileProbeKernel");
    CheckContains(profileProbePath, profileProbe, "PerfStageId::KERNEL_TOTAL");
    CheckContains(profileProbePath, profileProbe, "PerfStageId::CHUNK_TOTAL");
    CheckContains(profileProbePath, profileProbe, "PerfStageId::POST_SYNC");
    CheckContains(profileProbePath, profileProbe, "PerfStageId::LOCAL_INPUT_TO_IPC");
    CheckContains(profileProbePath, profileProbe, "PerfStageId::FLAG_POLL_WAIT");
    CheckContains(profileProbePath, profileProbe, "PerfStageId::PEER_IPC_TO_OUTPUT");
    CheckContains(profileProbePath, profileProbe, "PerfStageId::CHUNK_BARRIER");
    CheckContains(profileProbePath, profileProbe, "TileXRPerfStageBegin");
    CheckContains(profileProbePath, profileProbe, "TileXRPerfStageEnd");
    CheckContains(profileProbePath, profileProbe, "TileXRPerfAccumulateDuration");
    CheckContains(profileProbePath, profileProbe, "TILEXR_PROFILE_PROBE_COPY_UB_BYTES");
    CheckContains(profileProbePath, profileProbe, "while (copied < remain)");
    CheckContains(profileProbePath, profileProbe, "TileXR::TILEXR_PERF_TRACE_STATS_UB_OFFSET");
    CheckDoesNotContain(profileProbePath, profileProbe, "peerMems");
    CheckDoesNotContain(profileProbePath, profileProbe, "GET_IPC_MEM_ARGS");

    const std::string perfTraceLayoutPath = "src/include/tilexr_perf_trace.h";
    const auto perfTraceLayout = ReadFile(perfTraceLayoutPath);
    CheckContains(perfTraceLayoutPath, perfTraceLayout,
                  "Shared standalone-collectives profiling schema");
    CheckContains(perfTraceLayoutPath, perfTraceLayout, "PerfTraceCyclesToUs");
    CheckContains(perfTraceLayoutPath, perfTraceLayout, "__CCE_IS_AICORE__");
    CheckContains(perfTraceLayoutPath, perfTraceLayout, "!defined(__CCE__) || !defined(__CCE_IS_AICORE__)");

    const std::vector<std::string> kernelFiles = CollectFiles("src/collectives/kernels");
    CheckTrue(!kernelFiles.empty(), "expected collectives kernel files to be present");
    bool sawAllGatherCce = false;
    bool sawAllToAllCce = false;
    bool sawAllReduceCce = false;
    bool sawReduceScatterCce = false;
    bool sawBroadcastCce = false;
    bool sawProfileProbeCce = false;
    for (const auto &path : kernelFiles) {
        const auto text = ReadFile(path);
        CheckDoesNotContain(path, text, "tilexr_comm.h");
        CheckDoesNotContain(path, text, "reference/ascend-transformer-boost");
        CheckDoesNotContain(path, text, "LCAL_MAX_RANK_SIZE");
        if (path.find("lcal_allgather") != std::string::npos && path.find(".cce") != std::string::npos) {
            sawAllGatherCce = true;
        }
        if (path.find("lcal_all2all_transpose.cce") != std::string::npos) {
            sawAllToAllCce = true;
        }
        if (path.find("lcal_allreduce") != std::string::npos && path.find(".cce") != std::string::npos) {
            sawAllReduceCce = true;
        }
        if (path.find("lcal_reduce_scatter") != std::string::npos && path.find(".cce") != std::string::npos) {
            sawReduceScatterCce = true;
        }
        if (path.find("lcal_broadcast") != std::string::npos && path.find(".cce") != std::string::npos) {
            sawBroadcastCce = true;
        }
        if (path.find("lcal_profile_probe.cce") != std::string::npos) {
            sawProfileProbeCce = true;
        }
    }
    CheckTrue(sawAllGatherCce, "expected copied allgather .cce sources under src/collectives/kernels");
    CheckTrue(sawAllToAllCce, "expected copied all2all .cce source under src/collectives/kernels");
    CheckTrue(sawAllReduceCce, "expected copied allreduce .cce sources under src/collectives/kernels");
    CheckTrue(sawReduceScatterCce, "expected copied reduce_scatter .cce sources under src/collectives/kernels");
    CheckTrue(sawBroadcastCce, "expected copied broadcast .cce sources under src/collectives/kernels");
    CheckTrue(sawProfileProbeCce, "expected profile probe .cce source under src/collectives/kernels");
}

void TestHostRegistrationLivesInCollectives()
{
    const std::string kernelPath = "src/collectives/host/collective_kernel.cpp";
    const auto kernel = ReadFile(kernelPath);
    CheckContains(kernelPath, kernel, "rtDevBinaryRegister");
    CheckContains(kernelPath, kernel, "rtFunctionRegister");
    CheckContains(kernelPath, kernel, "rtKernelLaunchWithFlagV2");
    CheckContains(kernelPath, kernel, "TILEXR_TYPE2NAME");
    CheckContains(kernelPath, kernel, "TileXRCollectivesKernelBinaryData");
    CheckContains(kernelPath, kernel, "TileXRCollectivesKernelBinarySize");
    CheckContains(kernelPath, kernel, "std::mutex");
    CheckContains(kernelPath, kernel, "TileXR::TILEXR_DATA_TYPE_INT8");
    CheckContains(kernelPath, kernel, "TileXR::TILEXR_DATA_TYPE_INT16");
    CheckContains(kernelPath, kernel, "TileXR::TILEXR_DATA_TYPE_INT32");
    CheckContains(kernelPath, kernel, "TileXR::TILEXR_DATA_TYPE_INT64");
    CheckContains(kernelPath, kernel, "TileXR::TILEXR_DATA_TYPE_FP16");
    CheckContains(kernelPath, kernel, "TileXR::TILEXR_DATA_TYPE_FP32");
    CheckContains(kernelPath, kernel, "TileXR::TILEXR_DATA_TYPE_BFP16");
    CheckContains(kernelPath, kernel, "perfTrace");
    CheckContains(kernelPath, kernel, "PreparePerfTraceLaunch");
    CheckContains(kernelPath, kernel, "GetActivePerfTraceSession");
    CheckDoesNotContain(kernelPath, kernel, "g_collectiveKernelStub");

    const auto registeredTypes = ExtractInitializer(kernelPath, kernel, "kRegisteredCollectiveTypes");
    const std::string registeredTypesPath = kernelPath + " kRegisteredCollectiveTypes[]";
    CheckContains(registeredTypesPath, registeredTypes, "TileXR::TileXRType::ALL_GATHER");
    CheckContains(registeredTypesPath, registeredTypes, "TileXR::TileXRType::ALL2ALL");
    CheckContains(registeredTypesPath, registeredTypes, "TileXR::TileXRType::ALL_REDUCE");
    CheckContains(registeredTypesPath, registeredTypes, "TileXR::TileXRType::REDUCE_SCATTER");
    CheckContains(registeredTypesPath, registeredTypes, "TileXR::TileXRType::BROADCAST");
    CheckContains(registeredTypesPath, registeredTypes, "TileXR::TileXRType::PROFILE_PROBE");
}

void TestPerfTraceCycleDivisorIsA5Specific()
{
    const std::string commArgsPath = "src/include/comm_args.h";
    const auto commArgs = ReadFile(commArgsPath);
    CheckContains(commArgsPath, commArgs, "PERF_CYCLE_A5");

    const std::string commPath = "src/comm/tilexr_comm.cpp";
    const auto comm = ReadFile(commPath);
    CheckContains(commPath, comm, "GetChipName() == ChipName::CHIP_910A5");
    CheckContains(commPath, comm, "ExtraFlag::PERF_CYCLE_A5");

    const std::string sessionPath = "src/collectives/host/perf_trace_session.cpp";
    const auto session = ReadFile(sessionPath);
    CheckContains(sessionPath, session, "ExtraFlag::PERF_CYCLE_A5");
    CheckDoesNotContain(sessionPath, session, "TOPO_910A5) != 0 ? 1000u : 50u");
}

void TestDeviceKernelArgsMatchHostLaunchAbi()
{
    const std::string collectivesPath = "src/collectives/kernels/collectives.h";
    const auto collectives = ReadFile(collectivesPath);
    CheckContains(collectivesPath, collectives, "GM_ADDR perfTrace");
    CheckContains(collectivesPath, collectives, "KERNELS_ARGS_CALL()");
    CheckContains(collectivesPath, collectives,
                  "input, output, commArgs, len, magic, op, root, cycleCount, scale, scaleCount, offset, perfTrace");
}

void TestBigDataAllGatherPerfStages()
{
    const std::string path = "src/collectives/kernels/kernels/lcal_allgather_big_data.cce";
    const auto text = ReadFile(path);
    CheckDoesNotContain(path, text, "PerfStageId::KERNEL_TOTAL");
    CheckContains(path, text, "PerfStageId::CHUNK_TOTAL");
    CheckContains(path, text, "PerfStageId::POST_SYNC");
    CheckContains(path, text, "PerfStageId::LOCAL_INPUT_TO_IPC");
    CheckContains(path, text, "PerfStageId::FLAG_POLL_WAIT");
    CheckContains(path, text, "PerfStageId::PEER_IPC_TO_OUTPUT");
    CheckContains(path, text, "PerfStageId::CHUNK_BARRIER");
}

void TestTwoNpuBigDataAllGatherPerfStages()
{
    const std::string path = "src/collectives/kernels/kernels/lcal_allgather_2npu_big_data_write.cce";
    const auto text = ReadFile(path);
    CheckContains(path, text, "GM_ADDR perfTrace");
    CheckDoesNotContain(path, text, "PerfStageId::KERNEL_TOTAL");
    CheckContains(path, text, "PerfStageId::CHUNK_TOTAL");
    CheckContains(path, text, "PerfStageId::POST_SYNC");
    CheckContains(path, text, "PerfStageId::LOCAL_INPUT_TO_IPC");
    CheckContains(path, text, "PerfStageId::FLAG_POLL_WAIT");
    CheckContains(path, text, "PerfStageId::PEER_IPC_TO_OUTPUT");
    CheckContains(path, text, "PerfStageId::CHUNK_BARRIER");
}

void TestOtherStandaloneCollectivePerfStages()
{
    const char *stageNeedles[] = {
        "PerfStageId::CHUNK_TOTAL",
        "PerfStageId::POST_SYNC",
        "PerfStageId::LOCAL_INPUT_TO_IPC",
        "PerfStageId::FLAG_POLL_WAIT",
        "PerfStageId::PEER_IPC_TO_OUTPUT",
        "PerfStageId::CHUNK_BARRIER",
    };

    const char *paths[] = {
        "src/collectives/kernels/kernels/lcal_allreduce_big_data.cce",
        "src/collectives/kernels/kernels/lcal_allreduce_2npu_big_write.cce",
        "src/collectives/kernels/kernels/lcal_reduce_scatter_big_data.cce",
        "src/collectives/kernels/kernels/lcal_reduce_scatter_big_data_write.cce",
        "src/collectives/kernels/kernels/lcal_broadcast_big_data.cce",
        "src/collectives/kernels/kernels/lcal_broadcast_write.cce",
        "src/collectives/kernels/kernels/lcal_all2all_transpose.cce",
        "src/collectives/kernels/91093/all2all_hierarchy.h",
        "src/collectives/kernels/91093/all2all_hierarchy_small.h",
    };
    for (const char *path : paths) {
        const auto text = ReadFile(path);
        for (const char *stageNeedle : stageNeedles) {
            CheckContains(path, text, stageNeedle);
        }
    }
}

void TestStandaloneCollectiveWrappersHaveCoarsePerfTrace()
{
    const std::string path = "src/collectives/kernels/lccl_op.h";
    const auto text = ReadFile(path);
    CheckContains(path, text, "TileXRCoarsePerfToken");
    CheckContains(path, text, "TILEXR_COARSE_PERF_BEGIN");
    CheckContains(path, text, "TILEXR_COARSE_PERF_END");

    const char *wrappers[] = {
        "LCCL_ALLGATHER_FUNC_AUTO_DEF",
        "LCCL_BROADCAST_FUNC_AUTO_DEF",
        "LCCL_ALL_REDUCE_FUNC_AUTO_DEF",
        "LCCL_ALL2ALL_FUNC_AUTO_DEF",
        "LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF",
    };
    for (const char *wrapper : wrappers) {
        const auto body = ExtractInitializer(path, text, wrapper);
        CheckContains(std::string(path) + " " + wrapper, body, "TILEXR_COARSE_PERF_BEGIN");
        CheckContains(std::string(path) + " " + wrapper, body, "TILEXR_COARSE_PERF_END");
    }
}

void TestCommDoesNotOwnCollectiveRuntime()
{
    const auto commFiles = CollectFiles("src/comm");
    const std::string forbidden[] = {
        "src/collectives",
        "TILEXR_CCE_BIN_STR",
        "RegistKernel",
        "LoadMTE",
        "rtFunctionRegister",
        "rtDevBinaryRegister",
        "rtKernelLaunchWithFlagV2",
        "AscendCCLKernelArgs",
        "TileXRAllGather",
        "TileXRAllToAll",
        "TileXRAllReduce",
        "TileXRReduceScatter",
        "TileXRBroadcast",
    };

    for (const auto &path : commFiles) {
        const auto text = ReadFile(path);
        for (const auto &needle : forbidden) {
            CheckDoesNotContain(path, text, needle);
        }
    }
}

} // namespace

int main()
{
    TestCollectivesOwnsCceBuild();
    TestCollectivesKernelSourcesAreScoped();
    TestHostRegistrationLivesInCollectives();
    TestPerfTraceCycleDivisorIsA5Specific();
    TestDeviceKernelArgsMatchHostLaunchAbi();
    TestBigDataAllGatherPerfStages();
    TestTwoNpuBigDataAllGatherPerfStages();
    TestOtherStandaloneCollectivePerfStages();
    TestStandaloneCollectiveWrappersHaveCoarsePerfTrace();
    TestCommDoesNotOwnCollectiveRuntime();
    return g_failures == 0 ? 0 : 1;
}
