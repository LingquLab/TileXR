#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

#ifdef TILEXR_SOURCE_ROOT
const char *kSourceRoot = TILEXR_SOURCE_ROOT;
#else
const char *kSourceRoot = ".";
#endif

std::string ReadFile(const std::string &relativePath)
{
    std::ifstream stream((std::string(kSourceRoot) + "/" + relativePath).c_str());
    if (!stream.is_open()) {
        std::cerr << "missing file: " << relativePath << '\n';
        ++failures;
        return std::string();
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    return contents.str();
}

std::string Lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

void Contains(const char *label, const std::string &contents, const char *needle)
{
    if (contents.find(needle) == std::string::npos) {
        std::cerr << label << " missing: " << needle << '\n';
        ++failures;
    }
}

void Excludes(const char *label, const std::string &contents, const char *needle)
{
    if (contents.find(needle) != std::string::npos) {
        std::cerr << label << " contains forbidden text: " << needle << '\n';
        ++failures;
    }
}

} // namespace

int main()
{
    const std::string plannerHost =
        ReadFile("src/moonep/planner/host/planner_host.cpp");
    const std::string plannerKernel =
        ReadFile("src/moonep/planner/kernels/tilexr_moonep_planner_kernel.cpp");
    const std::string dispatchCommon =
        ReadFile("src/moonep/dispatch/common/dispatch_common.h");
    const std::string dispatchHost =
        ReadFile("src/moonep/dispatch/host/dispatch_host.cpp");
    const std::string dispatchLaunch =
        ReadFile("src/moonep/dispatch/host/dispatch_launch.cpp");
    const std::string dispatchKernel =
        ReadFile("src/moonep/dispatch/kernels/tilexr_moonep_dispatch_kernel.cpp");
    const std::string dispatchCmake = ReadFile("src/moonep/dispatch/CMakeLists.txt");
    const std::string combineCommon =
        ReadFile("src/moonep/combine/common/combine_common.h");
    const std::string combineHost =
        ReadFile("src/moonep/combine/host/combine_host.cpp");
    const std::string combineLaunch =
        ReadFile("src/moonep/combine/host/combine_launch.cpp");
    const std::string combineKernel =
        ReadFile("src/moonep/combine/kernels/tilexr_moonep_combine_kernel.cpp");
    const std::string combineCmake = ReadFile("src/moonep/combine/CMakeLists.txt");
    const std::string prefetchLaunch =
        ReadFile("src/moonep/prefetch_weight/host/prefetch_weight_launch.cpp");
    const std::string prefetchHost =
        ReadFile("src/moonep/prefetch_weight/host/prefetch_weight_host.cpp");
    const std::string prefetchKernel = ReadFile(
        "src/moonep/prefetch_weight/kernels/tilexr_moonep_prefetch_weight_kernel.cpp");
    const std::string prefetchCmake = ReadFile("src/moonep/prefetch_weight/CMakeLists.txt");
    const std::string reduceLaunch =
        ReadFile("src/moonep/reduce_grad/host/reduce_grad_launch.cpp");
    const std::string reduceHost =
        ReadFile("src/moonep/reduce_grad/host/reduce_grad_host.cpp");
    const std::string reduceKernel = ReadFile(
        "src/moonep/reduce_grad/kernels/tilexr_moonep_reduce_grad_kernel.cpp");
    const std::string reduceCmake = ReadFile("src/moonep/reduce_grad/CMakeLists.txt");
    const std::string registration =
        ReadFile("src/moonep/common/moonep_kernel_registration.h");
    const std::string registrationImpl =
        ReadFile("src/moonep/common/moonep_kernel_registration.cpp");
    const std::string stageHost =
        ReadFile("src/moonep/common/moonep_stage_host.h");
    const std::string kernelLaunch =
        ReadFile("src/moonep/common/moonep_kernel_launch.h");
    const std::string kernelBuild = ReadFile("src/moonep/cmake/MoonEpKernel.cmake");

    Contains("planner host", plannerHost,
        "commArgs->localRankSize != commArgs->rankSize");
    Contains("planner host", plannerHost,
        "commArgs.localRankSize != commArgs.rankSize");
    Contains("planner kernel", plannerKernel, "TileXR::IPC_DATA_OFFSET");
    Excludes("planner kernel", Lower(plannerKernel), "udma");
    Contains("dispatch common", dispatchCommon, "moonep_peer_window.h");
    Contains("common stage Host", stageHost,
        "context->hostArgs->localRankSize != context->hostArgs->rankSize");
    Contains("registration implementation", registrationImpl, "rtDevBinaryRegister");
    Contains("registration implementation", registrationImpl, "rtFunctionRegister");
    Contains("common kernel launch", kernelLaunch, "rtKernelLaunchWithFlagV2");
    Contains("common kernel launch", kernelLaunch, "LaunchRegisteredMoonEpKernel");
    Contains("dispatch launch", dispatchLaunch, "#include \"moonep_kernel_launch.h\"");
    Contains("dispatch launch", dispatchLaunch, "LaunchRegisteredMoonEpKernel(");
    Excludes("dispatch launch", dispatchLaunch, "rtKernelLaunchWithFlagV2");
    Contains("dispatch launch", dispatchLaunch, "kDispatchKernelSignature");
    Excludes("dispatch launch", dispatchLaunch, "launch_tilexr_moonep_dispatch_kernel");
    Contains("dispatch kernel", dispatchKernel,
        "extern \"C\" __global__ __aicore__ void tilexr_moonep_dispatch_kernel");
    Excludes("dispatch kernel", dispatchKernel, "launch_tilexr_moonep_dispatch_kernel");
    Excludes("dispatch kernel", dispatchKernel, "<<<");
    Contains("kernel build", kernelBuild, "--cce-aicore-only");
    Contains("dispatch cmake", dispatchCmake, "MoonEpKernel.cmake");
    Excludes("dispatch cmake", dispatchCmake, "libtilexr_moonep_dispatch_kernel.so");
    Contains("dispatch kernel", dispatchKernel, "TileXR::CommArgs");
    Contains("dispatch kernel", dispatchKernel, "TileXR::IPC_DATA_OFFSET");
    Contains("dispatch kernel", dispatchKernel, "SyncCollectives");
    Contains("dispatch kernel", dispatchKernel, "ClearLocalWindow");
    Contains("dispatch kernel", dispatchKernel, "CopyBytesGmToGm");
    Contains("dispatch kernel", dispatchKernel, "hiddenChunkStride");
    Contains("dispatch kernel", dispatchKernel, "routeWeightsSk");
    Contains("dispatch kernel", dispatchKernel, "zeroFillRanges");
    Contains("dispatch kernel", dispatchKernel, "BuildDuplicatePlan");
    Contains("dispatch kernel", dispatchKernel, "prefix != duplicateCount");
    Contains("dispatch kernel", dispatchKernel, "writeIndex >= duplicateCount");
    Contains("dispatch kernel", dispatchKernel, "ExpandDuplicateRows");
    Contains("dispatch kernel", dispatchKernel, "dedupParentsOffset");
    Contains("dispatch kernel", dispatchKernel, "ChunkStep");
    Contains("dispatch kernel", dispatchKernel, "DataCacheCleanAndInvalid");
    Contains("dispatch kernel", dispatchKernel, "static_cast<int64_t>(encoded)");
    Contains("dispatch kernel", dispatchKernel, "-encoded64 - 1");
    Contains("dispatch kernel", dispatchKernel, "kMoonEpDispatchWindowClearedStep");
    Contains("dispatch kernel", dispatchKernel, "kMoonEpDispatchDataReadyStep");
    Contains("dispatch kernel", dispatchKernel, "kMoonEpDispatchWindowDrainedStep");
    Contains("dispatch kernel", dispatchKernel, "kMoonEpDispatchFailedStep");
    Contains("dispatch kernel", dispatchKernel, "kMoonEpDispatchStatusInvalidRoute");
    Contains("dispatch kernel", dispatchKernel, "kMoonEpDispatchStatusTimeoutBase");
    Contains("dispatch kernel", dispatchKernel, "AscendC::PipeBarrier<PIPE_ALL>()");
    Excludes("dispatch kernel", dispatchKernel, "WaitRankInnerFlag");
    Excludes("dispatch kernel", dispatchKernel, "aclrtSynchronizeStream");
    Excludes("dispatch kernel", Lower(dispatchKernel), "udma");
    Contains("combine common", combineCommon, "moonep_peer_window.h");
    Contains("combine launch", combineLaunch, "#include \"moonep_kernel_launch.h\"");
    Contains("combine launch", combineLaunch, "LaunchRegisteredMoonEpKernel(");
    Excludes("combine launch", combineLaunch, "rtKernelLaunchWithFlagV2");
    Contains("combine launch", combineLaunch, "kCombineKernelSignature");
    Excludes("combine launch", combineLaunch, "launch_tilexr_moonep_combine_kernel");
    Contains("combine kernel", combineKernel,
        "extern \"C\" __global__ __aicore__ void tilexr_moonep_combine_kernel");
    Excludes("combine kernel", combineKernel, "launch_tilexr_moonep_combine_kernel");
    Excludes("combine kernel", combineKernel, "<<<");
    Contains("combine cmake", combineCmake, "MoonEpKernel.cmake");
    Excludes("combine cmake", combineCmake, "libtilexr_moonep_combine_kernel.so");
    Contains("combine kernel", combineKernel, "TileXR::CommArgs");
    Contains("combine kernel", combineKernel, "TileXR::IPC_DATA_OFFSET");
    Contains("combine kernel", combineKernel, "SyncCollectives");
    Contains("combine kernel", combineKernel, "PublishLocalInput");
    Contains("combine kernel", combineKernel, "CopyBytesGmToGm");
    Contains("combine kernel", combineKernel, "hiddenChunkStride");
    Contains("combine kernel", combineKernel, "routeWeightsNvs");
    Contains("combine kernel", combineKernel, "PreReduceDuplicates");
    Contains("combine kernel", combineKernel, "ChunkStep");
    Contains("combine kernel", combineKernel, "DataCacheCleanAndInvalid");
    Contains("combine kernel", combineKernel, "static_cast<int64_t>(encoded)");
    Contains("combine kernel", combineKernel, "-encoded64 - 1");
    Contains("combine kernel", combineKernel, "ReduceHiddenChunk");
    Contains("combine kernel", combineKernel, "AscendC::LocalTensor<bfloat16_t>");
    Contains("combine kernel", combineKernel, "AscendC::LocalTensor<float>");
    Contains("combine kernel", combineKernel, "AscendC::RoundMode::CAST_NONE");
    Contains("combine kernel", combineKernel, "AscendC::Add");
    Contains("combine kernel", combineKernel, "AscendC::RoundMode::CAST_RINT");
    Contains("combine kernel", combineKernel, "GatherWeights");
    Contains("combine kernel", combineKernel, "kMoonEpCombineDataReadyStep");
    Contains("combine kernel", combineKernel, "kMoonEpCombineWindowDrainedStep");
    Contains("combine kernel", combineKernel, "kMoonEpCombineFailedStep");
    Contains("combine kernel", combineKernel, "kMoonEpCombineStatusInvalidRoute");
    Contains("combine kernel", combineKernel, "kMoonEpCombineStatusTimeoutBase");
    Contains("combine kernel", combineKernel, "AscendC::PipeBarrier<PIPE_ALL>()");
    Excludes("combine kernel", combineKernel, "WaitRankInnerFlag");
    Excludes("combine kernel", combineKernel, "aclrtSynchronizeStream");
    Excludes("combine kernel", Lower(combineKernel), "udma");
    Contains("prefetch launch", prefetchLaunch, "#include \"moonep_kernel_launch.h\"");
    Contains("prefetch launch", prefetchLaunch, "LaunchRegisteredMoonEpKernel(");
    Excludes("prefetch launch", prefetchLaunch, "rtKernelLaunchWithFlagV2");
    Contains("prefetch launch", prefetchLaunch, "kPrefetchWeightKernelSignature");
    Contains("prefetch kernel", prefetchKernel,
        "extern \"C\" __global__ __aicore__ void tilexr_moonep_prefetch_weight_kernel");
    Contains("prefetch kernel", prefetchKernel, "expertsToCopy");
    Contains("prefetch kernel", prefetchKernel, "PublishOwners");
    Contains("prefetch kernel", prefetchKernel, "e_ + slot");
    Contains("prefetch kernel", prefetchKernel, "CopyBytesGmToGm");
    Contains("prefetch kernel", prefetchKernel, "DataCacheCleanAndInvalid");
    Excludes("prefetch kernel", prefetchKernel, "<<<");
    Excludes("prefetch launch", prefetchLaunch, "launch_tilexr_moonep_prefetch_weight_kernel");
    Contains("prefetch cmake", prefetchCmake, "MoonEpKernel.cmake");
    Excludes("prefetch cmake", prefetchCmake, "libtilexr_moonep_prefetch_weight_kernel.so");
    Excludes("prefetch kernel", Lower(prefetchKernel), "udma");
    Contains("reduce launch", reduceLaunch, "#include \"moonep_kernel_launch.h\"");
    Contains("reduce launch", reduceLaunch, "LaunchRegisteredMoonEpKernel(");
    Excludes("reduce launch", reduceLaunch, "rtKernelLaunchWithFlagV2");
    Contains("reduce launch", reduceLaunch, "kReduceGradKernelSignature");
    Contains("reduce kernel", reduceKernel,
        "extern \"C\" __global__ __aicore__ void tilexr_moonep_reduce_grad_kernel");
    Contains("reduce kernel", reduceKernel, "AccumulateOwned");
    Contains("reduce kernel", reduceKernel, "sourceRank");
    Contains("reduce kernel", reduceKernel, "AscendC::Add");
    Contains("reduce kernel", reduceKernel, "ClearLocalLiveSlots");
    Contains("reduce kernel", reduceKernel, "DataCacheCleanAndInvalid");
    Excludes("reduce kernel", reduceKernel, "<<<");
    Excludes("reduce launch", reduceLaunch, "launch_tilexr_moonep_reduce_grad_kernel");
    Contains("reduce cmake", reduceCmake, "MoonEpKernel.cmake");
    Excludes("reduce cmake", reduceCmake, "libtilexr_moonep_reduce_grad_kernel.so");
    Excludes("reduce kernel", Lower(reduceKernel), "udma");
    Contains("registration", registration, "kPlannerKernelSignature");
    Contains("registration", registration, "kDispatchKernelSignature");
    Contains("registration", registration, "kCombineKernelSignature");
    Contains("registration", registration, "kPrefetchWeightKernelSignature");
    Contains("registration", registration, "kReduceGradKernelSignature");

    const std::string active = Lower(plannerHost + plannerKernel + dispatchCommon +
        dispatchHost + dispatchLaunch + dispatchKernel + combineCommon + combineHost +
        combineLaunch + combineKernel + prefetchHost + prefetchLaunch + prefetchKernel +
        reduceHost + reduceLaunch + reduceKernel + stageHost + kernelLaunch);
    Excludes("active MoonEP sources", active, "udma");
    Excludes("active MoonEP sources", active, "3rdparty/moonep");
    Excludes("active MoonEP sources", active, "reference/");
    Excludes("active MoonEP sources", active, "src/ep");
    Excludes("active MoonEP sources", active, "hccl");
    Excludes("active MoonEP sources", active, "shmem");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
