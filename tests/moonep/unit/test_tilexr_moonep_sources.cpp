#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#ifdef TILEXR_SOURCE_ROOT
const char *kSourceRoot = TILEXR_SOURCE_ROOT;
#else
const char *kSourceRoot = ".";
#endif

std::string ReadFile(const std::string &relativePath)
{
    const std::string path = std::string(kSourceRoot) + "/" + relativePath;
    std::ifstream stream(path.c_str());
    if (!stream.is_open()) {
        std::cerr << "missing file: " << relativePath << std::endl;
        ++g_failures;
        return std::string();
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void Contains(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) == std::string::npos) {
        std::cerr << label << " missing: " << needle << std::endl;
        ++g_failures;
    }
}

void Excludes(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) != std::string::npos) {
        std::cerr << label << " contains forbidden text: " << needle << std::endl;
        ++g_failures;
    }
}

std::string Lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

void CheckOrdered(const std::string &label, const std::string &contents,
                  const std::vector<std::string> &needles)
{
    size_t cursor = 0;
    for (std::vector<std::string>::const_iterator it = needles.begin();
         it != needles.end(); ++it) {
        const size_t position = contents.find(*it, cursor);
        if (position == std::string::npos) {
            std::cerr << label << " missing ordered item: " << *it << std::endl;
            ++g_failures;
            return;
        }
        cursor = position + it->size();
    }
}

} // namespace

int main()
{
    const std::string header = ReadFile("src/include/tilexr_moonep.h");
    const std::string host = ReadFile("src/moonep/host/tilexr_moonep.cpp");
    const std::string cmake = ReadFile("src/moonep/CMakeLists.txt");
    const std::string dispatchCmake = ReadFile("src/moonep/dispatch/CMakeLists.txt");
    const std::string dispatchUrmaHost =
        ReadFile("src/moonep/dispatch/urma/host/dispatch_host.cpp");
    const std::string dispatchUrmaLaunch =
        ReadFile("src/moonep/dispatch/urma/host/dispatch_launch.cpp");
    const std::string dispatchUrmaKernel =
        ReadFile("src/moonep/dispatch/urma/kernels/tilexr_moonep_dispatch_kernel.cpp");
    const std::string combineCmake = ReadFile("src/moonep/combine/CMakeLists.txt");
    const std::string prefetchCmake = ReadFile("src/moonep/prefetch_weight/CMakeLists.txt");
    const std::string reduceCmake = ReadFile("src/moonep/reduce_grad/CMakeLists.txt");
    const std::string testCmake = ReadFile("tests/moonep/CMakeLists.txt");
    const std::string flowDemo =
        ReadFile("tests/moonep/demo/tilexr_moonep_flow_demo.cpp");
    const std::string flowRunner = ReadFile("tests/moonep/demo/run_a5.sh");
    const std::string stageHost =
        ReadFile("src/moonep/common/moonep_stage_host.h");
    const std::string prefetchHost =
        ReadFile("src/moonep/prefetch_weight/host/prefetch_weight_host.cpp");
    const std::string reduceHost =
        ReadFile("src/moonep/reduce_grad/host/reduce_grad_host.cpp");
    const std::vector<std::string> nativeStageHosts = {
        ReadFile("src/moonep/dispatch/host/dispatch_host.cpp"),
        ReadFile("src/moonep/combine/host/combine_host.cpp"),
    };

    Contains("public header", header, "extern \"C\"");
    Contains("public header", header, "TileXRMoonEpTensorV1");
    Contains("public header", header, "TileXRMoonEpPlanV1");
    Contains("public header", header, "registeredWorkspace");
    Contains("public header", header, "TileXRMoonEpDispatchGetWorkspaceSizeV1");
    Excludes("public header", header, "tilexr_api.h");
    Excludes("public header", header, "std::");

    Contains("host", host, "TileXRMoonEpPlannerGetWorkspaceSizeV3");
    Contains("host", host, "TileXRMoonEpPlannerV3");
    Contains("host", host, "cuSeqlens");
    Contains("host", host, "zeroFillRanges");
    Contains("host", host, "dupCounts");
    Contains("host", host, "TileXRMoonEpRunDispatchV1");
    Contains("host", host, "TileXRMoonEpRunDispatchUrmaV1");
    Contains("host", host, "TileXRMoonEpRunCombineV1");
    Excludes("host dispatch stub", host,
        "RunLocalStub(args, stream, StubStage::Dispatch)");
    Excludes("host combine stub", host,
        "RunLocalStub(args, stream, StubStage::Combine)");
    Contains("host", host, "TileXRMoonEpRunPrefetchWeightV1");
    Contains("host", host, "RunReduceGradV1Stub");
    Contains("host", host, "aclrtMemsetAsync");
    Contains("host", host, "aclrtMemcpyAsync");
    Contains("host", host, "ACL_MEMCPY_DEVICE_TO_DEVICE");
    Excludes("host", host, "aclrtSynchronizeStream");

    Contains("CMake", cmake, "add_library(tilexr-moonep SHARED");
    Contains("CMake", cmake, "SOVERSION 1");
    Contains("CMake", cmake, "INSTALL_RPATH \"$ORIGIN\"");
    Contains("CMake", cmake, "tilexr-moonep-planner");
    Contains("CMake", cmake, "add_subdirectory(dispatch)");
    Contains("CMake", cmake, "add_subdirectory(combine)");
    Contains("CMake", cmake, "tilexr-moonep-dispatch");
    Contains("CMake", cmake, "tilexr-moonep-combine");
    Contains("CMake", cmake, "add_subdirectory(prefetch_weight)");
    Contains("CMake", cmake, "add_subdirectory(reduce_grad)");
    Contains("CMake", cmake, "tilexr-moonep-prefetch-weight");
    Contains("CMake", cmake, "tilexr-moonep-reduce-grad");
    Excludes("CMake", cmake, "devlib");

    Contains("dispatch CMake", dispatchCmake, "add_library(tilexr-moonep-dispatch SHARED");
    Contains("dispatch CMake", dispatchCmake, "TILEXR_MOONEP_DISPATCH_KERNEL_EMBED_CPP");
    Contains("dispatch CMake", dispatchCmake, "moonep_kernel_registration.cpp");
    Excludes("dispatch CMake", dispatchCmake, "libtilexr_moonep_dispatch_kernel.so");
    Contains("dispatch CMake", dispatchCmake, "--cce-aicore-arch=dav-c310-vec");
    Contains("dispatch CMake", dispatchCmake, "tilexr_moonep_dispatch_urma_kernel");
    Contains("dispatch CMake", dispatchCmake, "--cce-auto-sync");
    Contains("dispatch CMake", dispatchCmake, "-DCATLASS_ARCH=3510");
    Contains("dispatch CMake", dispatchCmake, "TILEXR_MOONEP_DISPATCH_ENABLE_DFX");
    Contains("dispatch CMake", dispatchCmake, "cxx_std_14");
    Contains("dispatch CMake", dispatchCmake, "BUILD_WITH_INSTALL_RPATH TRUE");
    Contains("dispatch CMake", dispatchCmake, "INSTALL_RPATH \"$ORIGIN\"");
    Contains("URMA dispatch Host", dispatchUrmaHost,
        "TileXRMoonEpBuildDispatchUrmaLayout");
    Contains("URMA dispatch Host", dispatchUrmaHost, "registeredWorkspaceBytes");
    Contains("URMA dispatch Host", dispatchUrmaHost,
        "params.zeroFillRanges = static_cast<const int32_t *>(args->plan->zeroFillRanges)");
    Contains("URMA dispatch Host", dispatchUrmaHost,
        "zeroFillRangeCount > UINT32_MAX");
    Contains("URMA dispatch launch", dispatchUrmaLaunch, "params.zeroFillRanges");
    Contains("URMA dispatch launch", dispatchUrmaLaunch, "zeroFillRangeCount");
    Contains("URMA dispatch kernel", dispatchUrmaKernel,
        "tilexr_moonep_dispatch_urma_kernel");
    Contains("URMA dispatch kernel", dispatchUrmaKernel, "destinationCapacityArg");
    Contains("URMA dispatch kernel", dispatchUrmaKernel,
        "ClearDispatchZeroFillRanges");
    Contains("URMA dispatch kernel", dispatchUrmaKernel, "udmaIssueQp0Buf");
    Contains("URMA dispatch kernel", dispatchUrmaKernel, "udmaIssueQp1Buf");
    Excludes("URMA dispatch kernel", dispatchUrmaKernel, "<<<");
    Contains("combine CMake", combineCmake, "add_library(tilexr-moonep-combine SHARED");
    Contains("combine CMake", combineCmake, "TILEXR_MOONEP_COMBINE_KERNEL_EMBED_CPP");
    Contains("combine CMake", combineCmake, "moonep_kernel_registration.cpp");
    Excludes("combine CMake", combineCmake, "libtilexr_moonep_combine_kernel.so");
    Contains("combine CMake", combineCmake, "--cce-aicore-arch=dav-c310-vec");
    Contains("combine CMake", combineCmake, "cxx_std_14");
    Contains("combine CMake", combineCmake, "BUILD_WITH_INSTALL_RPATH TRUE");
    Contains("combine CMake", combineCmake, "INSTALL_RPATH \"$ORIGIN\"");
    Contains("prefetch CMake", prefetchCmake,
        "add_library(tilexr-moonep-prefetch-weight SHARED");
    Contains("prefetch CMake", prefetchCmake, "BUILD_WITH_INSTALL_RPATH TRUE");
    Contains("prefetch CMake", prefetchCmake, "INSTALL_RPATH \"$ORIGIN\"");
    Contains("reduce CMake", reduceCmake,
        "add_library(tilexr-moonep-reduce-grad SHARED");
    Contains("reduce CMake", reduceCmake, "BUILD_WITH_INSTALL_RPATH TRUE");
    Contains("reduce CMake", reduceCmake, "INSTALL_RPATH \"$ORIGIN\"");

    const std::string nativeCmake = Lower(cmake + dispatchCmake + combineCmake +
        prefetchCmake + reduceCmake);
    Excludes("native CMake", nativeCmake, "3rdparty/moonep");
    Excludes("native CMake", nativeCmake, "reference/");
    Excludes("native CMake", nativeCmake, "src/ep");
    Excludes("native CMake", nativeCmake, "hccl");
    Excludes("native CMake", nativeCmake, "shmem");
    Excludes("dispatch runtime path", Lower(dispatchCmake), "install_rpath \"${ascend_home_path}");
    Excludes("combine runtime path", Lower(combineCmake), "install_rpath \"${ascend_home_path}");
    Excludes("prefetch runtime path", Lower(prefetchCmake),
        "install_rpath \"${ascend_home_path}");
    Excludes("reduce runtime path", Lower(reduceCmake),
        "install_rpath \"${ascend_home_path}");

    Excludes("public header", Lower(header), "hccl");
    Excludes("host", Lower(host), "hccl");
    Excludes("CMake", Lower(cmake), "hccl");

    Contains("common stage Host", stageHost, "PrepareStageHost");
    Contains("common stage Host", stageHost, "PrepareStageDevice");
    Excludes("common stage Host", stageHost, "localRankSize !=");
    for (std::vector<std::string>::const_iterator it = nativeStageHosts.begin();
         it != nativeStageHosts.end(); ++it) {
        Contains("native stage Host", *it, "#include \"moonep_stage_host.h\"");
        Contains("native stage Host", *it, "PrepareStageHost(");
        Contains("native stage Host", *it, "PrepareStageDevice(");
        Excludes("native stage Host", *it, "bool LocalityValid(");
        Excludes("native stage Host", *it, "bool PeerWindowsReady(");
    }
    Excludes("prefetch Host", prefetchHost, "moonep_stage_host.h");
    Excludes("prefetch Host", prefetchHost, "PrepareStageHost(");
    Excludes("prefetch Host", prefetchHost, "PrepareStageDevice(");
    Excludes("prefetch Host", prefetchHost, "peerMems");
    Excludes("prefetch Host", prefetchHost, "localRankSize !=");
    Contains("prefetch Host", prefetchHost, "TileXRGetUDMARegistryHost");
    Contains("ReduceGrad Host", reduceHost, "TileXRMoonEpPrepareReduceGradLayout");
    Contains("ReduceGrad Host", reduceHost, "TileXRMoonEpReduceGradV2");
    Excludes("ReduceGrad Host", reduceHost, "moonep_stage_host.h");

    Contains("test CMake", testCmake, "if(TARGET tilexr-moonep)");
    Contains("test CMake", testCmake, "add_executable(tilexr_moonep_flow_demo");
    Contains("test CMake", testCmake, "tilexr-moonep");
    Contains("test CMake", testCmake, "tilexr-moonep-reduce-grad");
    Contains("flow demo", flowDemo, "TileXRCommInitRankLocal");
    Contains("flow demo", flowDemo, "TileXRUDMARegister prefetch arena");
    Contains("flow demo", flowDemo, "TileXRUDMAUnregister prefetch arena");
    Contains("flow demo", flowDemo, "TileXRMoonEpGetCapabilitiesV2");
    Contains("flow demo", flowDemo, "TileXRMoonEpReduceGradGetWorkspaceSizeV2");
    Contains("flow demo", flowDemo, "TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER");
    Excludes("flow demo", flowDemo, "TileXRMoonEpReduceGradV1(&reduceGrad");
    Contains("flow demo", flowDemo, "nvS != static_cast<int64_t>(routeCount)");
    Contains("flow demo", flowDemo, "planning_status=");
    Contains("flow demo", flowDemo, "dispatch_status=");
    Contains("flow demo", flowDemo, "prefetch_weight_status=");
    Contains("flow demo", flowDemo, "combine_status=");
    Contains("flow demo", flowDemo, "reduce_grad_status=");
    Contains("flow demo", flowDemo, "cuHost.back()");
    Contains("flow demo", flowDemo, "TILEXR_MOONEP_DTYPE_BFLOAT16");
    Contains("flow demo", flowDemo, "TILEXR_MOONEP_DTYPE_FLOAT32");
    Contains("flow demo", flowDemo, "BuildExpectedDispatch");
    Contains("flow demo", flowDemo, "BuildExpectedCombine");
    Contains("flow demo", flowDemo, "route weights");
    Contains("flow demo", flowDemo, "dispatch=native");
    Contains("flow demo", flowDemo, "prefetch_weight=native");
    Contains("flow demo", flowDemo, "combine=native");
    Contains("flow demo", flowDemo, "reduce_grad=native");
    Excludes("flow demo", flowDemo, "CheckPrefixAndZeroTail");
    Contains("flow demo", flowDemo, "torch_validated=false");
    Contains("flow demo", flowDemo, "transport_performance_valid=false");
    CheckOrdered("native flow", flowDemo, {
        "TileXRMoonEpPlanningV1(&planning",
        "TileXRMoonEpReduceGradGetWorkspaceSizeV2(",
        "TileXRMoonEpDispatchV1(&forwardDispatch",
        "TileXRMoonEpPrefetchWeightV1(&prefetch",
        "TileXRMoonEpCombineV1(&forwardCombine",
        "TileXRMoonEpDispatchV1(&backwardDispatch",
        "TileXRMoonEpCombineV1(&backwardCombine",
        "TileXRMoonEpReduceGradV2(&reduceGrad",
    });
    CheckOrdered("native flow cleanup", flowDemo, {
        "\"local completion synchronize\"",
        "DemoBarrierAll(options.rank, options.world)",
        "Cleanup(&resources)",
    });
    Contains("flow demo", flowDemo, "TILEXR_MOONEP_FLOW_BARRIER_ADDR");
    Contains("flow runner", flowRunner, "block_dim=$((64 / ranks_per_device))");
    Contains("flow runner", flowRunner, "export TILEXR_ENABLE_UDMA=1");
    Contains("flow runner", flowRunner, "${SCRIPT_DIR}/tilexr_moonep_flow_demo");
    Contains("flow runner", flowRunner, "device=$((rank % physical_device_count))");
    Contains("flow runner", flowRunner, "TILEXR_MOONEP_PLANNER_BLOCK_DIM");
    Contains("flow runner", flowRunner, "torch_validated=false");
    Contains("flow runner", flowRunner, "transport_performance_valid=false");
    Excludes("flow demo", Lower(flowDemo), "hccl");
    Excludes("flow runner", Lower(flowRunner), "hccl");
    return g_failures == 0 ? 0 : 1;
}
