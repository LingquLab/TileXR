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

std::string JoinPath(const std::string &base, const std::string &path)
{
    if (base.empty() || base[base.size() - 1] == '/') {
        return base + path;
    }
    return base + "/" + path;
}

bool ReadFile(const std::string &relativePath, std::string *contents)
{
    const std::string fullPath = JoinPath(kSourceRoot, relativePath);
    std::ifstream stream(fullPath.c_str());
    if (!stream.is_open()) {
        std::cerr << "missing file: " << relativePath << std::endl;
        ++g_failures;
        return false;
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    *contents = buffer.str();
    return true;
}

void CheckContains(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) == std::string::npos) {
        std::cerr << label << " missing: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) != std::string::npos) {
        std::cerr << label << " contains forbidden string: " << needle << std::endl;
        ++g_failures;
    }
}

void TestKernelUsesTileXRPeerMemory()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "extern \"C\" __global__ __aicore__ void tilexr_ep_dispatch_kernel");
    CheckContains(path, contents, "launch_tilexr_ep_dispatch_kernel");
    CheckContains(path, contents, "CommArgs");
    CheckContains(path, contents, "peerMems");
    CheckContains(path, contents, "GlobalTensor<GM_ADDR> peerMems");
    CheckContains(path, contents, "peerMems.GetValue(peer)");
    CheckContains(path, contents, "IPC_DATA_OFFSET");
    CheckContains(path, contents, "SyncCollectives");
    CheckContains(path, contents, "DataCopyPad");
    CheckContains(path, contents, "kEpStepWindowCleared");
    CheckContains(path, contents, "kEpStepDispatchReady");
    CheckContains(path, contents, "kEpStepDispatchDrained");
    CheckContains(path, contents, "LoadInt32FromGm");
    CheckContains(path, contents, "LoadAssistTupleFromGm");
    CheckContains(path, contents, "StoreWindowHeader");
    CheckContains(path, contents, "StoreSlotHeader");
    CheckContains(path, contents, "StoreAssistTuple");
    CheckContains(path, contents, "TileXREpFlushDispatchSlotHeaders");
    CheckContains(path, contents, "sourceWindow = TileXREpWindowBase");
    CheckNotContains(path, contents, "expertIds[");
    CheckNotContains(path, contents, "assistBase[item]");
    CheckNotContains(path, contents, "args->peerMems[peer]");
    CheckNotContains(path, contents, "slot->count");
    CheckNotContains(path, contents, "assist[index]");
}

void TestCrossNodeDispatchUsesUDMARegistry()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "tilexr_udma.h");
    CheckContains(path, contents, "TileXR::UDMARegistryEnabled(args)");
    CheckContains(path, contents, "TileXREpUsesUdmaWindow");
    CheckContains(path, contents, "TileXR::UDMAPutNbi<uint8_t>");
    CheckContains(path, contents, "TileXR::UDMAQuiet(args, dstRank)");
    CheckContains(path, contents, "TileXREpFlushDispatchSlotHeaders");
    CheckContains(path, contents, "sourceWindow = TileXREpWindowBase");
}

void TestCrossNodeDispatchPullsRemoteSlots()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "tilexr_ep_dispatch_cross_node_kernel");
    CheckContains(path, contents, "launch_tilexr_ep_dispatch_cross_node_kernel");
    CheckContains(path, contents, "TileXREpPullUdmaSlots");
    CheckContains(path, contents, "TileXR::UDMAGetNbi<uint8_t>");
    CheckContains(path, contents, "TileXREpNotifyUdmaReady");
    CheckContains(path, contents, "TileXREpWaitUdmaReady");
    CheckContains(path, contents, "TileXREpNotifyAllUdmaReady");
    CheckContains(path, contents, "TileXREpWaitAllUdmaReady");
    CheckContains(path, contents, "TileXR::UDMAPutSignalNbi<uint8_t>");
}

void TestCrossNodeDispatchSeparatesLocalAndRemotePeers()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXREpIsSameNodePeer");
    CheckContains(path, contents, "TileXREpUsesUdmaPeer");
    CheckContains(path, contents, "if (localRankSize <= 1)");
    CheckContains(path, contents, "TileXREpPublishLocalUdmaSlot");
    CheckContains(path, contents, "TileXREpDispatchWriteWindow");
    CheckContains(path, contents, "args->localRankSize");
    CheckContains(path, contents, "if (localRankSize > 1)");
    CheckContains(path, contents, "dstRank != rank && TileXREpIsSameNodePeer(rank, dstRank, localRankSize)");
    CheckContains(path, contents, "srcRank != rank &&");
    CheckContains(path, contents, "TileXREpIsSameNodePeer(rank, srcRank, localRankSize)");
    CheckContains(path, contents, "sameNodeSource ? rank : srcRank");
    CheckContains(path, contents, "!TileXREpIsSameNodePeer(rank, peer, localRankSize)");
}

void TestHostDispatchSplitsCrossNodeKernel()
{
    const std::string path = "src/ep/host/ep_kernel_launch.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "launch_tilexr_ep_dispatch_cross_node_kernel");
    CheckContains(path, contents, "TileXREpUsesCrossNodeKernel");
}

void TestDispatchHelpersLiveInDispatchHelperFile()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_helpers.h";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXREpRouteToDstRank");
    CheckContains(path, contents, "TileXREpCopyRoutePayload");
    CheckContains(path, contents, "TileXREpStoreDispatchSlotHeader");
    CheckContains(path, contents, "TileXREpStoreAssistTuple");
    CheckContains(path, contents, "localWindow + PayloadOffset(dstRank, slotBytes)");
}

void TestCombineHelpersLiveInCombineHelperFile()
{
    const std::string path = "src/ep/kernels/tilexr_ep_combine_helpers.h";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXREpDrainSourceWindow");
    CheckContains(path, contents, "TileXREpWaitDispatchSlotReady");
    CheckContains(path, contents, "slotMagic");
    CheckContains(path, contents, "TileXREpLoadAssistTuple");
    CheckContains(path, contents, "TileXREpGetCombineTokenId");
    CheckContains(path, contents, "TileXREpGetCombineTopKId");
    CheckContains(path, contents, "sourceWindow + SlotOffset(slotRank, slotBytes)");
}

void TestCombineKernelUsesTileXRPeerMemory()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "extern \"C\" __global__ __aicore__ void tilexr_ep_combine_kernel");
    CheckContains(path, contents, "launch_tilexr_ep_combine_kernel");
    CheckContains(path, contents, "kEpStepCombineWindowCleared");
    CheckContains(path, contents, "kEpStepCombineReady");
    CheckContains(path, contents, "ScatterCombineRows");
    CheckContains(path, contents, "DrainCombineRows");
    CheckContains(path, contents, "AccumulateRow");

    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "TileXREpLaunchCombineKernel");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "launch_tilexr_ep_combine_kernel");
    }
}

void TestDispatchDemoRunsCombine()
{
    const std::string path = "tests/ep/demo/tilexr_ep_dispatch_demo.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXRMoeEpCombine");
    CheckContains(path, contents, "ValidateCombineOutputs");
    CheckContains(path, contents, "combine validation");
}

void TestKernelForwardsActiveMask()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "xActiveMask");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "xActiveMaskGM");
    }

    std::string dispatchHelpers;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_helpers.h", &dispatchHelpers)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_helpers.h", dispatchHelpers, "TileXREpIsTokenActive");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_helpers.h", dispatchHelpers, "xActiveMaskGM == nullptr");
    }
}

void TestKernelForwardsExpertTokenNumsType()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "expertTokenNumsType");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "expertTokenNumsType");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpClearExpertTokenNums");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpFinalizeExpertTokenNums");
    }

    std::string combineHelpers;
    if (ReadFile("src/ep/kernels/tilexr_ep_combine_helpers.h", &combineHelpers)) {
        CheckContains("src/ep/kernels/tilexr_ep_combine_helpers.h", combineHelpers,
            "TileXREpIncrementExpertTokenNum");
        CheckContains("src/ep/kernels/tilexr_ep_combine_helpers.h", combineHelpers,
            "running += expertTokenNumsOut[localExpert]");
    }
}

void TestKernelForwardsTpRecvCountsOut()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "tpRecvCountsOut");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "tpWorldSize");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "tpRankId");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "tpRecvCountsOutGM");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpAppendTpGroupRows");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpTpGroupStartRank");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "tpWorldSize");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "tpRankId");
    }

    const std::string demoPath = "tests/ep/demo/tilexr_ep_dispatch_demo.cpp";
    std::string demo;
    if (ReadFile(demoPath, &demo)) {
        CheckContains(demoPath, demo, "TILEXR_EP_DEMO_TP_RECV_COUNTS");
        CheckContains(demoPath, demo, "TILEXR_EP_DEMO_TP_WORLD_SIZE");
        CheckContains(demoPath, demo, "TILEXR_EP_DEMO_TP_RANK_ID");
        CheckContains(demoPath, demo, "BuildExpectedTpRoutes");
        CheckContains(demoPath, demo, "ValidateTpRecvCounts");
    }
}

void TestDispatchDemoExercisesV2OptionalInputs()
{
    const std::string path = "tests/ep/demo/tilexr_ep_dispatch_demo.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TILEXR_EP_DEMO_ACTIVE_MASK");
    CheckContains(path, contents, "TILEXR_EP_DEMO_EXPERT_TOKEN_NUMS_TYPE");
    CheckContains(path, contents, "xActiveMaskDev");
    CheckContains(path, contents, "expertTokenNumsType");
}

void TestKernelForwardsSharedExpertConfig()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "sharedExpertNum");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "sharedExpertRankNum");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpRouteToDstRank");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "sharedExpertRankNum");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpLocalExpertCount");
    }

    std::string demo;
    if (ReadFile("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", &demo)) {
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "TILEXR_EP_DEMO_SHARED_EXPERT_NUM");
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "TILEXR_EP_DEMO_SHARED_EXPERT_RANK_NUM");
    }
}

void TestKernelForwardsStaticQuantConfig()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "scales");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "quantMode");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "scalesGM");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "quantMode");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpCopyStaticQuantRoutePayload");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpClampInt8");
    }

    std::string demo;
    if (ReadFile("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", &demo)) {
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "TILEXR_EP_DEMO_QUANT_MODE");
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "TILEXR_EP_DEMO_STATIC_QUANT_SCALE");
    }
}

void TestKernelForwardsPerTokenDynamicQuantConfig()
{
    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "kEpQuantModePerTokenDynamic");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel,
            "TileXREpCopyPerTokenDynamicQuantRoutePayload");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "dynamicScalesOutGM");
    }

    std::string demo;
    if (ReadFile("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", &demo)) {
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "usePerTokenDynamicQuant");
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "DynamicScaleForXValue");
    }
}

void TestClearLocalWindowDoesNotPreclearSlotHeaders()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckNotContains(path, contents,
        "for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {\n"
        "        StoreSlotHeader(localWindow + SlotOffset(srcRank, slotBytes), 0, srcRank, 0, 0, tBuf);\n"
        "    }");
}

void TestNoForbiddenDependencies()
{
    const std::vector<std::string> paths = {
        "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp",
        "src/ep/kernels/tilexr_ep_dispatch_helpers.h",
        "src/ep/kernels/tilexr_ep_combine_helpers.h",
        "src/ep/host/ep_kernel_launch.cpp",
        "src/ep/CMakeLists.txt",
    };
    const std::vector<std::string> forbidden = {
        "src/mc2",
        "3rdparty/ops-transformer",
        "GetHcclContext",
        "TileXRUDMARegister",
        "shmem",
    };

    for (std::vector<std::string>::const_iterator path = paths.begin(); path != paths.end(); ++path) {
        std::string contents;
        if (!ReadFile(*path, &contents)) {
            continue;
        }
        for (std::vector<std::string>::const_iterator needle = forbidden.begin(); needle != forbidden.end(); ++needle) {
            CheckNotContains(*path, contents, *needle);
        }
    }
}

} // namespace

int main()
{
    TestKernelUsesTileXRPeerMemory();
    TestCrossNodeDispatchUsesUDMARegistry();
    TestCrossNodeDispatchPullsRemoteSlots();
    TestCrossNodeDispatchSeparatesLocalAndRemotePeers();
    TestHostDispatchSplitsCrossNodeKernel();
    TestDispatchHelpersLiveInDispatchHelperFile();
    TestCombineHelpersLiveInCombineHelperFile();
    TestCombineKernelUsesTileXRPeerMemory();
    TestDispatchDemoRunsCombine();
    TestKernelForwardsActiveMask();
    TestKernelForwardsExpertTokenNumsType();
    TestKernelForwardsTpRecvCountsOut();
    TestDispatchDemoExercisesV2OptionalInputs();
    TestKernelForwardsSharedExpertConfig();
    TestKernelForwardsStaticQuantConfig();
    TestKernelForwardsPerTokenDynamicQuantConfig();
    TestClearLocalWindowDoesNotPreclearSlotHeaders();
    TestNoForbiddenDependencies();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR EP kernel source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR EP kernel source checks passed" << std::endl;
    return 0;
}
