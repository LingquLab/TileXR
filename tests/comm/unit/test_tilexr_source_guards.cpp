#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    std::ifstream input(RepoPath(path).c_str());
    if (!input.is_open()) {
        std::cerr << "failed to open " << RepoPath(path) << std::endl;
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
        std::cerr << "expected text not found in " << path << ": " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string& path, const std::string& text, const std::string& needle)
{
    const auto pos = text.find(needle);
    if (pos != std::string::npos) {
        std::cerr << "unexpected text in " << path << ": " << needle
                  << " at byte " << pos << std::endl;
        ++g_failures;
    }
}

std::string RunCommand(const std::string& command)
{
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        std::cerr << "failed to run command: " << command << std::endl;
        ++g_failures;
        return {};
    }

    std::string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

    const int status = pclose(pipe);
    if (status != 0) {
        std::cerr << "command failed: " << command << std::endl;
        ++g_failures;
    }
    return output;
}

std::string ShellQuote(const std::string& value)
{
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::vector<std::string> SplitPathspec(const std::string& pathspec)
{
    std::vector<std::string> parts;
    std::istringstream input(pathspec);
    std::string part;
    while (input >> part) {
        parts.push_back(part);
    }
    return parts;
}

std::string StripTrailingDot(const std::string& path)
{
    if (path.size() >= 2U && path.substr(path.size() - 2U) == "/.") {
        return path.substr(0, path.size() - 2U);
    }
    return path;
}

std::string ListFilesCommand(const std::string& pathspec)
{
    const std::string root = StripTrailingDot(RepoPath("."));
    std::string command = "if git -C " + ShellQuote(root) +
        " rev-parse --is-inside-work-tree >/dev/null 2>&1; then git -C " +
        ShellQuote(root) + " ls-files --cached --others --exclude-standard -- " + pathspec + "; else ";
    for (const auto& part : SplitPathspec(pathspec)) {
        command += "find " + ShellQuote(root + "/" + part) +
            " -type f 2>/dev/null | sed " + ShellQuote("s#^" + root + "/##") + "; ";
    }
    command += "true; fi";
    return command;
}

void CheckNoLineContains(const std::string& description, const std::string& text, const std::string& needle)
{
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find(needle) != std::string::npos) {
            std::cerr << "unexpected " << description << ": " << line << std::endl;
            ++g_failures;
        }
    }
}

void CheckTrackedFilesDoNotContain(
    const std::string& description,
    const std::string& gitPathspec,
    const std::vector<std::string>& forbiddenNeedles)
{
    const auto trackedFiles = RunCommand(ListFilesCommand(gitPathspec));

    std::istringstream paths(trackedFiles);
    std::string path;
    while (std::getline(paths, path)) {
        if (path.empty()) {
            continue;
        }
        const auto text = ReadFile(path);
        for (const auto& needle : forbiddenNeedles) {
            const auto pos = text.find(needle);
            if (pos != std::string::npos) {
                std::cerr << "unexpected " << description << " in " << path
                          << ": " << needle << " at byte " << pos << std::endl;
                ++g_failures;
            }
        }
    }
}

bool ContainsPath(const std::vector<std::string>& paths, const std::string& path)
{
    for (const auto& item : paths) {
        if (item == path) {
            return true;
        }
    }
    return false;
}

void CheckTrackedFilesContainNeedleOnly(
    const std::string& description,
    const std::string& gitPathspec,
    const std::string& needle,
    const std::vector<std::string>& allowedPaths)
{
    const auto trackedFiles = RunCommand(ListFilesCommand(gitPathspec));

    std::istringstream paths(trackedFiles);
    std::string path;
    while (std::getline(paths, path)) {
        if (path.empty()) {
            continue;
        }
        const auto text = ReadFile(path);
        const auto pos = text.find(needle);
        if (pos != std::string::npos && !ContainsPath(allowedPaths, path)) {
            std::cerr << "unexpected " << description << " in " << path
                      << ": " << needle << " at byte " << pos << std::endl;
            ++g_failures;
        }
    }

    for (const auto& allowedPath : allowedPaths) {
        CheckContains(allowedPath, ReadFile(allowedPath), needle);
    }
}

void TestOpenSourceTarballsAreNotTracked()
{
    const auto trackedFiles = RunCommand(ListFilesCommand("3rdparty/open_source"));

    CheckNoLineContains("tracked open-source dependency archive", trackedFiles, ".tar.gz");
    CheckNoLineContains("tracked open-source dependency archive", trackedFiles, ".tar.xz");
}

void TestCommInitChecksDeviceCommArgsSync()
{
    const std::string path = "src/comm/tilexr_comm.cpp";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "InitMem();\n    g_localPeerMemMap");
    CheckNotContains(path, text, "    SyncCommArgs();");
    CheckContains(path, text, "ret = SyncCommArgs();");
    CheckContains(path, text, "ret = InitMem();");
}

void TestCWrappersDoNotPublishFailedCommunicators()
{
    const std::string path = "src/comm/comm_wrap.cpp";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "*comm = c;\n    int ret = c->Init();");
    CheckNotContains(path, text, "*comm = c;\n    int ret = c->InitThread");
    CheckNotContains(path, text, "comms[i] = new (std::nothrow) TileXRComm");
    CheckContains(path, text, "c.release()");
    CheckContains(path, text, "commHolders[i].release()");
}

void TestDumpInitCleansFailedAllocations()
{
    const std::string path = "src/comm/tilexr_comm.cpp";
    const auto text = ReadFile(path);

    CheckContains(path, text, "aclrtFree(dumpAddr);");
    CheckContains(path, text, "std::free(memory);");
}

void TestSocketExchangeUsesDirectConnectionsOnly()
{
    const std::string cppPath = "src/comm/tools/socket/tilexr_sock_exchange.cpp";
    const std::string headerPath = "src/comm/tools/socket/tilexr_sock_exchange.h";
    const auto cppText = ReadFile(cppPath);
    const auto headerText = ReadFile(headerPath);

    CheckNotContains(cppPath, cppText, "StartSecureTunnel");
    CheckNotContains(headerPath, headerText, "StartSecureTunnel");
    CheckNotContains(cppPath, cppText, "popen");
    CheckNotContains(cppPath, cppText, "/usr/bin/ssh");
    CheckNotContains(headerPath, headerText, "FILE* pipe_");
    CheckNotContains(headerPath, headerText, "lockFileDescriptor_");
    CheckContains(cppPath, cppText, "return Connect();");
}

void TestRuntimeEnvDoesNotPrependCannDevlib()
{
    const std::string path = "scripts/common_env.sh";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "${ASCEND_HOME_PATH}/${TILEXR_OS_ARCH}-linux/devlib");
}

void TestRuntimeEnvUsesReadableDriverShim()
{
    const std::string path = "scripts/common_env.sh";
    const auto text = ReadFile(path);

    CheckContains(path, text, "TILEXR_DRIVER_SHIM_HOME");
    CheckContains(path, text, "[ ! -r \"${ASCEND_DRIVER_PATH}/kernel/inc\" ]");
    CheckContains(path, text, "${ASCEND_HOME_PATH}/${TILEXR_OS_ARCH}-linux/include/driver");
    CheckContains(path, text, "export ASCEND_DRIVER_PATH=${TILEXR_DRIVER_SHIM_HOME}");
}

void TestRootCMakeRespectsAscendDriverOverride()
{
    const std::string path = "CMakeLists.txt";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)");
    CheckContains(path, text, "set(_tilexr_default_ascend_driver_path \"$ENV{ASCEND_DRIVER_PATH}\")");
    CheckContains(path, text, "set(ASCEND_DRIVER_PATH \"${_tilexr_default_ascend_driver_path}\" CACHE PATH");
}

void TestCommBuildIncludesProfilingHeaders()
{
    const std::string rootPath = "CMakeLists.txt";
    const std::string commPath = "src/comm/CMakeLists.txt";
    const auto rootText = ReadFile(rootPath);
    const auto commText = ReadFile(commPath);

    CheckContains(rootPath, rootText, "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/profiling/");
    CheckContains(commPath, commText, "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/profiling/");
}

void TestChipNameResolverRecognizesAscend950PrVariants()
{
    const std::string path = "src/comm/tilexr_internal.cpp";
    const auto text = ReadFile(path);

    CheckContains(path, text, "ChipName ResolveChipNameFromSocVersion");
    CheckContains(path, text, "{\"Ascend950PR\", ChipName::CHIP_950PR}");
    CheckContains(path, text, "{\"Ascend950PR_\", ChipName::CHIP_950PR}");
    CheckContains(path, text, "{\"Ascend950DT_\", ChipName::CHIP_950}");
    CheckContains(path, text, "ResolveChipNameFromSocVersion(chipName)");
}

void TestCommRuntimeDoesNotUseHcommOrHcclV2()
{
    const std::string path = "src/comm/CMakeLists.txt";
    const auto text = ReadFile(path);

    CheckNotContains(path, text, "hcomm");
    CheckNotContains(path, text, "hccl");
    CheckNotContains(path, text, "libhccl_v2");
    CheckNotContains(path, text, "libhccl_fwk");
    CheckNotContains(path, text, "libmc2_client");

    CheckTrackedFilesDoNotContain("hcomm or HCCL CCU runtime dependency", "src/comm src/include", {
        "#include <hcomm/",
        "#include \"hcomm/",
        "#include <hccl/",
        "#include \"hccl/",
        "#include <hccl.h>",
        "#include \"hccl.h\"",
        "pkg_inc/hcomm",
        "pkg_inc/hccl",
        "include/hccl",
        "${ARCH}-linux/include/hccl",
        "${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl",
        "libhcomm",
        "libhccl_v2",
        "libhccl_fwk",
        "libmc2_client",
        "HcclAllocComResourceByTiling",
        "HcclCreateOpResCtx",
        "HcclEngineCtx",
        "HcclGetCcuTaskInfo",
        "HcomGetCcuTaskInfo",
        "HcclChannelAcquire",
        "HcclGetChannelForCcu",
        "HcclAllocAlgResourceCcu",
        "HcclCcuKernel",
        "HcommChannelNotify",
        "HcommChannelFence",
        "rtGetNotifyAddress",
        "HrtCcuLaunch",
        "HrtGetDevResAddress",
        "HrtReleaseDevResAddress",
        "HrtNotifyGetAddr",
        "HrtRaCustomChannel",
        "HrtCntNotify",
        "CcuResBatchAllocator",
        "CcuResRepository",
        "CcuResReq",
        "CcuDeviceManager",
        "CcuDevMgrImp",
        "CcuRepContext",
        "CcuKernelMgr",
        "CtxMgrImp",
        "CcuTaskParam",
        "CcuTaskArg",
        "GeneTaskParam",
        "GetMissionKey",
        "SetMissionId",
        "SetMissionKey",
        "SetInstrId",
        "SetCcuInstrInfo",
        "LoadInstruction",
        "AllocIns",
        "AllocCke",
        "AllocXn",
        "COMM_ENGINE_CCU",
        "COMM_PROTOCOL_UBC_CTP",
        "RT_RES_TYPE_CCU_CKE",
        "RT_RES_TYPE_CCU_XN",
        "HCCL_SERVER_TYPE_CCU",
    });
}

void TestRootCMakeHcclIncludesAreNotTileCommSurface()
{
    const std::string rootPath = "CMakeLists.txt";
    const std::string commPath = "src/comm/CMakeLists.txt";
    const auto rootText = ReadFile(rootPath);
    const auto commText = ReadFile(commPath);

    for (const auto& privateInclude : {
        "${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl",
        "${ARCH}-linux/include/hccl",
        "pkg_inc/hccl",
        "include/hccl",
        "hccl/",
    }) {
        CheckNotContains(rootPath, rootText, privateInclude);
        CheckNotContains(commPath, commText, privateInclude);
    }
}

void TestCommDirectCcuInstallAttemptDoesNotSubmit()
{
    const std::string commHeaderPath = "src/comm/tilexr_comm.h";
    const std::string commSourcePath = "src/comm/tilexr_comm.cpp";
    const std::string ccuRuntimeHeaderPath = "src/comm/ccu/tilexr_ccu_direct_runtime.h";
    const std::string ccuRuntimeSourcePath = "src/comm/ccu/tilexr_ccu_direct_runtime.cpp";
    const auto commHeaderText = ReadFile(commHeaderPath);
    const auto commSourceText = ReadFile(commSourcePath);
    const auto ccuRuntimeHeaderText = ReadFile(ccuRuntimeHeaderPath);
    const auto ccuRuntimeSourceText = ReadFile(ccuRuntimeSourcePath);

    CheckContains(commHeaderPath, commHeaderText, "PrepareDirectCcuInstallAttempt");
    CheckContains(commHeaderPath, commHeaderText, "FillDirectCcuLowerLayerPlanFromAllocation");
    CheckContains(commHeaderPath, commHeaderText, "PrepareDirectCcuLowerLayerPlanCallback");
    CheckContains(commHeaderPath, commHeaderText, "std::unique_ptr<TileXRCcuDirectRuntime> ccuDirectRuntime_");
    CheckContains(commSourcePath, commSourceText, "#include \"ccu/tilexr_ccu_repository.h\"");
    CheckContains(commSourcePath, commSourceText, "int TileXRComm::InitDirectCcuRuntime");
    CheckContains(commSourcePath, commSourceText, "int TileXRComm::PrepareDirectCcuInstallAttempt");
    CheckContains(commSourcePath, commSourceText, "ccuDirectRuntime_->CreateDriverAdapter");
    CheckContains(commSourcePath, commSourceText, "TileXRCcuMakeAclDeviceMemoryOps()");
    CheckContains(commSourcePath, commSourceText, "next.lowerLayerPlan = nullptr");
    CheckContains(
        commSourcePath,
        commSourceText,
        "next.prepareLowerLayerPlan = &TileXRComm::PrepareDirectCcuLowerLayerPlanCallback");
    CheckContains(commSourcePath, commSourceText, "next.lowerLayerPlanUserData = this");
    CheckContains(commSourcePath, commSourceText, "TileXRCcuRunDirectInstallAttempt(next, attempt, report)");
    CheckContains(ccuRuntimeHeaderPath, ccuRuntimeHeaderText, "int CreateDriverAdapter(");
    CheckContains(ccuRuntimeSourcePath, ccuRuntimeSourceText, "int TileXRCcuDirectRuntime::CreateDriverAdapter");
    CheckContains(ccuRuntimeHeaderPath, ccuRuntimeHeaderText, "TileXRCcuHccpLoader");
    for (const auto& forbiddenUdmaCcuCall : {
        std::string("udmaTransport_->") + "CreateCcuDriverAdapter",
        std::string("udmaTransport_->") + "QueryCcuBasicInfo",
        std::string("udmaTransport_->") + "RegisterCcuResourceRmaBuffer",
        std::string("udmaTransport_->") + "ExportLocalCcuRmaBuffer",
        std::string("udmaTransport_->") + "ExportRemoteCcuRmaBuffers",
        std::string("udmaTransport_->") + "ExportLowerLayerTransportSnapshot",
    }) {
        CheckNotContains(commSourcePath, commSourceText, forbiddenUdmaCcuCall);
    }

    const auto initUdmaBegin = commSourceText.find("int TileXRComm::InitUDMA");
    const auto initDirectCcuBegin = commSourceText.find("int TileXRComm::InitDirectCcuRuntime");
    if (initUdmaBegin == std::string::npos || initDirectCcuBegin == std::string::npos ||
        initUdmaBegin >= initDirectCcuBegin) {
        std::cerr << commSourcePath << ": cannot isolate InitUDMA body" << std::endl;
        ++g_failures;
    } else {
        const auto initUdmaBody = commSourceText.substr(initUdmaBegin, initDirectCcuBegin - initUdmaBegin);
        CheckNotContains(commSourcePath, initUdmaBody, "RefreshDirectCcuBasicInfo");
        CheckNotContains(commSourcePath, initUdmaBody, "ResetDirectCcuBasicInfo");
    }

    const auto registerUdmaBegin = commSourceText.find("int TileXRComm::RegisterUDMAMemory");
    const auto unregisterUdmaBegin = commSourceText.find("int TileXRComm::UnregisterUDMAMemory");
    const auto getUdmaRegistryBegin = commSourceText.find("GM_ADDR TileXRComm::GetUDMARegistryPtr");
    if (registerUdmaBegin == std::string::npos || unregisterUdmaBegin == std::string::npos ||
        getUdmaRegistryBegin == std::string::npos || registerUdmaBegin >= unregisterUdmaBegin ||
        unregisterUdmaBegin >= getUdmaRegistryBegin) {
        std::cerr << commSourcePath << ": cannot isolate UDMA memory registration bodies" << std::endl;
        ++g_failures;
    } else {
        const auto registerUdmaBody = commSourceText.substr(
            registerUdmaBegin, unregisterUdmaBegin - registerUdmaBegin);
        const auto unregisterUdmaBody = commSourceText.substr(
            unregisterUdmaBegin, getUdmaRegistryBegin - unregisterUdmaBegin);
        CheckNotContains(commSourcePath, registerUdmaBody, "ResetDirectCcuLowerLayerPlan");
        CheckNotContains(commSourcePath, unregisterUdmaBody, "ResetDirectCcuLowerLayerPlan");
    }

    for (const auto& forbidden : {
        "TileXRCcuPrepareSubmitTasks",
        "TileXRCcuSubmitTask",
        "rtCCULaunch",
        "HcclGetCcuTaskInfo",
        "HcomGetCcuTaskInfo",
        "libhcomm",
        "libhccl_v2",
    }) {
        CheckNotContains(commHeaderPath, commHeaderText, forbidden);
        CheckNotContains(commSourcePath, commSourceText, forbidden);
        CheckNotContains(ccuRuntimeHeaderPath, ccuRuntimeHeaderText, forbidden);
        CheckNotContains(ccuRuntimeSourcePath, ccuRuntimeSourceText, forbidden);
    }
}

void TestCcuRuntimeSubmitBoundaryUsesPublicRuntimeOnly()
{
    const std::string cmakePath = "src/comm/CMakeLists.txt";
    const std::string abiConstantsHeaderPath = "src/comm/ccu/tilexr_ccu_abi_constants.h";
    const std::string barrierHeaderPath = "src/comm/ccu/tilexr_ccu_barrier_program.h";
    const std::string barrierSourcePath = "src/comm/ccu/tilexr_ccu_barrier_program.cpp";
    const std::string directHeaderPath = "src/comm/ccu/tilexr_ccu_direct_orchestrator.h";
    const std::string directSourcePath = "src/comm/ccu/tilexr_ccu_direct_orchestrator.cpp";
    const std::string directRuntimeHeaderPath = "src/comm/ccu/tilexr_ccu_direct_runtime.h";
    const std::string directRuntimeSourcePath = "src/comm/ccu/tilexr_ccu_direct_runtime.cpp";
    const std::string driverHeaderPath = "src/comm/ccu/tilexr_ccu_driver_adapter.h";
    const std::string driverSourcePath = "src/comm/ccu/tilexr_ccu_driver_adapter.cpp";
    const std::string hccpTypesHeaderPath = "src/comm/ccu/tilexr_ccu_hccp_types.h";
    const std::string hccpLoaderHeaderPath = "src/comm/ccu/tilexr_ccu_hccp_loader.h";
    const std::string hccpLoaderSourcePath = "src/comm/ccu/tilexr_ccu_hccp_loader.cpp";
    const std::string installHeaderPath = "src/comm/ccu/tilexr_ccu_install_provider.h";
    const std::string installSourcePath = "src/comm/ccu/tilexr_ccu_install_provider.cpp";
    const std::string lowerLayerPlanHeaderPath = "src/comm/ccu/tilexr_ccu_lower_layer_plan_builder.h";
    const std::string lowerLayerPlanSourcePath = "src/comm/ccu/tilexr_ccu_lower_layer_plan_builder.cpp";
    const std::string packageHeaderPath = "src/comm/ccu/tilexr_ccu_launch_package.h";
    const std::string packageSourcePath = "src/comm/ccu/tilexr_ccu_launch_package.cpp";
    const std::string microcodeHeaderPath = "src/comm/ccu/tilexr_ccu_microcode.h";
    const std::string microcodeSourcePath = "src/comm/ccu/tilexr_ccu_microcode.cpp";
    const std::string planHeaderPath = "src/comm/ccu/tilexr_ccu_producer_plan.h";
    const std::string planSourcePath = "src/comm/ccu/tilexr_ccu_producer_plan.cpp";
    const std::string providerHeaderPath = "src/comm/ccu/tilexr_ccu_provider.h";
    const std::string providerSourcePath = "src/comm/ccu/tilexr_ccu_provider.cpp";
    const std::string raProviderHeaderPath = "src/comm/ccu/tilexr_ccu_ra_custom_channel_provider.h";
    const std::string raProviderSourcePath = "src/comm/ccu/tilexr_ccu_ra_custom_channel_provider.cpp";
    const std::string repositoryHeaderPath = "src/comm/ccu/tilexr_ccu_repository.h";
    const std::string repositorySourcePath = "src/comm/ccu/tilexr_ccu_repository.cpp";
    const std::string allocatorHeaderPath = "src/comm/ccu/tilexr_ccu_resource_allocator.h";
    const std::string allocatorSourcePath = "src/comm/ccu/tilexr_ccu_resource_allocator.cpp";
    const std::string specsHeaderPath = "src/comm/ccu/tilexr_ccu_specs.h";
    const std::string specsSourcePath = "src/comm/ccu/tilexr_ccu_specs.cpp";
    const std::string headerPath = "src/comm/ccu/tilexr_ccu_runtime.h";
    const std::string sourcePath = "src/comm/ccu/tilexr_ccu_runtime.cpp";
    const auto cmakeText = ReadFile(cmakePath);
    const auto abiConstantsHeaderText = ReadFile(abiConstantsHeaderPath);
    const auto barrierHeaderText = ReadFile(barrierHeaderPath);
    const auto barrierSourceText = ReadFile(barrierSourcePath);
    const auto directHeaderText = ReadFile(directHeaderPath);
    const auto directSourceText = ReadFile(directSourcePath);
    const auto directRuntimeHeaderText = ReadFile(directRuntimeHeaderPath);
    const auto directRuntimeSourceText = ReadFile(directRuntimeSourcePath);
    const auto driverHeaderText = ReadFile(driverHeaderPath);
    const auto driverSourceText = ReadFile(driverSourcePath);
    const auto hccpTypesHeaderText = ReadFile(hccpTypesHeaderPath);
    const auto hccpLoaderHeaderText = ReadFile(hccpLoaderHeaderPath);
    const auto hccpLoaderSourceText = ReadFile(hccpLoaderSourcePath);
    const auto installHeaderText = ReadFile(installHeaderPath);
    const auto installSourceText = ReadFile(installSourcePath);
    const auto lowerLayerPlanHeaderText = ReadFile(lowerLayerPlanHeaderPath);
    const auto lowerLayerPlanSourceText = ReadFile(lowerLayerPlanSourcePath);
    const auto packageHeaderText = ReadFile(packageHeaderPath);
    const auto packageSourceText = ReadFile(packageSourcePath);
    const auto microcodeHeaderText = ReadFile(microcodeHeaderPath);
    const auto microcodeSourceText = ReadFile(microcodeSourcePath);
    const auto planHeaderText = ReadFile(planHeaderPath);
    const auto planSourceText = ReadFile(planSourcePath);
    const auto providerHeaderText = ReadFile(providerHeaderPath);
    const auto providerSourceText = ReadFile(providerSourcePath);
    const auto raProviderHeaderText = ReadFile(raProviderHeaderPath);
    const auto raProviderSourceText = ReadFile(raProviderSourcePath);
    const auto repositoryHeaderText = ReadFile(repositoryHeaderPath);
    const auto repositorySourceText = ReadFile(repositorySourcePath);
    const auto allocatorHeaderText = ReadFile(allocatorHeaderPath);
    const auto allocatorSourceText = ReadFile(allocatorSourcePath);
    const auto specsHeaderText = ReadFile(specsHeaderPath);
    const auto specsSourceText = ReadFile(specsSourcePath);
    const auto headerText = ReadFile(headerPath);
    const auto sourceText = ReadFile(sourcePath);

    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_abi_constants.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_barrier_program.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_barrier_program.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_driver_adapter.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_direct_orchestrator.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_direct_orchestrator.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_direct_runtime.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_direct_runtime.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_driver_adapter.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_hccp_types.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_hccp_loader.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_hccp_loader.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_launch_package.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_launch_package.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_install_provider.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_install_provider.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_lower_layer_plan_builder.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_lower_layer_plan_builder.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_microcode.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_microcode.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_producer_plan.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_producer_plan.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_provider.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_provider.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_ra_custom_channel_provider.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_ra_custom_channel_provider.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_repository.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_repository.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_resource_allocator.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_resource_allocator.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_runtime.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_runtime.cpp");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_specs.h");
    CheckContains(cmakePath, cmakeText, "ccu/tilexr_ccu_specs.cpp");

    CheckContains(abiConstantsHeaderPath, abiConstantsHeaderText, "TILEXR_CCU_EID_BYTES");
    CheckContains(abiConstantsHeaderPath, abiConstantsHeaderText, "TILEXR_CCU_REMOTE_CCU_VA_SHIFT");
    CheckNotContains(abiConstantsHeaderPath, abiConstantsHeaderText, "runtime/kernel.h");
    CheckNotContains(abiConstantsHeaderPath, abiConstantsHeaderText, "rtCcuTaskInfo_t");

    CheckContains(directHeaderPath, directHeaderText, "TileXRCcuDirectInstallOptions");
    CheckContains(directHeaderPath, directHeaderText, "TileXRCcuDirectInstallAttempt");
    CheckContains(directHeaderPath, directHeaderText, "TileXRCcuDirectInstallReport");
    CheckContains(directHeaderPath, directHeaderText, "TileXRCcuRunDirectInstallAttempt");
    CheckContains(directSourcePath, directSourceText, "TileXRCcuDecodeBasicInfo");
    CheckContains(directSourcePath, directSourceText, "TileXRCcuBuildResourceSpec");
    CheckContains(directSourcePath, directSourceText, "TileXRCcuResourceAllocator");
    CheckContains(directSourcePath, directSourceText, "TileXRCcuBuildLaunchPackage");
    CheckContains(directSourcePath, directSourceText, "TileXRCcuBindLaunchPackageInstallScope");
    CheckContains(directSourcePath, directSourceText, "TileXRCcuBuildInstallManifest");
    CheckContains(directSourcePath, directSourceText, "TileXRCcuInstallHardware");
    CheckContains(directSourcePath, directSourceText, "TileXRCcuPrepareSubmitTasks");
    CheckNotContains(directHeaderPath, directHeaderText, "runtime/kernel.h");
    CheckNotContains(directHeaderPath, directHeaderText, "rtCcuTaskInfo_t");

    CheckContains(directRuntimeHeaderPath, directRuntimeHeaderText, "TileXRCcuDirectRuntime");
    CheckContains(directRuntimeHeaderPath, directRuntimeHeaderText, "QueryBasicInfo");
    CheckContains(directRuntimeHeaderPath, directRuntimeHeaderText, "CreateDriverAdapter");
    CheckContains(directRuntimeHeaderPath, directRuntimeHeaderText, "RegisterCcuResourceRmaBuffer");
    CheckContains(directRuntimeSourcePath, directRuntimeSourceText, "TileXRCcuRaCustomChannelProvider");
    CheckContains(directRuntimeSourcePath, directRuntimeSourceText, "loader_.Load");
    CheckContains(directRuntimeSourcePath, directRuntimeSourceText, "loader_.ResolveDevicePhyId");
    CheckContains(directRuntimeSourcePath, directRuntimeSourceText, "loader_.RaCustomChannel");
    CheckNotContains(directRuntimeHeaderPath, directRuntimeHeaderText, "udma/");
    CheckNotContains(directRuntimeSourcePath, directRuntimeSourceText, "udma/");

    CheckContains(driverHeaderPath, driverHeaderText, "TileXRCcuDriverAdapter");
    CheckContains(driverHeaderPath, driverHeaderText, "TileXRCcuCustomChannelIn");
    CheckContains(driverHeaderPath, driverHeaderText, "TILEXR_CCU_U_OP_GET_BASIC_INFO");
    CheckContains(driverHeaderPath, driverHeaderText, "TILEXR_CCU_U_OP_GET_DIE_WORKING");
    CheckContains(driverHeaderPath, driverHeaderText, "GetBasicInfo");
    CheckContains(driverHeaderPath, driverHeaderText, "GetDieEnabled");
    CheckNotContains(driverHeaderPath, driverHeaderText, "runtime/kernel.h");
    CheckNotContains(driverHeaderPath, driverHeaderText, "rtCcuTaskInfo_t");

    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuInstallRequest");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuInstallManifest");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuInstallManifestReport");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuInstallRequirement");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuInstallRequirementKind");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuBuildInstallManifest");
    CheckContains(installHeaderPath, installHeaderText, "provider");
    CheckContains(installHeaderPath, installHeaderText, "manifest");
    CheckContains(installHeaderPath, installHeaderText, "installAttemptReceiptRequired");
    CheckContains(installHeaderPath, installHeaderText, "requiredEvidenceKind");
    CheckContains(installHeaderPath, installHeaderText, "requiredEvidenceSurface");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuInstallStepEvidence");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuInstallProviderReport");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuBuildInstallEvidence");
    CheckContains(installHeaderPath, installHeaderText, "TileXRCcuInstallHardware");
    CheckContains(installHeaderPath, installHeaderText, "installAttemptReceiptId");
    CheckContains(installSourcePath, installSourceText, "ValidateInstallRequestScope");
    CheckContains(installSourcePath, installSourceText, "ValidateInstallManifestScope");
    CheckContains(installSourcePath, installSourceText, "ValidateInstallRequestManifest");
    CheckContains(installSourcePath, installSourceText, "ValidatePublicVerifiedStepScope");
    CheckContains(installHeaderPath, installHeaderText, "requiredInstallSurfaceCount");
    CheckContains(installHeaderPath, installHeaderText, "publicVerifiedInstallSurfaceCount");
    CheckContains(installHeaderPath, installHeaderText, "missingInstallSurfaceCount");
    CheckContains(installSourcePath, installSourceText, "FillManifestInstallSurfaceCounts");
    CheckContains(installSourcePath, installSourceText, "launch install scope is stale");
    CheckContains(installSourcePath, installSourceText, "install manifest requirement kind mismatch");
    CheckContains(installSourcePath, installSourceText, "install manifest hardware requirement mismatch");
    CheckContains(installSourcePath, installSourceText, "install manifest mission requirement mismatch");
    CheckContains(installSourcePath, installSourceText, "install manifest channel requirement mismatch");
    CheckContains(installSourcePath, installSourceText, "public verified evidence scope is incomplete");
    CheckContains(installSourcePath, installSourceText, "missing CCU repository image for install manifest");
    CheckContains(installSourcePath, installSourceText, "install generated CCU repository image");
    CheckContains(installSourcePath, installSourceText, "bind CCU channel routes for sync resources");
    CheckContains(installHeaderPath, installHeaderText, "localWaitCke");
    CheckContains(installSourcePath, installSourceText, "local wait CKE");
    CheckContains(installSourcePath, installSourceText, "remote notify CKE");
    CheckContains(installSourcePath, installSourceText, "ValidateInstallReceipt");
    CheckContains(installSourcePath, installSourceText, "install attempt did not succeed");
    CheckContains(installSourcePath, installSourceText, "install attempt receipt mismatch");
    CheckContains(installSourcePath, installSourceText, "device scope mismatch");
    CheckContains(installSourcePath, installSourceText, "rank scope mismatch");
    CheckContains(installSourcePath, installSourceText, "provider scope mismatch");
    CheckContains(installHeaderPath, installHeaderText, "tilexr_ccu_provider.h");
    CheckNotContains(installHeaderPath, installHeaderText, "runtime/kernel.h");
    CheckNotContains(installHeaderPath, installHeaderText, "rtCcuTaskInfo_t");

    CheckContains(lowerLayerPlanHeaderPath, lowerLayerPlanHeaderText, "TileXRCcuLowerLayerPlanSpec");
    CheckContains(lowerLayerPlanHeaderPath, lowerLayerPlanHeaderText, "TileXRCcuBuildLowerLayerInstallPlan");
    CheckContains(lowerLayerPlanHeaderPath, lowerLayerPlanHeaderText, "remoteNotifyCke");
    CheckContains(lowerLayerPlanSourcePath, lowerLayerPlanSourceText, "TileXRCcuBuildPfeCtx");
    CheckContains(lowerLayerPlanSourcePath, lowerLayerPlanSourceText, "TileXRCcuBuildLocalJettyCtx");
    CheckContains(lowerLayerPlanSourcePath, lowerLayerPlanSourceText, "TileXRCcuBuildChannelCtxV1");
    CheckContains(lowerLayerPlanSourcePath, lowerLayerPlanSourceText, "localWaitCke");
    CheckNotContains(lowerLayerPlanHeaderPath, lowerLayerPlanHeaderText, "runtime/kernel.h");
    CheckNotContains(lowerLayerPlanHeaderPath, lowerLayerPlanHeaderText, "rtCcuTaskInfo_t");

    CheckContains(packageHeaderPath, packageHeaderText, "TileXRCcuLaunchPackage");
    CheckContains(packageHeaderPath, packageHeaderText, "TileXRCcuLaunchPackageReport");
    CheckContains(packageHeaderPath, packageHeaderText, "TileXRCcuBuildLaunchPackage");
    CheckContains(packageHeaderPath, packageHeaderText, "TileXRCcuComputeLaunchPackageFingerprint");
    CheckContains(packageSourcePath, packageSourceText, "TileXRCcuComputeLaunchPackageFingerprint");
    CheckContains(packageHeaderPath, packageHeaderText, "TileXRCcuLaunchInstallScope");
    CheckContains(packageHeaderPath, packageHeaderText, "installScope");
    CheckContains(packageHeaderPath, packageHeaderText, "TileXRCcuBindLaunchPackageInstallScope");
    CheckContains(packageSourcePath, packageSourceText, "TileXRCcuBindLaunchPackageInstallScope");
    CheckContains(packageHeaderPath, packageHeaderText, "requiresHardwareInstall");
    CheckContains(packageHeaderPath, packageHeaderText, "tilexr_ccu_repository.h");
    CheckNotContains(packageHeaderPath, packageHeaderText, "runtime/kernel.h");
    CheckNotContains(packageHeaderPath, packageHeaderText, "rtCcuTaskInfo_t");

    CheckContains(microcodeHeaderPath, microcodeHeaderText, "struct TileXRCcuInstr");
    CheckContains(microcodeHeaderPath, microcodeHeaderText, "struct TileXRCcuSyncXnSpec");
    CheckContains(microcodeHeaderPath, microcodeHeaderText, "struct TileXRCcuCkeSpec");
    CheckContains(microcodeHeaderPath, microcodeHeaderText, "TileXRCcuEncodeLoadSqeArgsToX");
    CheckContains(microcodeHeaderPath, microcodeHeaderText, "TileXRCcuEncodeSyncXn");
    CheckContains(microcodeHeaderPath, microcodeHeaderText, "TileXRCcuEncodeSetCke");
    CheckContains(microcodeHeaderPath, microcodeHeaderText, "TileXRCcuEncodeClearCke");
    CheckContains(microcodeSourcePath, microcodeSourceText, "0x0001U");
    CheckContains(microcodeSourcePath, microcodeSourceText, "0x0802U");
    CheckContains(microcodeSourcePath, microcodeSourceText, "0x0804U");
    CheckContains(microcodeSourcePath, microcodeSourceText, "0x100dU");
    CheckNotContains(microcodeHeaderPath, microcodeHeaderText, "runtime/kernel.h");
    CheckNotContains(microcodeHeaderPath, microcodeHeaderText, "rtCcuTaskInfo_t");

    CheckContains(barrierHeaderPath, barrierHeaderText, "TileXRCcuBarrierSyncSpec");
    CheckContains(barrierHeaderPath, barrierHeaderText, "TileXRCcuBarrierProgramReport");
    CheckContains(barrierHeaderPath, barrierHeaderText, "TileXRCcuBuildBarrierProgram");
    CheckContains(barrierSourcePath, barrierSourceText, "TileXRCcuEncodeSyncXn");
    CheckContains(barrierSourcePath, barrierSourceText, "TileXRCcuEncodeClearCke");
    CheckNotContains(barrierHeaderPath, barrierHeaderText, "runtime/kernel.h");
    CheckNotContains(barrierHeaderPath, barrierHeaderText, "rtCcuTaskInfo_t");

    CheckContains(planHeaderPath, planHeaderText, "TileXRCcuProducerPlan");
    CheckContains(planHeaderPath, planHeaderText, "TileXRCcuValidateProducerPlan");
    CheckContains(planHeaderPath, planHeaderText, "TileXRCcuBuildTasks");
    CheckContains(planHeaderPath, planHeaderText, "TileXRCcuBuildMicrocode");
    CheckContains(planHeaderPath, planHeaderText, "TileXRCcuSyncResource");
    CheckContains(planHeaderPath, planHeaderText, "localWaitCke");
    CheckContains(planHeaderPath, planHeaderText, "localWaitMask");
    CheckContains(planHeaderPath, planHeaderText, "remoteNotifyMask");
    CheckContains(planHeaderPath, planHeaderText, "TileXRCcuInstructionWindow");
    CheckContains(planHeaderPath, planHeaderText, "TileXRCcuProgram");
    CheckContains(planHeaderPath, planHeaderText, "tilexr_ccu_barrier_program.h");
    CheckContains(planSourcePath, planSourceText, "TileXRCcuBuildBarrierProgram");
    CheckContains(planSourcePath, planSourceText, "spec.localWaitCke");
    CheckNotContains(planHeaderPath, planHeaderText, "runtime/kernel.h");
    CheckNotContains(planHeaderPath, planHeaderText, "rtCcuTaskInfo_t");

    CheckContains(providerHeaderPath, providerHeaderText, "TileXRCcuHardwareInstallEvidence");
    CheckContains(providerHeaderPath, providerHeaderText, "TileXRCcuValidateHardwareInstall");
    CheckContains(providerHeaderPath, providerHeaderText, "TileXRCcuPrepareSubmitTasks");
    CheckContains(providerHeaderPath, providerHeaderText, "submitReady");
    CheckContains(providerHeaderPath, providerHeaderText, "packageFingerprint");
    CheckContains(providerHeaderPath, providerHeaderText, "deviceId");
    CheckContains(providerHeaderPath, providerHeaderText, "rank");
    CheckContains(providerHeaderPath, providerHeaderText, "provider");
    CheckContains(providerHeaderPath, providerHeaderText, "installAttemptReceiptId");
    CheckContains(providerSourcePath, providerSourceText, "package fingerprint mismatch");
    CheckContains(providerSourcePath, providerSourceText, "launch install scope is not bound");
    CheckContains(providerSourcePath, providerSourceText, "device scope mismatch");
    CheckContains(providerSourcePath, providerSourceText, "rank scope mismatch");
    CheckContains(providerSourcePath, providerSourceText, "provider scope mismatch");
    CheckContains(providerSourcePath, providerSourceText, "install attempt receipt is missing");
    CheckContains(providerSourcePath, providerSourceText, "install attempt receipt mismatch");
    CheckContains(providerHeaderPath, providerHeaderText, "tilexr_ccu_launch_package.h");
    CheckNotContains(providerHeaderPath, providerHeaderText, "runtime/kernel.h");
    CheckNotContains(providerHeaderPath, providerHeaderText, "rtCcuTaskInfo_t");

    CheckContains(raProviderHeaderPath, raProviderHeaderText, "TileXRCcuRaCustomChannelProvider");
    CheckContains(raProviderHeaderPath, raProviderHeaderText, "CreateAdapter");
    CheckContains(raProviderHeaderPath, raProviderHeaderText, "TileXRCcuRaCustomChannelFunc");
    CheckContains(raProviderSourcePath, raProviderSourceText, "TILEXR_CCU_NETWORK_OFFLINE");
    CheckNotContains(raProviderHeaderPath, raProviderHeaderText, "udma/");
    CheckNotContains(raProviderHeaderPath, raProviderHeaderText, "runtime/kernel.h");
    CheckNotContains(raProviderHeaderPath, raProviderHeaderText, "rtCcuTaskInfo_t");

    CheckContains(hccpTypesHeaderPath, hccpTypesHeaderText, "TileXRCcuRaInfo");
    CheckContains(hccpTypesHeaderPath, hccpTypesHeaderText, "TileXRCcuRaCustomChannelFunc");
    CheckContains(hccpLoaderHeaderPath, hccpLoaderHeaderText, "TileXRCcuHccpLoader");
    CheckContains(hccpLoaderSourcePath, hccpLoaderSourceText, "dlopen(\"libra.so\", RTLD_NOW)");
    CheckContains(hccpLoaderSourcePath, hccpLoaderSourceText, "RaCustomChannel");
    CheckContains(hccpLoaderSourcePath, hccpLoaderSourceText, "rtGetDevicePhyIdByIndex");

    CheckContains(repositoryHeaderPath, repositoryHeaderText, "TileXRCcuRepositoryImage");
    CheckContains(repositoryHeaderPath, repositoryHeaderText, "TileXRCcuRepositoryReport");
    CheckContains(repositoryHeaderPath, repositoryHeaderText, "TileXRCcuBuildRepositoryImage");
    CheckContains(repositoryHeaderPath, repositoryHeaderText, "missionOffset");
    CheckContains(repositoryHeaderPath, repositoryHeaderText, "sqeLoadOffset");
    CheckContains(repositoryHeaderPath, repositoryHeaderText, "syncOffset");
    CheckNotContains(repositoryHeaderPath, repositoryHeaderText, "runtime/kernel.h");
    CheckNotContains(repositoryHeaderPath, repositoryHeaderText, "rtCcuTaskInfo_t");

    CheckContains(allocatorHeaderPath, allocatorHeaderText, "TileXRCcuResourceAllocator");
    CheckContains(allocatorHeaderPath, allocatorHeaderText, "TileXRCcuResourceSpec");
    CheckContains(allocatorHeaderPath, allocatorHeaderText, "TileXRCcuResourceRequest");
    CheckContains(allocatorSourcePath, allocatorSourceText, "TileXRCcuValidateProducerPlan");
    CheckNotContains(allocatorHeaderPath, allocatorHeaderText, "runtime/kernel.h");
    CheckNotContains(allocatorHeaderPath, allocatorHeaderText, "rtCcuTaskInfo_t");

    CheckContains(specsHeaderPath, specsHeaderText, "TileXRCcuBasicInfo");
    CheckContains(specsHeaderPath, specsHeaderText, "TileXRCcuDecodeBasicInfo");
    CheckContains(specsHeaderPath, specsHeaderText, "TileXRCcuBuildResourceSpec");
    CheckContains(specsSourcePath, specsSourceText, "TILEXR_CCU_V1_XN_RESOURCE_OFFSET");
    CheckNotContains(specsHeaderPath, specsHeaderText, "runtime/kernel.h");
    CheckNotContains(specsHeaderPath, specsHeaderText, "rtCcuTaskInfo_t");

    CheckContains(headerPath, headerText, "TILEXR_CCU_SQE_ARGS_LEN");
    CheckContains(headerPath, headerText, "struct TileXRCcuTask");
    CheckContains(headerPath, headerText, "TileXRCcuValidateTask");
    CheckContains(headerPath, headerText, "TileXRCcuSubmitTask");
    CheckNotContains(headerPath, headerText, "runtime/kernel.h");
    CheckNotContains(headerPath, headerText, "rtCcuTaskInfo_t");

    CheckContains(sourcePath, sourceText, "#include <runtime/kernel.h>");
    CheckContains(sourcePath, sourceText, "rtCcuTaskInfo_t runtimeTask");
    CheckContains(sourcePath, sourceText, "rtCCULaunch(&runtimeTask, stream)");
    CheckContains(sourcePath, sourceText, "RT_CCU_INST_CNT_INVALID");
    CheckContains(sourcePath, sourceText, "RT_CCU_INST_START_MAX");
    CheckContains(sourcePath, sourceText, "task.argSize != 1 && task.argSize != TILEXR_CCU_SQE_ARGS_LEN");
    CheckContains(sourcePath, sourceText, "TILEXR_ERROR_MKIRT");

    for (const auto& privateNeedle : {
        "#include <hcomm/",
        "#include <hccl/",
        "#include \"hcomm/",
        "#include \"hccl/",
        "#include <hccl.h>",
        "#include \"hccl.h\"",
        "pkg_inc/hcomm",
        "pkg_inc/hccl",
        "include/hccl",
        "${ARCH}-linux/include/hccl",
        "${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl",
        "libhcomm",
        "libhccl_v2",
        "libhccl_fwk",
        "libmc2_client",
        "HcclCcuKernel",
        "HcclGetCcuTaskInfo",
        "HcomGetCcuTaskInfo",
        "HcclChannelAcquire",
        "HcclGetChannelForCcu",
        "CcuResBatchAllocator",
        "CcuResRepository",
        "GetMissionKey",
        "SetMissionId",
        "SetMissionKey",
        "SetInstrId",
        "SetCcuInstrInfo",
        "LoadInstruction",
        "AllocIns",
        "AllocXn",
        "AllocCke",
        "RT_RES_TYPE_CCU_CKE",
        "RT_RES_TYPE_CCU_XN",
        "dlopen",
        "dlsym",
    }) {
        CheckNotContains(directSourcePath, directSourceText, privateNeedle);
        CheckNotContains(abiConstantsHeaderPath, abiConstantsHeaderText, privateNeedle);
        CheckNotContains(directHeaderPath, directHeaderText, privateNeedle);
        CheckNotContains(directRuntimeSourcePath, directRuntimeSourceText, privateNeedle);
        CheckNotContains(directRuntimeHeaderPath, directRuntimeHeaderText, privateNeedle);
        CheckNotContains(driverSourcePath, driverSourceText, privateNeedle);
        CheckNotContains(driverHeaderPath, driverHeaderText, privateNeedle);
        CheckNotContains(barrierSourcePath, barrierSourceText, privateNeedle);
        CheckNotContains(barrierHeaderPath, barrierHeaderText, privateNeedle);
        CheckNotContains(installSourcePath, installSourceText, privateNeedle);
        CheckNotContains(installHeaderPath, installHeaderText, privateNeedle);
        CheckNotContains(packageSourcePath, packageSourceText, privateNeedle);
        CheckNotContains(packageHeaderPath, packageHeaderText, privateNeedle);
        CheckNotContains(microcodeSourcePath, microcodeSourceText, privateNeedle);
        CheckNotContains(microcodeHeaderPath, microcodeHeaderText, privateNeedle);
        CheckNotContains(planSourcePath, planSourceText, privateNeedle);
        CheckNotContains(planHeaderPath, planHeaderText, privateNeedle);
        CheckNotContains(providerSourcePath, providerSourceText, privateNeedle);
        CheckNotContains(providerHeaderPath, providerHeaderText, privateNeedle);
        CheckNotContains(raProviderSourcePath, raProviderSourceText, privateNeedle);
        CheckNotContains(raProviderHeaderPath, raProviderHeaderText, privateNeedle);
        CheckNotContains(repositorySourcePath, repositorySourceText, privateNeedle);
        CheckNotContains(repositoryHeaderPath, repositoryHeaderText, privateNeedle);
        CheckNotContains(allocatorSourcePath, allocatorSourceText, privateNeedle);
        CheckNotContains(allocatorHeaderPath, allocatorHeaderText, privateNeedle);
        CheckNotContains(specsSourcePath, specsSourceText, privateNeedle);
        CheckNotContains(specsHeaderPath, specsHeaderText, privateNeedle);
        CheckNotContains(sourcePath, sourceText, privateNeedle);
        CheckNotContains(headerPath, headerText, privateNeedle);
    }

    for (const auto& loaderPrivateNeedle : {
        "#include <hcomm/",
        "#include <hccl/",
        "#include \"hcomm/",
        "#include \"hccl/",
        "#include <hccl.h>",
        "#include \"hccl.h\"",
        "pkg_inc/hcomm",
        "pkg_inc/hccl",
        "include/hccl",
        "libhcomm",
        "libhccl_v2",
        "libhccl_fwk",
        "libmc2_client",
        "HcclGetCcuTaskInfo",
        "HcomGetCcuTaskInfo",
    }) {
        CheckNotContains(hccpTypesHeaderPath, hccpTypesHeaderText, loaderPrivateNeedle);
        CheckNotContains(hccpLoaderHeaderPath, hccpLoaderHeaderText, loaderPrivateNeedle);
        CheckNotContains(hccpLoaderSourcePath, hccpLoaderSourceText, loaderPrivateNeedle);
    }

    CheckTrackedFilesContainNeedleOnly(
        "public CCU runtime launch ABI",
        "src/comm src/include",
        "rtCCULaunch",
        {sourcePath});
    CheckTrackedFilesContainNeedleOnly(
        "public CCU runtime kernel header",
        "src/comm src/include",
        "runtime/kernel.h",
        {sourcePath});
}

void TestDirectCcuHeadersOwnStandardDependencies()
{
    const std::string commHeaderPath = "src/comm/tilexr_comm.h";
    const std::string abiConstantsPath = "src/comm/ccu/tilexr_ccu_abi_constants.h";
    const std::string hccpTypesPath = "src/comm/ccu/tilexr_ccu_hccp_types.h";
    const auto commHeaderText = ReadFile(commHeaderPath);
    const auto abiConstantsText = ReadFile(abiConstantsPath);
    const auto hccpTypesText = ReadFile(hccpTypesPath);

    CheckContains(commHeaderPath, commHeaderText, "#include <cstddef>");
    CheckContains(abiConstantsPath, abiConstantsText, "#include <cstdint>");
    CheckContains(hccpTypesPath, hccpTypesText, "#include <cstdint>");
    CheckContains(hccpTypesPath, hccpTypesText, "tilexr_ccu_abi_constants.h");
    CheckContains(hccpTypesPath, hccpTypesText, "TileXRCcuEndpointRouteProviderFunc");
    CheckNotContains(hccpTypesPath, hccpTypesText, "tilexr_ccu_driver_adapter.h");
    CheckNotContains(hccpTypesPath, hccpTypesText, "tilexr_ccu_lower_layer_payloads.h");
    CheckNotContains(abiConstantsPath, abiConstantsText, "udma/");
    CheckNotContains(hccpTypesPath, hccpTypesText, "udma/");
    CheckNotContains(hccpTypesPath, hccpTypesText, "<dlfcn.h>");
    CheckNotContains(hccpTypesPath, hccpTypesText, "dlopen");
    CheckNotContains(hccpTypesPath, hccpTypesText, "dlsym");
    CheckNotContains(hccpTypesPath, hccpTypesText, "std::string");
}

} // namespace

int main()
{
    TestOpenSourceTarballsAreNotTracked();
    TestCommInitChecksDeviceCommArgsSync();
    TestCWrappersDoNotPublishFailedCommunicators();
    TestDumpInitCleansFailedAllocations();
    TestSocketExchangeUsesDirectConnectionsOnly();
    TestRuntimeEnvDoesNotPrependCannDevlib();
    TestRootCMakeRespectsAscendDriverOverride();
    TestCommBuildIncludesProfilingHeaders();
    TestChipNameResolverRecognizesAscend950PrVariants();
    TestCommRuntimeDoesNotUseHcommOrHcclV2();
    TestRootCMakeHcclIncludesAreNotTileCommSurface();
    TestCommDirectCcuInstallAttemptDoesNotSubmit();
    TestCcuRuntimeSubmitBoundaryUsesPublicRuntimeOnly();
    TestDirectCcuHeadersOwnStandardDependencies();

    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR source guard checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR source guard checks passed" << std::endl;
    return 0;
}
