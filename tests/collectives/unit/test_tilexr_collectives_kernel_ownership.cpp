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

    const std::string kernelsCmakePath = "src/collectives/kernels/CMakeLists.txt";
    const auto kernelsCmake = ReadFile(kernelsCmakePath);
    CheckContains(kernelsCmakePath, kernelsCmake, "enable_language(CCE)");
    CheckContains(kernelsCmakePath, kernelsCmake, "tilexr_lccl_op.cpp");
    CheckContains(kernelsCmakePath, kernelsCmake, "tilexr_collectives_op.o");
    CheckContains(kernelsCmakePath, kernelsCmake, "tilexr_collectives_op");
    CheckDoesNotContain(kernelsCmakePath, kernelsCmake, "src/comm");
}

void TestCollectivesKernelSourcesAreScoped()
{
    const std::string kernelTuPath = "src/collectives/kernels/tilexr_lccl_op.cpp";
    const auto kernelTu = ReadFile(kernelTuPath);
    CheckContains(kernelTuPath, kernelTu, "LCCL_TYPE_AIV_FUNC(LCCL_ALLGATHER_FUNC_AUTO_DEF)");
    CheckContains(kernelTuPath, kernelTu, "LCCL_TYPE_AIV_FUNC(LCCL_ALL2ALL_FUNC_AUTO_DEF)");
    CheckDoesNotContain(kernelTuPath, kernelTu, "LCCL_ALL_REDUCE_FUNC_AUTO_DEF");
    CheckDoesNotContain(kernelTuPath, kernelTu, "LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF");
    CheckDoesNotContain(kernelTuPath, kernelTu, "LCCL_BROADCAST_FUNC_AUTO_DEF");

    const std::vector<std::string> kernelFiles = CollectFiles("src/collectives/kernels");
    CheckTrue(!kernelFiles.empty(), "expected collectives kernel files to be present");
    bool sawAllGatherCce = false;
    bool sawAllToAllCce = false;
    for (const auto &path : kernelFiles) {
        const auto text = ReadFile(path);
        CheckDoesNotContain(path, text, "tilexr_comm.h");
        if (path.find("lcal_allgather") != std::string::npos && path.find(".cce") != std::string::npos) {
            sawAllGatherCce = true;
        }
        if (path.find("lcal_all2all_transpose.cce") != std::string::npos) {
            sawAllToAllCce = true;
        }
    }
    CheckTrue(sawAllGatherCce, "expected copied allgather .cce sources under src/collectives/kernels");
    CheckTrue(sawAllToAllCce, "expected copied all2all .cce source under src/collectives/kernels");
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
    CheckContains(kernelPath, kernel, "TileXR::TileXRType::ALL_GATHER");
    CheckContains(kernelPath, kernel, "TileXR::TileXRType::ALL2ALL");
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
    TestDeviceKernelArgsMatchHostLaunchAbi();
    TestCommDoesNotOwnCollectiveRuntime();
    return g_failures == 0 ? 0 : 1;
}
