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
    CheckContains(path, contents, "LoadInt32FromGm");
    CheckContains(path, contents, "LoadAssistTupleFromGm");
    CheckContains(path, contents, "StoreWindowHeader");
    CheckContains(path, contents, "StoreSlotHeader");
    CheckContains(path, contents, "StoreAssistTuple");
    CheckContains(path, contents, "localWindow + PayloadOffset(dstRank, slotBytes)");
    CheckContains(path, contents, "localWindow + SlotOffset(dstRank, slotBytes)");
    CheckContains(path, contents, "sourceWindow = shareAddrs[srcRank] + TileXR::IPC_DATA_OFFSET");
    CheckContains(path, contents, "sourceWindow + SlotOffset(rank, slotBytes)");
    CheckNotContains(path, contents, "expertIds[");
    CheckNotContains(path, contents, "assistBase[item]");
    CheckNotContains(path, contents, "args->peerMems[peer]");
    CheckNotContains(path, contents, "shareAddrs[dstRank] + TileXR::IPC_DATA_OFFSET");
    CheckNotContains(path, contents, "slot->count");
    CheckNotContains(path, contents, "assist[index]");
}

void TestNoForbiddenDependencies()
{
    const std::vector<std::string> paths = {
        "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp",
        "src/ep/host/ep_kernel_launch.cpp",
        "src/ep/CMakeLists.txt",
    };
    const std::vector<std::string> forbidden = {
        "src/mc2",
        "3rdparty/ops-transformer",
        "GetHcclContext",
        "TileXRUDMARegister",
        "UDMAPut",
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
    TestNoForbiddenDependencies();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR EP kernel source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR EP kernel source checks passed" << std::endl;
    return 0;
}
