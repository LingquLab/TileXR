#include <acl/acl.h>
#include <rt.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

inline void CheckRtError(rtError_t error, const std::string& msg)
{
    if (error != RT_ERROR_NONE) {
        throw std::runtime_error(msg + " | Error Code: " + std::to_string(error));
    }
}

std::vector<char> ReadBinFile(const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + fileName);
    }
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file content: " + fileName);
    }
    return buffer;
}

class RtStream {
public:
    RtStream()
    {
        CheckRtError(rtStreamCreate(&stream_, 0), "Failed to create stream");
    }

    ~RtStream()
    {
        if (stream_) {
            rtStreamDestroy(stream_);
        }
    }

    rtStream_t Get() const
    {
        return stream_;
    }

    void Synchronize()
    {
        CheckRtError(rtStreamSynchronize(stream_), "Stream sync failed");
    }

private:
    rtStream_t stream_{nullptr};
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : size_(bytes)
    {
        CheckRtError(rtMalloc(&ptr_, size_, RT_MEMORY_HBM, 0), "rtMalloc failed");
    }

    ~DeviceBuffer()
    {
        if (ptr_) {
            rtFree(ptr_);
        }
    }

    void* Get() const
    {
        return ptr_;
    }

    size_t Size() const
    {
        return size_;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

private:
    void* ptr_{nullptr};
    size_t size_{0};
};

class KernelManager {
public:
    KernelManager(std::string funcName, const std::string& binFile) : functionName_(std::move(funcName))
    {
        binData_ = ReadBinFile(binFile);
        binary_.data = binData_.data();
        binary_.length = static_cast<uint32_t>(binData_.size());
        binary_.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
        binary_.version = 0;

        CheckRtError(rtDevBinaryRegister(&binary_, &binHandle_), "rtDevBinaryRegister failed");
        CheckRtError(rtFunctionRegister(binHandle_, functionName_.data(), functionName_.data(),
                                        functionName_.data(), 0),
                     "rtFunctionRegister failed");
    }

    const std::string& GetFuncName() const
    {
        return functionName_;
    }

private:
    std::string functionName_;
    std::vector<char> binData_;
    rtDevBinary_t binary_{};
    void* binHandle_{nullptr};
};

static size_t GetEnvSize(const char* name, size_t fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return static_cast<size_t>(std::stoull(value));
}

static std::vector<int32_t> BuildExpertIds(size_t bs, size_t axisK, size_t totalExperts)
{
    std::vector<int32_t> ids(bs * axisK);
    for (size_t token = 0; token < bs; ++token) {
        for (size_t k = 0; k < axisK; ++k) {
            ids[token * axisK + k] = static_cast<int32_t>((token * 37 + k * 17 + (token / 3) * 11) % totalExperts);
        }
    }
    return ids;
}

static void FillPackedWindow(std::vector<unsigned char>& workspace, size_t bs, size_t axisK, size_t hAlignWinSize,
    size_t splitBlockDataSize, size_t splitBlockFlagSize, size_t blockCntPerToken)
{
    for (size_t token = 0; token < bs; ++token) {
        for (size_t k = 0; k < axisK; ++k) {
            size_t slotBase = (token * axisK + k) * hAlignWinSize;
            for (size_t block = 0; block < blockCntPerToken; ++block) {
                size_t blockBase = slotBase + block * (splitBlockDataSize + splitBlockFlagSize);
                for (size_t i = 0; i < splitBlockDataSize; ++i) {
                    workspace[blockBase + i] = static_cast<unsigned char>((token * 13 + k * 29 + block * 7 + i) & 0xff);
                }
                float* flags = reinterpret_cast<float*>(workspace.data() + blockBase + splitBlockDataSize);
                for (size_t f = 0; f < splitBlockFlagSize / sizeof(float); ++f) {
                    flags[f] = 1.0f;
                }
            }
        }
    }
}

void RunKernel(const std::string& funcName, const std::string& binFile, uint32_t blockNum)
{
    KernelManager manager(funcName, binFile);

    constexpr size_t axisH = 5120;
    constexpr size_t axisK = 6;
    constexpr size_t totalExperts = 128;
    constexpr size_t splitBlockDataSize = 480;
    constexpr size_t splitBlockFlagSize = 32;

    size_t bs = GetEnvSize("CASE_BS", 32);
    size_t ep = GetEnvSize("CASE_EP", 32);
    size_t blockCntPerToken = ((axisH + 255) & ~static_cast<size_t>(255)) / sizeof(uint16_t);
    blockCntPerToken += (((axisH + 31) >> 5) + 1) & ~static_cast<size_t>(1);
    blockCntPerToken = ((blockCntPerToken * sizeof(uint16_t)) + splitBlockDataSize - 1) / splitBlockDataSize;
    size_t hAlignWinSize = blockCntPerToken * (splitBlockDataSize + splitBlockFlagSize);

    size_t scaleCount = bs * axisK;
    size_t workspaceBytes = scaleCount * hAlignWinSize;
    size_t outputBytes = axisH * sizeof(float);
    size_t rankStatsBytes = (128 + 16) * sizeof(uint32_t);

    DeviceBuffer expertIds(scaleCount * sizeof(int32_t));
    DeviceBuffer expertScales(scaleCount * sizeof(float));
    DeviceBuffer output(outputBytes);
    DeviceBuffer rankStats(rankStatsBytes);
    DeviceBuffer workspace(workspaceBytes);
    DeviceBuffer tiling(4096);

    std::vector<int32_t> idsHost = BuildExpertIds(bs, axisK, totalExperts);
    std::vector<float> scalesHost(scaleCount, 1.0f);
    std::vector<unsigned char> outputHost(outputBytes, 0);
    std::vector<uint32_t> rankStatsHost((128 + 16), 0);
    std::vector<unsigned char> workspaceHost(workspaceBytes, 0);
    FillPackedWindow(workspaceHost, bs, axisK, hAlignWinSize, splitBlockDataSize, splitBlockFlagSize, blockCntPerToken);

    CheckRtError(rtMemcpy(expertIds.Get(), expertIds.Size(), idsHost.data(), idsHost.size() * sizeof(int32_t),
                          RT_MEMCPY_HOST_TO_DEVICE),
                 "rtMemcpy expertIds failed");
    CheckRtError(rtMemcpy(expertScales.Get(), expertScales.Size(), scalesHost.data(), scalesHost.size() * sizeof(float),
                          RT_MEMCPY_HOST_TO_DEVICE),
                 "rtMemcpy expertScales failed");
    CheckRtError(rtMemcpy(output.Get(), output.Size(), outputHost.data(), outputHost.size(),
                          RT_MEMCPY_HOST_TO_DEVICE),
                 "rtMemcpy output failed");
    CheckRtError(rtMemcpy(rankStats.Get(), rankStats.Size(), rankStatsHost.data(), rankStatsHost.size() * sizeof(uint32_t),
                          RT_MEMCPY_HOST_TO_DEVICE),
                 "rtMemcpy rankStats failed");
    CheckRtError(rtMemcpy(workspace.Get(), workspace.Size(), workspaceHost.data(), workspaceHost.size(),
                          RT_MEMCPY_HOST_TO_DEVICE),
                 "rtMemcpy workspace failed");

    struct KernelArgs {
        void *expertIds, *expertScales, *output, *rankStats, *workspace, *tiling;
    } args {
        expertIds.Get(), expertScales.Get(), output.Get(), rankStats.Get(), workspace.Get(), tiling.Get()
    };

    RtStream stream;
    std::cout << "Launching kernel: " << funcName
              << " ep=" << ep
              << " totalExperts=" << totalExperts
              << " expertsPerRank=" << (totalExperts / ep)
              << " bs=" << bs
              << " topK=" << axisK
              << " h=" << axisH
              << " loopOnly=1"
              << " blockCntPerToken=" << blockCntPerToken
              << " hAlignWinSize=" << hAlignWinSize
              << " workspaceBytes=" << workspaceBytes
              << " blockNum=" << blockNum << std::endl;

    CheckRtError(rtKernelLaunch(manager.GetFuncName().data(), blockNum, &args, sizeof(args), nullptr, stream.Get()),
                 "Kernel launch failed");
    stream.Synchronize();

    CheckRtError(rtMemcpy(rankStatsHost.data(), rankStatsHost.size() * sizeof(uint32_t), rankStats.Get(),
                          rankStats.Size(), RT_MEMCPY_DEVICE_TO_HOST),
                 "rtMemcpy rankStats back failed");
    size_t nonZeroRanks = 0;
    size_t maxRankCount = 0;
    for (size_t i = 0; i < std::min(ep, static_cast<size_t>(128)); ++i) {
        if (rankStatsHost[i] != 0) {
            nonZeroRanks++;
            maxRankCount = std::max(maxRankCount, static_cast<size_t>(rankStatsHost[i]));
        }
    }
    std::cout << "RankStats ep=" << rankStatsHost[128]
              << " totalExperts=" << rankStatsHost[129]
              << " expertsPerRank=" << rankStatsHost[130]
              << " bs=" << rankStatsHost[131]
              << " topK=" << rankStatsHost[132]
              << " blockCntPerToken=" << rankStatsHost[133]
              << " hAlignWinSize=" << rankStatsHost[134]
              << " nonZeroRanks=" << nonZeroRanks
              << " maxRankCount=" << maxRankCount << std::endl;
}

int main(int argc, char* argv[])
{
    try {
        const char* envName = std::getenv("KERNEL_NAME");
        std::string kernelName = (envName != nullptr) ? envName : "MoeEp32Bs32Baseline";
        const char* envObj = std::getenv("KERNEL_OBJECT");
        std::string kernelObject = (envObj != nullptr) ? envObj : "/tmp/tilexr_moe_epcard_compare/op/MoeEp32Bs32Baseline/my_kernel.o";

        uint32_t blockNum = 1;
        if (argc > 1) {
            blockNum = static_cast<uint32_t>(std::stoul(argv[1]));
        }

        if (aclInit(nullptr) != ACL_SUCCESS) {
            throw std::runtime_error("aclInit failed");
        }

        CheckRtError(rtSetDevice(0), "rtSetDevice failed");
        RunKernel(kernelName, kernelObject, blockNum);

        rtDeviceReset(0);
        aclFinalize();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
