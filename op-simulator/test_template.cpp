/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>
#include <rt.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

inline void CheckRtError(rtError_t error, const std::string& msg)
{
    if (error != RT_ERROR_NONE) {
        throw std::runtime_error(msg + " | Error Code: " + std::to_string(error));
    }
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

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : size_(count * sizeof(T))
    {
        CheckRtError(rtMalloc(&ptr_, size_, RT_MEMORY_HBM, 0), "rtMalloc failed");
    }
    ~DeviceBuffer()
    {
        if (ptr_) {
            rtFree(ptr_);
        }
    }
    T* Get() const
    {
        return static_cast<T*>(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
private:
    void* ptr_{nullptr};
    size_t size_;
};

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

void RunKernel(const std::string& funcName, const std::string& binFile, uint32_t blockNum)
{
    KernelManager manager(funcName, binFile);

    constexpr int32_t totalDataLen = 51200 * 20;
    DeviceBuffer<uint16_t> x(totalDataLen);
    DeviceBuffer<uint16_t> perm(totalDataLen);
    DeviceBuffer<uint16_t> y(totalDataLen);
    DeviceBuffer<uint16_t> workspace(totalDataLen);
    DeviceBuffer<uint16_t> tiling(totalDataLen);

    struct KernelArgs {
        void *x, *perm, *y, *workspace, *tiling;
    } args {
        x.Get(), perm.Get(), y.Get(), workspace.Get(), tiling.Get()
    };

    RtStream stream;
    std::cout << "Launching kernel: " << funcName << " with blockNum: " << blockNum << std::endl;

    CheckRtError(rtKernelLaunch(manager.GetFuncName().data(),
                                blockNum, &args, sizeof(args), nullptr, stream.Get()),
                 "Kernel launch failed");

    stream.Synchronize();
}

int main(int argc, char* argv[])
{
    try {
        const char* envName = std::getenv("KERNEL_NAME");
        std::string kernelName = (envName != nullptr) ? envName : "DemoTest";

        uint32_t blockNum = 16;
        if (argc > 1) {
            blockNum = static_cast<uint32_t>(std::stoul(argv[1]));
        }

        if (aclInit(nullptr) != ACL_SUCCESS) {
            throw std::runtime_error("aclInit failed");
        }

        CheckRtError(rtSetDevice(0), "rtSetDevice failed");
        RunKernel(kernelName, "./op/my_kernel.o", blockNum);

        rtDeviceReset(0);
        aclFinalize();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
