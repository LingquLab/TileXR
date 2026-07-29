/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "sdma/tilexr_sdma_a5_backend.h"

#include <array>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <new>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "driver/ascend_hal.h"
#include "tilexr_log.h"

#ifndef ACL_STREAM_DEVICE_USE_ONLY
#define ACL_STREAM_DEVICE_USE_ONLY 0x00000020U
#endif

namespace TileXR {
namespace {

constexpr size_t kBuiltinWorkspaceBytes = 16U * 1024U;
constexpr aclError kExpectedAicpuQueryFailure =
    static_cast<aclError>(detail::TILEXR_SDMA_A5_EXPECTED_QUERY_STATUS);
constexpr int32_t kDeviceInfoModuleType = 0;
constexpr int32_t kPhysicalDieInfoType = 19;

using RtGetDevicePhyIdByIndexFn = int32_t (*)(uint32_t, uint32_t*);
using RtStreamGetSqidFn = int32_t (*)(const void*, uint32_t*);
using RtStreamGetCqidFn = int32_t (*)(const void*, uint32_t*, uint32_t*);
using RtGetDeviceInfoFn = int32_t (*)(uint32_t, int32_t, int32_t, int64_t*);
using AclCreateTensorFn = aclTensor* (*)(const int64_t*, uint64_t, aclDataType,
                                         const int64_t*, int64_t, aclFormat,
                                         const int64_t*, uint64_t, void*);
using AclDestroyTensorFn = int32_t (*)(const aclTensor*);
using AclnnQueryWorkspaceFn = aclnnStatus (*)(const aclTensor*, aclTensor*,
                                               uint64_t*, aclOpExecutor**);
using AclnnQueryFn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

template <typename T>
bool LoadSymbol(void* handle, const char* name, T& symbol)
{
    symbol = reinterpret_cast<T>(dlsym(handle, name));
    if (symbol != nullptr) {
        return true;
    }
    TILEXR_LOG(WARN) << "TileXR A5 SDMA missing runtime symbol " << name;
    return false;
}

class A5RuntimeApi {
public:
    ~A5RuntimeApi()
    {
        Close();
    }

    bool Load()
    {
        Close();
        runtimeHandle_ = dlopen("libruntime.so", RTLD_NOW | RTLD_LOCAL);
        if (runtimeHandle_ == nullptr) {
            TILEXR_LOG(WARN) << "TileXR A5 SDMA could not load libruntime.so: " << dlerror();
            return false;
        }
        if (!LoadSymbol(runtimeHandle_, "rtGetDevicePhyIdByIndex", getPhysicalDevice) ||
            !LoadSymbol(runtimeHandle_, "rtStreamGetSqid", getSqId) ||
            !LoadSymbol(runtimeHandle_, "rtStreamGetCqid", getCqId) ||
            !LoadSymbol(runtimeHandle_, "rtGetDeviceInfo", getDeviceInfo)) {
            Close();
            return false;
        }

        opapiHandle_ = dlopen("libopapi.so", RTLD_NOW | RTLD_LOCAL);
        if (opapiHandle_ == nullptr) {
            TILEXR_LOG(WARN) << "TileXR A5 SDMA could not load libopapi.so: " << dlerror();
            Close();
            return false;
        }
        if (!LoadSymbol(opapiHandle_, "aclCreateTensor", createTensor) ||
            !LoadSymbol(opapiHandle_, "aclDestroyTensor", destroyTensor) ||
            !LoadSymbol(opapiHandle_, "aclnnShmemSdmaStarsQueryGetWorkspaceSize", prepareQuery) ||
            !LoadSymbol(opapiHandle_, "aclnnShmemSdmaStarsQuery", executeQuery)) {
            Close();
            return false;
        }
        return true;
    }

    void Close()
    {
        getPhysicalDevice = nullptr;
        getSqId = nullptr;
        getCqId = nullptr;
        getDeviceInfo = nullptr;
        createTensor = nullptr;
        destroyTensor = nullptr;
        prepareQuery = nullptr;
        executeQuery = nullptr;
        if (opapiHandle_ != nullptr) {
            (void)dlclose(opapiHandle_);
            opapiHandle_ = nullptr;
        }
        if (runtimeHandle_ != nullptr) {
            (void)dlclose(runtimeHandle_);
            runtimeHandle_ = nullptr;
        }
    }

    RtGetDevicePhyIdByIndexFn getPhysicalDevice = nullptr;
    RtStreamGetSqidFn getSqId = nullptr;
    RtStreamGetCqidFn getCqId = nullptr;
    RtGetDeviceInfoFn getDeviceInfo = nullptr;
    AclCreateTensorFn createTensor = nullptr;
    AclDestroyTensorFn destroyTensor = nullptr;
    AclnnQueryWorkspaceFn prepareQuery = nullptr;
    AclnnQueryFn executeQuery = nullptr;

private:
    void* runtimeHandle_ = nullptr;
    void* opapiHandle_ = nullptr;
};

class DeviceBuffer {
public:
    ~DeviceBuffer()
    {
        Reset();
    }

    bool Allocate(size_t bytes, bool zero = false)
    {
        Reset();
        if (aclrtMalloc(&address_, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return false;
        }
        bytes_ = bytes;
        if (zero && aclrtMemset(address_, bytes_, 0, bytes_) != ACL_SUCCESS) {
            Reset();
            return false;
        }
        return true;
    }

    void Reset()
    {
        if (address_ != nullptr) {
            (void)aclrtFree(address_);
            address_ = nullptr;
            bytes_ = 0U;
        }
    }

    void* Get() const
    {
        return address_;
    }

private:
    void* address_ = nullptr;
    size_t bytes_ = 0U;
};

class TensorHandle {
public:
    explicit TensorHandle(AclDestroyTensorFn destroy) : destroy_(destroy) {}

    ~TensorHandle()
    {
        Reset();
    }

    void Reset()
    {
        if (tensor_ != nullptr) {
            (void)destroy_(tensor_);
            tensor_ = nullptr;
        }
    }

    aclTensor*& Ref()
    {
        return tensor_;
    }

    aclTensor* Get() const
    {
        return tensor_;
    }

private:
    AclDestroyTensorFn destroy_;
    aclTensor* tensor_ = nullptr;
};

struct QuerySnapshot {
    uint32_t flag = 0U;
    uint32_t totalQueueCount = 0U;
    aclError syncStatus = ACL_SUCCESS;
    std::vector<detail::A5BuiltinChannelInfo> channels;
};

bool CreateUint64Tensor(A5RuntimeApi& api, void* address, int64_t elements,
                        TensorHandle& tensor)
{
    const int64_t shape[] = {elements};
    const int64_t strides[] = {1};
    tensor.Ref() = api.createTensor(shape, 1U, ACL_UINT64, strides, 0,
                                    ACL_FORMAT_ND, shape, 1U, address);
    return tensor.Get() != nullptr;
}

bool RestoreQueryContext(aclrtContext previous, aclrtContext& isolated)
{
    const aclError destroyStatus = isolated == nullptr
        ? ACL_SUCCESS
        : aclrtDestroyContext(isolated);
    isolated = nullptr;
    const aclError restoreStatus = aclrtSetCurrentContext(previous);
    if (destroyStatus != ACL_SUCCESS || restoreStatus != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA query context restore failed, destroy "
                         << destroyStatus << ", restore " << restoreStatus;
        return false;
    }
    return true;
}

bool CheckRuntimeHealth(DeviceBuffer& scratch)
{
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        return false;
    }
    const aclError memsetStatus = aclrtMemsetAsync(
        scratch.Get(), sizeof(uint64_t), 0xA5, sizeof(uint64_t), stream);
    const aclError syncStatus = memsetStatus == ACL_SUCCESS
        ? aclrtSynchronizeStream(stream)
        : memsetStatus;
    uint64_t value = 0U;
    const aclError copyStatus = syncStatus == ACL_SUCCESS
        ? aclrtMemcpy(&value, sizeof(value), scratch.Get(), sizeof(value), ACL_MEMCPY_DEVICE_TO_HOST)
        : syncStatus;
    const aclError destroyStatus = aclrtDestroyStream(stream);
    return memsetStatus == ACL_SUCCESS && syncStatus == ACL_SUCCESS &&
        copyStatus == ACL_SUCCESS && destroyStatus == ACL_SUCCESS &&
        value == 0xA5A5A5A5A5A5A5A5ULL;
}

bool RunBuiltinQuery(A5RuntimeApi& api,
                     int32_t logicalDevice,
                     const std::vector<detail::A5BuiltinStreamInfo>& streams,
                     QuerySnapshot& snapshot)
{
    if (streams.empty() || streams.size() > detail::TILEXR_SDMA_A5_CHANNEL_COUNT) {
        return false;
    }

    const size_t streamsBytes = streams.size() * sizeof(detail::A5BuiltinStreamInfo);
    DeviceBuffer streamsDev;
    DeviceBuffer resourceDev;
    DeviceBuffer builtinWorkspaceDev;
    DeviceBuffer inputDev;
    DeviceBuffer outputDev;
    DeviceBuffer opWorkspaceDev;
    if (!streamsDev.Allocate(streamsBytes) ||
        !resourceDev.Allocate(sizeof(detail::A5BuiltinOpResource), true) ||
        !builtinWorkspaceDev.Allocate(kBuiltinWorkspaceBytes, true) ||
        !inputDev.Allocate(2U * sizeof(uint64_t)) ||
        !outputDev.Allocate(sizeof(uint64_t), true) ||
        aclrtMemcpy(streamsDev.Get(), streamsBytes, streams.data(), streamsBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA query buffer setup failed";
        return false;
    }

    detail::A5BuiltinOpResource resource {};
    resource.size = streams.size();
    resource.streamsAddress = reinterpret_cast<uint64_t>(streamsDev.Get());
    resource.workspaceAddress = reinterpret_cast<uint64_t>(builtinWorkspaceDev.Get());
    const uint64_t inputs[] = {
        reinterpret_cast<uint64_t>(resourceDev.Get()),
        reinterpret_cast<uint64_t>(builtinWorkspaceDev.Get()),
    };
    if (aclrtMemcpy(resourceDev.Get(), sizeof(resource), &resource, sizeof(resource),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(inputDev.Get(), sizeof(inputs), inputs, sizeof(inputs),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA query resource upload failed";
        return false;
    }

    aclrtContext previous = nullptr;
    aclrtContext isolated = nullptr;
    if (aclrtGetCurrentContext(&previous) != ACL_SUCCESS ||
        aclrtCreateContext(&isolated, logicalDevice) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA isolated query context creation failed";
        if (isolated != nullptr) {
            (void)RestoreQueryContext(previous, isolated);
        }
        return false;
    }

    TensorHandle inputTensor(api.destroyTensor);
    TensorHandle outputTensor(api.destroyTensor);
    aclrtStream queryStream = nullptr;
    bool queryLaunched = false;
    aclError syncStatus = ACL_SUCCESS;
    uint64_t opWorkspaceBytes = 0U;
    aclOpExecutor* executor = nullptr;
    bool ok = CreateUint64Tensor(api, inputDev.Get(), 2, inputTensor) &&
        CreateUint64Tensor(api, outputDev.Get(), 1, outputTensor) &&
        api.prepareQuery(inputTensor.Get(), outputTensor.Get(), &opWorkspaceBytes, &executor) == ACL_SUCCESS;
    if (ok && opWorkspaceBytes != 0U) {
        ok = opWorkspaceDev.Allocate(static_cast<size_t>(opWorkspaceBytes));
    }
    if (ok) {
        ok = aclrtCreateStreamWithConfig(
            &queryStream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStreamAttrValue failureMode {};
        failureMode.failureMode = 0;
        ok = aclrtSetStreamAttribute(
            queryStream, ACL_STREAM_ATTR_FAILURE_MODE, &failureMode) == ACL_SUCCESS;
    }
    if (ok) {
        const aclnnStatus launchStatus = api.executeQuery(
            opWorkspaceDev.Get(), opWorkspaceBytes, executor, queryStream);
        queryLaunched = launchStatus == ACL_SUCCESS;
        ok = queryLaunched;
        if (queryLaunched) {
            syncStatus = aclrtSynchronizeStream(queryStream);
        }
    }

    aclError streamDestroyStatus = ACL_SUCCESS;
    if (queryStream != nullptr) {
        streamDestroyStatus = aclrtDestroyStream(queryStream);
        queryStream = nullptr;
    }
    inputTensor.Reset();
    outputTensor.Reset();
    const bool contextRestored = RestoreQueryContext(previous, isolated);
    if (!ok || !queryLaunched || streamDestroyStatus != ACL_SUCCESS || !contextRestored) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA built-in query launch or cleanup failed";
        return false;
    }

    const size_t snapshotBytes = sizeof(detail::A5BuiltinWorkspaceHeader) +
        streams.size() * sizeof(detail::A5BuiltinChannelInfo);
    std::vector<uint8_t> bytes(snapshotBytes, 0U);
    if (aclrtMemcpy(bytes.data(), bytes.size(), builtinWorkspaceDev.Get(), bytes.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA query workspace download failed";
        return false;
    }
    detail::A5BuiltinWorkspaceHeader header {};
    std::memcpy(&header, bytes.data(), sizeof(header));
    snapshot.flag = header.flag;
    snapshot.totalQueueCount = header.totalQueueCount;
    snapshot.syncStatus = syncStatus;
    snapshot.channels.resize(streams.size());
    std::memcpy(snapshot.channels.data(), bytes.data() + sizeof(header),
                snapshot.channels.size() * sizeof(snapshot.channels[0]));

    if (syncStatus == kExpectedAicpuQueryFailure && !CheckRuntimeHealth(outputDev)) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA context health check failed after expected AICPU error";
        return false;
    }
    return true;
}

int32_t QueryHostSq(uint32_t physicalDevice, uint32_t sqId,
                    drvSqCqPropType_t property, uint32_t (&values)[3])
{
    halSqCqQueryInfo query {};
    query.type = DRV_NORMAL_TYPE;
    query.tsId = 0U;
    query.sqId = sqId;
    query.cqId = 0U;
    query.prop = property;
    const drvError_t status = halSqCqQuery(physicalDevice, &query);
    values[0] = query.value[0];
    values[1] = query.value[1];
    values[2] = query.value[2];
    return static_cast<int32_t>(status);
}

} // namespace

struct TileXRA5SDMABackend::Impl {
    struct OwnedChannel {
        aclrtStream stream = nullptr;
        res_addr_info mapInfo {};
        bool mapped = false;
        uint64_t rtsqAddress = 0U;
        uint32_t rtsqLength = 0U;
        detail::A5HostChannelIdentity identity {};
        detail::A5BuiltinChannelInfo query {};
    };

    int32_t logicalDevice = -1;
    uint32_t physicalDevice = 0U;
    uint32_t physicalDieId = 0U;
    aclrtContext ownerContext = nullptr;
    aclrtContext restoreContext = nullptr;
    bool restorePending = false;
    void* workspaceDev = nullptr;
    std::array<OwnedChannel, detail::TILEXR_SDMA_A5_CHANNEL_COUNT> channels {};

    bool HasOwnedResources() const
    {
        if (workspaceDev != nullptr) {
            return true;
        }
        for (const OwnedChannel& channel : channels) {
            if (channel.mapped || channel.stream != nullptr) {
                return true;
            }
        }
        return false;
    }
};

TileXRA5SDMABackend::TileXRA5SDMABackend() = default;

TileXRA5SDMABackend::~TileXRA5SDMABackend()
{
    (void)Shutdown();
}

bool TileXRA5SDMABackend::Init(int32_t deviceId)
{
    if (!Shutdown()) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA previous resources could not be released";
        return false;
    }
    std::unique_ptr<Impl> state(new (std::nothrow) Impl());
    if (state == nullptr) {
        return false;
    }
    state->logicalDevice = deviceId;
    if (aclrtGetCurrentContext(&state->ownerContext) != ACL_SUCCESS ||
        state->ownerContext == nullptr) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA requires an active owner context";
        return false;
    }

    A5RuntimeApi api;
    if (!api.Load() ||
        api.getPhysicalDevice(static_cast<uint32_t>(deviceId), &state->physicalDevice) != 0) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA runtime discovery failed";
        return false;
    }
    int64_t physicalDie = -1;
    if (api.getDeviceInfo(static_cast<uint32_t>(deviceId), kDeviceInfoModuleType,
                          kPhysicalDieInfoType, &physicalDie) != 0 ||
        physicalDie < 0 || static_cast<uint64_t>(physicalDie) >
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA physical die discovery failed";
        return false;
    }
    state->physicalDieId = static_cast<uint32_t>(physicalDie);
    impl_ = std::move(state);

    std::vector<detail::A5BuiltinStreamInfo> streamInfos;
    streamInfos.reserve(detail::TILEXR_SDMA_A5_CHANNEL_COUNT);
    for (uint32_t index = 0U; index < detail::TILEXR_SDMA_A5_CHANNEL_COUNT; ++index) {
        Impl::OwnedChannel& owned = impl_->channels[index];
        if (aclrtCreateStreamWithConfig(
                &owned.stream, 0, ACL_STREAM_DEVICE_USE_ONLY) != ACL_SUCCESS) {
            TILEXR_LOG(WARN) << "TileXR A5 SDMA STARS stream creation failed at channel " << index;
            Shutdown();
            return false;
        }
        int32_t streamId = -1;
        uint32_t sqId = 0U;
        uint32_t cqId = 0U;
        uint32_t logicalCqId = 0U;
        if (aclrtStreamGetId(owned.stream, &streamId) != ACL_SUCCESS || streamId < 0 ||
            api.getSqId(owned.stream, &sqId) != 0 ||
            api.getCqId(owned.stream, &cqId, &logicalCqId) != 0) {
            TILEXR_LOG(WARN) << "TileXR A5 SDMA stream identifier query failed at channel " << index;
            Shutdown();
            return false;
        }
        owned.identity = {
            static_cast<uint32_t>(streamId), sqId, cqId, logicalCqId, impl_->physicalDieId,
        };
        owned.mapInfo.id = 0U;
        owned.mapInfo.target_proc_type = PROCESS_CP1;
        owned.mapInfo.res_type = RES_ADDR_TYPE_STARS_RTSQ;
        owned.mapInfo.res_id = sqId;
        unsigned long mappedAddress = 0UL;
        unsigned int mappedLength = 0U;
        const drvError_t mapStatus = halResAddrMap(
            impl_->physicalDevice, &owned.mapInfo, &mappedAddress, &mappedLength);
        owned.mapped = mapStatus == DRV_ERROR_NONE;
        if (mapStatus != DRV_ERROR_NONE || mappedAddress == 0UL ||
            mappedLength < sizeof(uint32_t)) {
            TILEXR_LOG(WARN) << "TileXR A5 SDMA RTSQ map failed at channel " << index
                             << ", status " << mapStatus;
            Shutdown();
            return false;
        }
        owned.rtsqAddress = static_cast<uint64_t>(mappedAddress);
        owned.rtsqLength = mappedLength;

        detail::A5BuiltinStreamInfo info {};
        info.stream = reinterpret_cast<uint64_t>(owned.stream);
        info.context = reinterpret_cast<uint64_t>(impl_->ownerContext);
        info.streamId = streamId;
        info.sqId = sqId;
        info.cqId = cqId;
        info.logicalCqId = logicalCqId;
        info.deviceId = static_cast<int32_t>(impl_->physicalDieId);
        streamInfos.push_back(info);
    }

    QuerySnapshot batch;
    if (!RunBuiltinQuery(api, deviceId, streamInfos, batch)) {
        Shutdown();
        return false;
    }
    const detail::A5QueryResultKind batchKind = batch.channels.empty()
        ? detail::A5QueryResultKind::INVALID
        : detail::ClassifyA5QueryResult(
            static_cast<int32_t>(batch.syncStatus), batch.flag, batch.totalQueueCount,
            batch.channels[0], impl_->channels[0].identity);
    if (batchKind == detail::A5QueryResultKind::COMPLETE) {
        if (batch.channels.size() != detail::TILEXR_SDMA_A5_CHANNEL_COUNT) {
            Shutdown();
            return false;
        }
        for (uint32_t index = 0U; index < detail::TILEXR_SDMA_A5_CHANNEL_COUNT; ++index) {
            if (!detail::ValidateA5BuiltinChannel(
                    batch.channels[index], impl_->channels[index].identity, true)) {
                TILEXR_LOG(WARN) << "TileXR A5 SDMA complete query validation failed at channel " << index;
                Shutdown();
                return false;
            }
            impl_->channels[index].query = batch.channels[index];
        }
    } else if (batchKind == detail::A5QueryResultKind::EXPECTED_PARTIAL) {
        impl_->channels[0].query = batch.channels[0];
        for (uint32_t index = 1U; index < detail::TILEXR_SDMA_A5_CHANNEL_COUNT; ++index) {
            std::vector<detail::A5BuiltinStreamInfo> oneStream(1U, streamInfos[index]);
            QuerySnapshot isolated;
            if (!RunBuiltinQuery(api, deviceId, oneStream, isolated) ||
                isolated.channels.size() != 1U ||
                detail::ClassifyA5QueryResult(
                    static_cast<int32_t>(isolated.syncStatus), isolated.flag,
                    isolated.totalQueueCount, isolated.channels[0],
                    impl_->channels[index].identity) !=
                    detail::A5QueryResultKind::EXPECTED_PARTIAL) {
                TILEXR_LOG(WARN) << "TileXR A5 SDMA isolated query validation failed at channel " << index;
                Shutdown();
                return false;
            }
            impl_->channels[index].query = isolated.channels[0];
        }
    } else {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA query returned unsupported status " << batch.syncStatus;
        Shutdown();
        return false;
    }

    detail::A5SdmaWorkspace hostWorkspace {};
    hostWorkspace.header.magic = detail::TILEXR_SDMA_A5_WORKSPACE_MAGIC;
    hostWorkspace.header.abiVersion = detail::TILEXR_SDMA_A5_ABI_VERSION;
    hostWorkspace.header.backendKind = detail::TILEXR_SDMA_A5_BACKEND_KIND;
    hostWorkspace.header.channelCount = detail::TILEXR_SDMA_A5_CHANNEL_COUNT;
    hostWorkspace.header.sqeSize = detail::TILEXR_SDMA_A5_SQE_BYTES;
    hostWorkspace.header.channelStride = sizeof(detail::A5SdmaChannel);
    hostWorkspace.header.workspaceSize = sizeof(hostWorkspace);
    hostWorkspace.header.maxTransferBytes =
        static_cast<uint32_t>(detail::TILEXR_SDMA_A5_MAX_TRANSFER_BYTES);

    if (aclrtMalloc(&impl_->workspaceDev, sizeof(hostWorkspace),
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA workspace allocation failed";
        Shutdown();
        return false;
    }
    const uint64_t workspaceBase = reinterpret_cast<uint64_t>(impl_->workspaceDev);
    for (uint32_t index = 0U; index < detail::TILEXR_SDMA_A5_CHANNEL_COUNT; ++index) {
        const Impl::OwnedChannel& owned = impl_->channels[index];
        detail::A5SdmaChannel& channel = hostWorkspace.channels[index];
        channel.sqBase = owned.query.sqBase;
        channel.rtsqAddress = owned.rtsqAddress;
        channel.completionPayloadAddress = workspaceBase +
            offsetof(detail::A5SdmaWorkspace, completionPayloads) +
            index * sizeof(detail::A5SdmaCompletionLine);
        channel.completionRecordAddress = workspaceBase +
            offsetof(detail::A5SdmaWorkspace, completionRecords) +
            index * sizeof(detail::A5SdmaCompletionLine);
        channel.depth = owned.query.sqDepth;
        channel.head = owned.query.sqHead;
        channel.tail = owned.query.sqTail;
        channel.taskId = detail::A5SdmaQueueDistance(
            owned.query.sqHead, owned.query.sqTail, owned.query.sqDepth);
        channel.rtsqLength = owned.rtsqLength;
        channel.streamId = owned.identity.streamId;
        channel.sqId = owned.identity.sqId;
        channel.cqId = owned.identity.cqId;
        channel.logicalCqId = owned.identity.logicalCqId;
        channel.physicalDieId = owned.identity.physicalDieId;

        uint32_t tailValues[3] = {0U, 0U, 0U};
        uint32_t sqeSizeValues[3] = {0U, 0U, 0U};
        if (QueryHostSq(impl_->physicalDevice, channel.sqId,
                        DRV_SQCQ_PROP_SQ_TAIL, tailValues) != 0 ||
            QueryHostSq(impl_->physicalDevice, channel.sqId,
                        DRV_SQCQ_PROP_SQE_SIZE, sqeSizeValues) != 0 ||
            tailValues[0] != channel.tail ||
            sqeSizeValues[0] != detail::TILEXR_SDMA_A5_SQE_BYTES) {
            TILEXR_LOG(WARN) << "TileXR A5 SDMA Host SQ cross-check failed at channel " << index;
            Shutdown();
            return false;
        }
    }
    if (aclrtMemcpy(impl_->workspaceDev, sizeof(hostWorkspace), &hostWorkspace,
                    sizeof(hostWorkspace), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA workspace upload failed";
        Shutdown();
        return false;
    }

    TILEXR_LOG(INFO) << "TileXR A5 direct SDMA initialized on device " << deviceId
                     << " with " << detail::TILEXR_SDMA_A5_CHANNEL_COUNT << " channels";
    return true;
}

bool TileXRA5SDMABackend::Shutdown()
{
    if (impl_ == nullptr) {
        return true;
    }
    if (impl_->restorePending) {
        if (aclrtSetCurrentContext(impl_->restoreContext) != ACL_SUCCESS) {
            TILEXR_LOG(WARN) << "TileXR A5 SDMA could not complete pending context restore";
            return false;
        }
        impl_->restoreContext = nullptr;
        impl_->restorePending = false;
        if (!impl_->HasOwnedResources()) {
            impl_.reset();
            return true;
        }
    }

    aclrtContext previous = nullptr;
    if (aclrtGetCurrentContext(&previous) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA could not capture current context for cleanup";
        return false;
    }
    if (impl_->ownerContext == nullptr ||
        aclrtSetCurrentContext(impl_->ownerContext) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA could not switch to owner context for cleanup";
        return false;
    }

    if (impl_->workspaceDev != nullptr) {
        if (aclrtFree(impl_->workspaceDev) == ACL_SUCCESS) {
            impl_->workspaceDev = nullptr;
        }
    }
    if (impl_->workspaceDev == nullptr) {
        for (size_t reverse = detail::TILEXR_SDMA_A5_CHANNEL_COUNT; reverse > 0U; --reverse) {
            Impl::OwnedChannel& owned = impl_->channels[reverse - 1U];
            bool mappingReleased = true;
            if (owned.mapped) {
                if (halResAddrUnmap(impl_->physicalDevice, &owned.mapInfo) == DRV_ERROR_NONE) {
                    owned.mapped = false;
                } else {
                    mappingReleased = false;
                }
            }
            if (mappingReleased && owned.stream != nullptr &&
                aclrtDestroyStream(owned.stream) == ACL_SUCCESS) {
                owned.stream = nullptr;
            }
        }
    }
    if (previous != impl_->ownerContext &&
        aclrtSetCurrentContext(previous) != ACL_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA could not restore previous context after cleanup";
        impl_->restoreContext = previous;
        impl_->restorePending = true;
        return false;
    }
    if (impl_->HasOwnedResources()) {
        TILEXR_LOG(WARN) << "TileXR A5 SDMA cleanup incomplete; retained resources for retry";
        return false;
    }
    impl_.reset();
    return true;
}

GM_ADDR TileXRA5SDMABackend::GetWorkspaceDev() const
{
    return impl_ == nullptr ? nullptr : static_cast<GM_ADDR>(impl_->workspaceDev);
}

} // namespace TileXR
