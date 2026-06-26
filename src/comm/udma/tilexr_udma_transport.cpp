/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_transport.h"

#include <acl/acl_rt.h>
#include <algorithm>
#include <climits>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <unordered_map>

#include "tilexr_log.h"
#include "tools/socket/tilexr_sock_exchange.h"

namespace TileXR {
namespace {

uint32_t Log2Uint64(uint64_t value)
{
    uint32_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

HccpEid SwapEidForDevice(const HccpEid& hccpEid)
{
    HccpEid swapped {};
    uint64_t eidL = 0;
    uint64_t eidH = 0;
    std::memcpy(&eidL, hccpEid.raw, sizeof(uint64_t));
    std::memcpy(&eidH, hccpEid.raw + sizeof(uint64_t), sizeof(uint64_t));
    eidL = __builtin_bswap64(eidL);
    eidH = __builtin_bswap64(eidH);
    std::memcpy(swapped.raw, &eidH, sizeof(uint64_t));
    std::memcpy(swapped.raw + sizeof(uint64_t), &eidL, sizeof(uint64_t));
    return swapped;
}

struct TileXRRootInfo {
    std::string topoPath;
    uint32_t deviceIdOffset = 0;
    uint32_t eidCount = 0;
    std::unordered_map<uint32_t, uint32_t> deviceToLocalId;
    std::unordered_map<uint32_t, std::unordered_map<std::string, uint32_t>> portToEidByLocalId;
    std::unordered_map<uint32_t, std::map<uint32_t, HccpEid>> eidByLocalId;
};

struct TileXRTopoEdge {
    uint32_t localA = 0;
    uint32_t localB = 0;
    std::vector<std::string> localAPorts;
    std::vector<std::string> localBPorts;
};

std::string ReadTextFile(const std::string& path)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

bool ParseUint(const std::string& value, uint32_t& out)
{
    if (value.empty()) {
        return false;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    out = static_cast<uint32_t>(parsed);
    return true;
}

std::string JsonStringField(const std::string& object, const std::string& field)
{
    const std::regex pattern("\"" + field + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    return std::regex_search(object, match, pattern) ? match[1].str() : std::string();
}

bool JsonUintField(const std::string& object, const std::string& field, uint32_t& out)
{
    const std::regex quoted("\"" + field + "\"\\s*:\\s*\"([0-9]+)\"");
    const std::regex plain("\"" + field + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (std::regex_search(object, match, quoted) || std::regex_search(object, match, plain)) {
        return ParseUint(match[1].str(), out);
    }
    return false;
}

bool ParseEidHex(const std::string& text, HccpEid& eid)
{
    if (text.size() != sizeof(eid.raw) * 2) {
        return false;
    }
    for (size_t i = 0; i < sizeof(eid.raw); ++i) {
        const char hi = text[i * 2];
        const char lo = text[i * 2 + 1];
        if (!std::isxdigit(static_cast<unsigned char>(hi)) || !std::isxdigit(static_cast<unsigned char>(lo))) {
            return false;
        }
        eid.raw[i] = static_cast<uint8_t>(std::strtoul(text.substr(i * 2, 2).c_str(), nullptr, 16));
    }
    return true;
}

std::vector<std::string> JsonStringArrayField(const std::string& object, const std::string& field)
{
    const std::regex arrayPattern("\"" + field + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch arrayMatch;
    if (!std::regex_search(object, arrayMatch, arrayPattern)) {
        return {};
    }
    const std::string body = arrayMatch[1].str();
    std::vector<std::string> values;
    const std::regex valuePattern("\"([^\"]*)\"");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), valuePattern);
         it != std::sregex_iterator(); ++it) {
        values.push_back((*it)[1].str());
    }
    return values;
}

std::vector<std::string> ExtractObjectsWithKey(const std::string& text, const std::string& key)
{
    std::vector<std::string> objects;
    const std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        const size_t begin = text.rfind('{', pos);
        if (begin == std::string::npos) {
            ++pos;
            continue;
        }
        int depth = 0;
        bool inString = false;
        bool escaped = false;
        for (size_t i = begin; i < text.size(); ++i) {
            const char ch = text[i];
            if (inString) {
                escaped = (!escaped && ch == '\\');
                if (ch == '"' && !escaped) {
                    inString = false;
                } else if (ch != '\\') {
                    escaped = false;
                }
                continue;
            }
            if (ch == '"') {
                inString = true;
            } else if (ch == '{') {
                ++depth;
            } else if (ch == '}') {
                --depth;
                if (depth == 0) {
                    objects.emplace_back(text.substr(begin, i - begin + 1));
                    pos = i + 1;
                    break;
                }
            }
        }
        if (depth != 0) {
            break;
        }
    }
    return objects;
}

bool ParseRootInfo(TileXRRootInfo& root)
{
    const std::string content = ReadTextFile("/etc/hccl_rootinfo.json");
    if (content.empty()) {
        return false;
    }
    root.topoPath = JsonStringField(content, "topo_file_path");
    if (root.topoPath.empty()) {
        return false;
    }

    for (const std::string& rankObj : ExtractObjectsWithKey(content, "device_id")) {
        uint32_t deviceId = 0;
        uint32_t localId = 0;
        if (!JsonUintField(rankObj, "device_id", deviceId) || !JsonUintField(rankObj, "local_id", localId)) {
            continue;
        }
        if (root.deviceToLocalId.empty()) {
            root.deviceIdOffset = deviceId;
        }
        root.deviceToLocalId[deviceId] = localId;

        uint32_t eidIndex = 0;
        for (const std::string& addrObj : ExtractObjectsWithKey(rankObj, "addr")) {
            HccpEid eid {};
            const std::string addr = JsonStringField(addrObj, "addr");
            if (!addr.empty() && ParseEidHex(addr, eid)) {
                root.eidByLocalId[localId][eidIndex] = eid;
            }
            for (const std::string& port : JsonStringArrayField(addrObj, "ports")) {
                root.portToEidByLocalId[localId][port] = eidIndex;
            }
            ++eidIndex;
        }
        if (root.eidCount == 0 && eidIndex != 0) {
            root.eidCount = eidIndex;
        }
    }
    return !root.deviceToLocalId.empty() && root.eidCount != 0;
}

std::vector<TileXRTopoEdge> ParseTopoInfo(const std::string& path)
{
    const std::string content = ReadTextFile(path);
    std::vector<TileXRTopoEdge> edges;
    if (content.empty()) {
        return edges;
    }
    for (const std::string& edgeObj : ExtractObjectsWithKey(content, "local_a")) {
        TileXRTopoEdge edge {};
        if (!JsonUintField(edgeObj, "local_a", edge.localA) || !JsonUintField(edgeObj, "local_b", edge.localB)) {
            continue;
        }
        edge.localAPorts = JsonStringArrayField(edgeObj, "local_a_ports");
        edge.localBPorts = JsonStringArrayField(edgeObj, "local_b_ports");
        if (!edge.localAPorts.empty() && !edge.localBPorts.empty()) {
            edges.push_back(edge);
        }
    }
    return edges;
}

bool ResolveLocalIdWithOffset(const TileXRRootInfo& root, uint32_t deviceId, uint32_t& localId)
{
    auto it = root.deviceToLocalId.find(deviceId + root.deviceIdOffset);
    if (it != root.deviceToLocalId.end()) {
        localId = it->second;
        return true;
    }
    it = root.deviceToLocalId.find(deviceId);
    if (it != root.deviceToLocalId.end()) {
        localId = it->second;
        return true;
    }
    return false;
}

bool ResolveLocalEidRoute(
    const TileXRRootInfo& root, const std::vector<TileXRTopoEdge>& edges, uint32_t localId, uint32_t peerLocalId,
    uint32_t& eidIndex)
{
    std::string localPort;
    for (const auto& edge : edges) {
        if (edge.localA == localId && edge.localB == peerLocalId) {
            localPort = edge.localAPorts[0];
            break;
        }
        if (edge.localB == localId && edge.localA == peerLocalId) {
            localPort = edge.localBPorts[0];
            break;
        }
    }
    const auto localIt = root.portToEidByLocalId.find(localId);
    if (localPort.empty() || localIt == root.portToEidByLocalId.end()) {
        return false;
    }
    const auto portIt = localIt->second.find(localPort);
    if (portIt == localIt->second.end()) {
        return false;
    }
    eidIndex = portIt->second;
    return root.eidByLocalId.count(localId) != 0 && root.eidByLocalId.at(localId).count(eidIndex) != 0;
}

} // namespace

struct TileXRUDMATransport::PerEidState {
    uint32_t eidIndex = 0;
    void* ctxHandle = nullptr;
    void* tokenHandle = nullptr;
    void* chanHandle = nullptr;
    void* cqHandle = nullptr;
    void* qpHandle = nullptr;
    CqInfoT cqInfo {};
    QpCreateInfo qpInfo {};
    std::vector<void*> remoteQpHandles;
    std::vector<uint32_t> tpnList;
    void* cqPiAddr = nullptr;
    void* cqCiAddr = nullptr;
    void* sqPiAddr = nullptr;
    void* sqCiAddr = nullptr;
    void* wqeCntAddr = nullptr;
    void* amoAddr = nullptr;
    UDMAWQCtx localWq {};
    UDMACQCtx localCq {};
};

TileXRUDMATransport::TileXRUDMATransport() = default;

TileXRUDMATransport::~TileXRUDMATransport()
{
    Shutdown();
}

int TileXRUDMATransport::Init(const TileXRUDMATransportOptions& options)
{
    if (available_) {
        return TILEXR_SUCCESS;
    }
    if (options.rankSize <= 1) {
        return TILEXR_SUCCESS;
    }
    if (options.rank < 0 || options.rank >= options.rankSize || options.exchange == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    options_ = options;

    int ret = loader_.Load();
    if (ret != TILEXR_HCCP_LOADER_SUCCESS) {
        return TILEXR_ERROR_NOT_FOUND;
    }
    ret = OpenDevice();
    if (ret != TILEXR_SUCCESS) {
        Shutdown();
        return ret;
    }
    ret = BuildRoutes();
    if (ret != TILEXR_SUCCESS) {
        Shutdown();
        return ret;
    }
    ret = CreateContexts();
    if (ret != TILEXR_SUCCESS) {
        Shutdown();
        return ret;
    }
    ret = CreateQueues();
    if (ret != TILEXR_SUCCESS) {
        Shutdown();
        return ret;
    }
    ret = RefreshUDMAInfo();
    if (ret != TILEXR_SUCCESS) {
        Shutdown();
        return ret;
    }

    available_ = true;
    return TILEXR_SUCCESS;
}

int TileXRUDMATransport::OpenDevice()
{
    logicDevId_ = static_cast<uint32_t>(options_.devId);
    TileXRRootInfo rootInfo {};
    if (ParseRootInfo(rootInfo)) {
        deviceIdOffset_ = rootInfo.deviceIdOffset;
    }

    ProcOpenArgs args {};
    args.procType = TSD_SUB_PROC_HCCP;
    char paramInfo[] = "--hdcType=18";
    ProcExtParam extParam {};
    extParam.paramInfo = paramInfo;
    extParam.paramLen = sizeof(paramInfo);
    args.extParamList = &extParam;
    args.extParamCnt = 1UL;
    args.subPid = &subPid_;
    auto tsdRet = loader_.TsdProcessOpen(logicDevId_, &args);
    if (tsdRet != 0) {
        TILEXR_LOG(WARN) << "TileXR UDMA TsdProcessOpen failed: " << tsdRet;
        return TILEXR_ERROR_INTERNAL;
    }
    tsdOpened_ = true;

    RaInitConfig initConfig {};
    initConfig.phyId = logicDevId_ + deviceIdOffset_;
    initConfig.nicPosition = NETWORK_OFFLINE;
    initConfig.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
    initConfig.enableHdcAsync = 1;
    int ret = loader_.RaInit(&initConfig);
    if (ret != 0) {
        TILEXR_LOG(WARN) << "TileXR UDMA RaInit failed: " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    raInitialized_ = true;
    return TILEXR_SUCCESS;
}

int TileXRUDMATransport::BuildRoutes()
{
    RaInfo info {};
    info.phyId = logicDevId_ + deviceIdOffset_;
    info.mode = NETWORK_OFFLINE;
    unsigned int eidNum = 0;
    int ret = loader_.RaGetDevEidInfoNum(info, &eidNum);
    if (ret != 0 || eidNum == 0) {
        TILEXR_LOG(WARN) << "TileXR UDMA RaGetDevEidInfoNum failed: " << ret << ", eidNum=" << eidNum;
        return TILEXR_ERROR_INTERNAL;
    }
    std::vector<DevEidInfo> devEids(eidNum);
    ret = loader_.RaGetDevEidInfoList(info, devEids.data(), &eidNum);
    if (ret != 0 || eidNum == 0) {
        TILEXR_LOG(WARN) << "TileXR UDMA RaGetDevEidInfoList failed: " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    eidCount_ = eidNum;

    uint32_t localId = static_cast<uint32_t>(options_.devId);
    const bool useTopoRoutes = std::getenv("TILEXR_UDMA_TOPO_ROUTE") != nullptr &&
        std::strcmp(std::getenv("TILEXR_UDMA_TOPO_ROUTE"), "0") != 0;
    bool topoReady = false;
    TileXRRootInfo rootInfo {};
    std::vector<TileXRTopoEdge> topoEdges;
    if (useTopoRoutes && ParseRootInfo(rootInfo)) {
        if (rootInfo.eidCount > eidCount_) {
            eidCount_ = rootInfo.eidCount;
        }
        topoEdges = ParseTopoInfo(rootInfo.topoPath);
        topoReady = ResolveLocalIdWithOffset(rootInfo, static_cast<uint32_t>(options_.devId), localId) &&
            !topoEdges.empty();
    }
    if (topoReady) {
        const auto localEids = rootInfo.eidByLocalId.find(localId);
        if (localEids != rootInfo.eidByLocalId.end()) {
            localEidByEid_ = localEids->second;
        } else {
            topoReady = false;
        }
    }
    if (!topoReady) {
        for (const auto& eid : devEids) {
            localEidByEid_[eid.eidIndex] = eid.eid;
        }
    }

    std::vector<uint32_t> allLocalIds(options_.rankSize);
    ret = options_.exchange->AllGather(&localId, 1, allLocalIds.data());
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }

    std::vector<int32_t> localRouteByPeer(options_.rankSize, -1);
    for (int peer = 0; peer < options_.rankSize; ++peer) {
        if (peer == options_.rank) {
            continue;
        }
        uint32_t localEid = devEids[0].eidIndex;
        if (topoReady && !ResolveLocalEidRoute(rootInfo, topoEdges, localId, allLocalIds[peer], localEid)) {
            topoReady = false;
            TILEXR_LOG(WARN) << "TileXR UDMA topology route resolution failed, falling back to EID "
                          << devEids[0].eidIndex;
            localEid = devEids[0].eidIndex;
        }
        peerLocalEid_[peer] = localEid;
        localRouteByPeer[peer] = static_cast<int32_t>(localEid);
    }

    std::vector<int32_t> allRouteByPeer(options_.rankSize * options_.rankSize, -1);
    ret = options_.exchange->AllGather(localRouteByPeer.data(), localRouteByPeer.size(), allRouteByPeer.data());
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }

    for (int peer = 0; peer < options_.rankSize; ++peer) {
        if (peer == options_.rank) {
            continue;
        }
        int32_t remoteEid = allRouteByPeer[peer * options_.rankSize + options_.rank];
        if (remoteEid < 0 || static_cast<uint32_t>(remoteEid) >= eidCount_) {
            remoteEid = static_cast<int32_t>(devEids[0].eidIndex);
        }
        peerRemoteEid_[peer] = static_cast<uint32_t>(remoteEid);
    }
    return TILEXR_SUCCESS;
}

int TileXRUDMATransport::CreateContexts()
{
    for (const auto& route : peerLocalEid_) {
        const uint32_t eidIndex = route.second;
        if (ctxHandleByEid_.count(eidIndex) != 0) {
            continue;
        }

        RaInfo info {};
        info.phyId = logicDevId_ + deviceIdOffset_;
        info.mode = NETWORK_OFFLINE;
        unsigned int eidNum = 0;
        int ret = loader_.RaGetDevEidInfoNum(info, &eidNum);
        if (ret != 0 || eidNum == 0) {
            return TILEXR_ERROR_INTERNAL;
        }
        std::vector<DevEidInfo> infoList(eidNum);
        ret = loader_.RaGetDevEidInfoList(info, infoList.data(), &eidNum);
        if (ret != 0) {
            return TILEXR_ERROR_INTERNAL;
        }

        bool found = false;
        CtxInitAttr attr {};
        auto targetEidIt = localEidByEid_.find(eidIndex);
        for (unsigned int i = 0; i < eidNum; ++i) {
            bool matched = infoList[i].eidIndex == eidIndex;
            if (targetEidIt != localEidByEid_.end()) {
                matched = std::memcmp(infoList[i].eid.raw, targetEidIt->second.raw, sizeof(infoList[i].eid.raw)) == 0;
            }
            if (!matched) {
                continue;
            }
            attr.phyId = logicDevId_ + deviceIdOffset_;
            attr.ub.eid = infoList[i].eid;
            attr.ub.eidIndex = infoList[i].eidIndex;
            localEidByEid_[eidIndex] = infoList[i].eid;
            found = true;
            break;
        }
        if (!found) {
            return TILEXR_ERROR_INTERNAL;
        }

        CtxInitCfg cfg {};
        cfg.mode = NETWORK_OFFLINE;
        void* ctxHandle = nullptr;
        ret = loader_.RaCtxInit(&cfg, &attr, &ctxHandle);
        if (ret != 0 || ctxHandle == nullptr) {
            TILEXR_LOG(WARN) << "TileXR UDMA RaCtxInit failed: " << ret;
            return TILEXR_ERROR_INTERNAL;
        }
        void* tokenHandle = nullptr;
        HccpTokenId tokenId {};
        ret = loader_.RaCtxTokenIdAlloc(ctxHandle, &tokenId, &tokenHandle);
        if (ret != 0 || tokenHandle == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }
        ctxHandleByEid_[eidIndex] = ctxHandle;
        tokenHandleByEid_[eidIndex] = tokenHandle;
    }
    return ctxHandleByEid_.empty() ? TILEXR_ERROR_INTERNAL : TILEXR_SUCCESS;
}

int TileXRUDMATransport::AllocDeviceScalar(void** ptr, size_t bytes) const
{
    int ret = aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }
    ret = aclrtMemset(*ptr, bytes, 0, bytes);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*ptr);
        *ptr = nullptr;
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

void TileXRUDMATransport::FreeDeviceScalar(void*& ptr) const
{
    if (ptr != nullptr) {
        aclrtFree(ptr);
        ptr = nullptr;
    }
}

int TileXRUDMATransport::CreateQueues()
{
    for (const auto& ctxEntry : ctxHandleByEid_) {
        PerEidState state {};
        state.eidIndex = ctxEntry.first;
        state.ctxHandle = ctxEntry.second;
        state.tokenHandle = tokenHandleByEid_[ctxEntry.first];
        state.remoteQpHandles.assign(options_.rankSize, nullptr);
        state.tpnList.assign(options_.rankSize, 0);

        ChanInfoT chanInfo {};
        chanInfo.in.dataPlaneFlag.bs.poolCqCstm = 1;
        int ret = loader_.RaCtxChanCreate(state.ctxHandle, &chanInfo, &state.chanHandle);
        if (ret != 0) {
            return TILEXR_ERROR_INTERNAL;
        }

        state.cqInfo.in.chanHandle = state.chanHandle;
        state.cqInfo.in.depth = TILEXR_UDMA_CQ_DEPTH;
        state.cqInfo.in.ub.mode = JFC_MODE_USER_CTL_NORMAL;
        ret = loader_.RaCtxCqCreate(state.ctxHandle, &state.cqInfo, &state.cqHandle);
        if (ret != 0) {
            return TILEXR_ERROR_INTERNAL;
        }
        state.localCq.cqn = 0;
        state.localCq.bufAddr = state.cqInfo.out.bufAddr;
        state.localCq.baseBkShift = Log2Uint64(state.cqInfo.out.cqeSize);
        state.localCq.depth = state.cqInfo.in.depth;
        if (AllocDeviceScalar(&state.cqPiAddr, sizeof(uint32_t)) != TILEXR_SUCCESS ||
            AllocDeviceScalar(&state.cqCiAddr, sizeof(uint32_t)) != TILEXR_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }
        state.localCq.headAddr = reinterpret_cast<uintptr_t>(state.cqPiAddr);
        state.localCq.tailAddr = reinterpret_cast<uintptr_t>(state.cqCiAddr);
        state.localCq.dbMode = UDMADBMode::SW_DB;
        state.localCq.dbAddr = state.cqInfo.out.swdbAddr;

        QpCreateAttr qpAttr {};
        qpAttr.scqHandle = state.cqHandle;
        qpAttr.rcqHandle = state.cqHandle;
        qpAttr.srqHandle = state.cqHandle;
        qpAttr.sqDepth = TILEXR_UDMA_SQ_DEPTH;
        qpAttr.rqDepth = TILEXR_UDMA_RQ_DEPTH_DEFAULT;
        qpAttr.transportMode = CONN_RM;
        qpAttr.ub.mode = JETTY_MODE_USER_CTL_NORMAL;
        qpAttr.ub.flag.value = 1;
        qpAttr.ub.jfsFlag.value = 2;
        qpAttr.ub.tokenValue = TILEXR_UDMA_TOKEN_VALUE;
        qpAttr.ub.rnrRetry = 7;
        qpAttr.ub.extMode.piType = 0;
        qpAttr.ub.extMode.cstmFlag.bs.sqCstm = 0;
        qpAttr.ub.extMode.sqebbNum = TILEXR_UDMA_SQ_DEPTH;
        qpAttr.ub.tokenIdHandle = state.tokenHandle;
        ret = loader_.RaCtxQpCreate(state.ctxHandle, &qpAttr, &state.qpInfo, &state.qpHandle);
        if (ret != 0) {
            return TILEXR_ERROR_INTERNAL;
        }
        state.localWq.wqn = 0;
        state.localWq.bufAddr = state.qpInfo.ub.sqBuffVa;
        state.localWq.baseBkShift = Log2Uint64(state.qpInfo.ub.wqebbSize);
        state.localWq.depth = TILEXR_UDMA_SQ_BB_COUNT;
        if (AllocDeviceScalar(&state.sqPiAddr, sizeof(uint32_t)) != TILEXR_SUCCESS ||
            AllocDeviceScalar(&state.sqCiAddr, sizeof(uint32_t)) != TILEXR_SUCCESS ||
            AllocDeviceScalar(&state.wqeCntAddr, sizeof(uint32_t)) != TILEXR_SUCCESS ||
            AllocDeviceScalar(&state.amoAddr, sizeof(uint64_t)) != TILEXR_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }
        state.localWq.headAddr = reinterpret_cast<uintptr_t>(state.sqPiAddr);
        state.localWq.tailAddr = reinterpret_cast<uintptr_t>(state.sqCiAddr);
        state.localWq.dbMode = UDMADBMode::SW_DB;
        state.localWq.dbAddr = state.qpInfo.ub.dbAddr;
        state.localWq.wqeCntAddr = reinterpret_cast<uintptr_t>(state.wqeCntAddr);
        state.localWq.amoAddr = reinterpret_cast<uintptr_t>(state.amoAddr);
        states_[state.eidIndex] = state;
    }
    return states_.empty() ? TILEXR_ERROR_INTERNAL : TILEXR_SUCCESS;
}

int TileXRUDMATransport::ImportQueues()
{
    std::vector<QpImportInfoT> localImports(eidCount_);
    std::vector<QpKeyT> localKeys(eidCount_);
    for (const auto& stateEntry : states_) {
        const auto& state = stateEntry.second;
        if (state.eidIndex >= eidCount_) {
            return TILEXR_ERROR_INTERNAL;
        }
        localImports[state.eidIndex].in.ub.mode = JETTY_IMPORT_MODE_NORMAL;
        localImports[state.eidIndex].in.ub.tokenValue = TILEXR_UDMA_TOKEN_VALUE;
        localImports[state.eidIndex].in.ub.policy = JETTY_GRP_POLICY_RR;
        localImports[state.eidIndex].in.ub.type = TARGET_TYPE_JETTY;
        localImports[state.eidIndex].in.ub.flag.bs.tokenPolicy = TOKEN_POLICY_PLAIN_TEXT;
        localImports[state.eidIndex].in.ub.tpType = 1;
        localKeys[state.eidIndex] = state.qpInfo.key;
    }

    std::vector<QpImportInfoT> allImports(options_.rankSize * eidCount_);
    int ret = options_.exchange->AllGather(localImports.data(), localImports.size(), allImports.data());
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    std::vector<QpKeyT> allKeys(options_.rankSize * eidCount_);
    ret = options_.exchange->AllGather(localKeys.data(), localKeys.size(), allKeys.data());
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }

    for (auto& stateEntry : states_) {
        auto& state = stateEntry.second;
        for (int peer = 0; peer < options_.rankSize; ++peer) {
            if (peer == options_.rank) {
                continue;
            }
            const auto localRoute = peerLocalEid_.find(peer);
            if (localRoute == peerLocalEid_.end() || localRoute->second != state.eidIndex) {
                continue;
            }
            const uint32_t remoteEid = peerRemoteEid_[peer];
            if (remoteEid >= eidCount_) {
                return TILEXR_ERROR_INTERNAL;
            }
            QpImportInfoT importInfo = allImports[peer * eidCount_ + remoteEid];
            importInfo.in.key = allKeys[peer * eidCount_ + remoteEid];
            ret = loader_.RaCtxQpImport(state.ctxHandle, &importInfo, &state.remoteQpHandles[peer]);
            if (ret != 0) {
                return TILEXR_ERROR_INTERNAL;
            }
            state.tpnList[peer] = importInfo.out.ub.tpn;
        }
    }
    return TILEXR_SUCCESS;
}

uint32_t TileXRUDMATransport::FallbackLocalEid() const
{
    if (!states_.empty()) {
        return states_.begin()->first;
    }
    return 0;
}

int TileXRUDMATransport::RefreshUDMAInfo()
{
    if (eidCount_ == 0 || states_.empty()) {
        return TILEXR_ERROR_INTERNAL;
    }
    if (eidTableDev_ == nullptr) {
        int ret = aclrtMalloc(reinterpret_cast<void**>(&eidTableDev_),
            options_.rankSize * eidCount_ * sizeof(HccpEid), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }
    }

    std::vector<UDMAMemInfo> localMemByEid(eidCount_);
    for (const auto& memEntry : localMemInfoByEid_) {
        if (memEntry.first < eidCount_) {
            localMemByEid[memEntry.first] = memEntry.second;
        }
    }
    std::vector<HccpEid> localEids(eidCount_);
    for (const auto& eidEntry : localEidByEid_) {
        if (eidEntry.first < eidCount_) {
            localEids[eidEntry.first] = SwapEidForDevice(eidEntry.second);
        }
    }

    std::vector<UDMAMemInfo> allMem(options_.rankSize * eidCount_);
    int ret = options_.exchange->AllGather(localMemByEid.data(), localMemByEid.size(), allMem.data());
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    std::vector<HccpEid> allEids(options_.rankSize * eidCount_);
    ret = options_.exchange->AllGather(localEids.data(), localEids.size(), allEids.data());
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(eidTableDev_, options_.rankSize * eidCount_ * sizeof(HccpEid),
        allEids.data(), options_.rankSize * eidCount_ * sizeof(HccpEid), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }

    const uint32_t fallbackEid = FallbackLocalEid();
    auto fallbackIt = states_.find(fallbackEid);
    if (fallbackIt == states_.end()) {
        return TILEXR_ERROR_INTERNAL;
    }

    std::vector<UDMAWQCtx> sq(options_.rankSize);
    std::vector<UDMAWQCtx> rq(options_.rankSize);
    std::vector<UDMACQCtx> scq(options_.rankSize);
    std::vector<UDMACQCtx> rcq(options_.rankSize);
    std::vector<UDMAMemInfo> mem(options_.rankSize);

    for (int rank = 0; rank < options_.rankSize; ++rank) {
        uint32_t localEid = fallbackEid;
        uint32_t remoteEid = fallbackEid;
        if (rank != options_.rank) {
            localEid = peerLocalEid_[rank];
            remoteEid = peerRemoteEid_[rank];
        }
        auto stateIt = states_.find(localEid);
        if (stateIt == states_.end()) {
            stateIt = fallbackIt;
        }
        const auto& state = stateIt->second;
        sq[rank] = state.localWq;
        rq[rank] = state.localWq;
        scq[rank] = state.localCq;
        rcq[rank] = state.localCq;
        if (rank == options_.rank) {
            const auto localMemIt = localMemInfoByEid_.find(localEid);
            if (localMemIt != localMemInfoByEid_.end()) {
                mem[rank] = localMemIt->second;
            }
        } else {
            mem[rank] = allMem[rank * eidCount_ + remoteEid];
            mem[rank].tpn = state.tpnList[rank];
        }
        mem[rank].eidAddr = reinterpret_cast<uint64_t>(
            eidTableDev_ + (rank * eidCount_ + remoteEid) * sizeof(HccpEid));
    }

    if (udmaInfoDev_ == nullptr) {
        const size_t oneRankSize = 2 * sizeof(UDMAWQCtx) + 2 * sizeof(UDMACQCtx) + sizeof(UDMAMemInfo);
        udmaInfoSize_ = static_cast<uint32_t>(sizeof(UDMAInfo) + oneRankSize * options_.rankSize);
        ret = aclrtMalloc(reinterpret_cast<void**>(&udmaInfoDev_), udmaInfoSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }
    }

    UDMAInfo info {};
    std::vector<uint8_t> image;
    ret = BuildUDMAInfoImage(reinterpret_cast<uintptr_t>(udmaInfoDev_), sq, rq, scq, rcq, mem, info, image);
    if (ret != TILEXR_UDMA_LAYOUT_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    ret = aclrtMemcpy(udmaInfoDev_, udmaInfoSize_, image.data(), image.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRUDMATransport::RegisterMemory(GM_ADDR localPtr, size_t bytes)
{
    if (!available_ || localPtr == nullptr || bytes == 0) {
        return TILEXR_ERROR_NOT_FOUND;
    }
    int ret = RegisterMemoryOnContexts(localPtr, bytes);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    registeredPtr_ = localPtr;
    if (!queuesImported_) {
        ret = ImportQueues();
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
        queuesImported_ = true;
    }
    ret = ExchangeAndImportMemory();
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    return RefreshUDMAInfo();
}

int TileXRUDMATransport::RegisterMemoryOnContexts(GM_ADDR localPtr, size_t bytes)
{
    if (registeredPtr_ != nullptr) {
        UnregisterMemory(registeredPtr_);
    }
    std::map<uint32_t, RegMemResultInfo> byEid;
    localMemInfoByEid_.clear();
    for (const auto& ctxEntry : ctxHandleByEid_) {
        const uint32_t eidIndex = ctxEntry.first;
        void* tokenHandle = tokenHandleByEid_[eidIndex];
        MrRegInfoT mrInfo {};
        mrInfo.in.mem.addr = reinterpret_cast<uint64_t>(localPtr);
        mrInfo.in.mem.size = bytes;
        mrInfo.in.ub.tokenValue = TILEXR_UDMA_TOKEN_VALUE;
        mrInfo.in.ub.tokenIdHandle = tokenHandle;
        mrInfo.in.ub.flags.bs.access = MEM_SEG_ACCESS_DEFAULT;
        mrInfo.in.ub.flags.bs.cacheable = 0;
        mrInfo.in.ub.flags.bs.tokenIdValid = 1;
        mrInfo.in.ub.flags.bs.nonPin = 0;
        mrInfo.in.ub.flags.bs.userIova = 0;
        mrInfo.in.ub.flags.bs.tokenPolicy = MEM_SEG_TOKEN_PLAIN_TEXT;
        void* lmemHandle = nullptr;
        int ret = loader_.RaCtxLmemRegister(ctxEntry.second, &mrInfo, &lmemHandle);
        if (ret != 0 || lmemHandle == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }

        RegMemResultInfo result {};
        result.address = reinterpret_cast<uint64_t>(localPtr);
        result.size = bytes;
        result.lmemHandle = lmemHandle;
        result.key = mrInfo.out.key;
        result.tokenId = mrInfo.out.ub.tokenId;
        result.tokenValue = TILEXR_UDMA_TOKEN_VALUE;
        result.targetSegHandle = mrInfo.out.ub.targetSegHandle;
        result.tokenIdHandle = tokenHandle;
        result.access = MEM_SEG_ACCESS_DEFAULT;
        byEid[eidIndex] = result;

        UDMAMemInfo memInfo {};
        memInfo.tokenValueValid = true;
        memInfo.rmtJettyType = 1;
        memInfo.targetHint = 0;
        memInfo.tpn = 0;
        memInfo.tid = mrInfo.out.ub.tokenId >> 8;
        memInfo.rmtTokenValue = TILEXR_UDMA_TOKEN_VALUE;
        memInfo.len = static_cast<uint32_t>(std::min<size_t>(bytes, UINT32_MAX));
        memInfo.addr = reinterpret_cast<uint64_t>(localPtr);
        localMemInfoByEid_[eidIndex] = memInfo;
    }
    registeredMem_[reinterpret_cast<uint64_t>(localPtr)] = byEid;
    return TILEXR_SUCCESS;
}

int TileXRUDMATransport::ExchangeAndImportMemory()
{
    if (registeredMem_.empty()) {
        return TILEXR_ERROR_NOT_FOUND;
    }
    const auto& localByEid = registeredMem_.begin()->second;
    uint32_t localCount = static_cast<uint32_t>(localByEid.size());
    std::vector<uint32_t> allCounts(options_.rankSize);
    int ret = options_.exchange->AllGather(&localCount, 1, allCounts.data());
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    const uint32_t maxCount = *std::max_element(allCounts.begin(), allCounts.end());
    if (maxCount == 0) {
        return TILEXR_ERROR_INTERNAL;
    }

    struct ExchangedMrInfo {
        uint32_t eidIndex;
        uint32_t valid;
        RegMemResultInfo mr;
    };

    std::vector<ExchangedMrInfo> local(maxCount);
    uint32_t idx = 0;
    for (const auto& entry : localByEid) {
        local[idx].eidIndex = entry.first;
        local[idx].valid = 1;
        local[idx].mr = entry.second;
        ++idx;
    }
    std::vector<ExchangedMrInfo> all(options_.rankSize * maxCount);
    ret = options_.exchange->AllGather(local.data(), local.size(), all.data());
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }

    remoteMemHandles_.assign(options_.rankSize, nullptr);
    for (int peer = 0; peer < options_.rankSize; ++peer) {
        if (peer == options_.rank) {
            continue;
        }
        const uint32_t remoteEid = peerRemoteEid_[peer];
        const ExchangedMrInfo* remote = nullptr;
        for (uint32_t i = 0; i < allCounts[peer]; ++i) {
            const auto& candidate = all[peer * maxCount + i];
            if (candidate.valid != 0 && candidate.eidIndex == remoteEid) {
                remote = &candidate;
                break;
            }
        }
        if (remote == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }
        const uint32_t localEid = peerLocalEid_[peer];
        MrImportInfoT importInfo {};
        importInfo.in.key = remote->mr.key;
        importInfo.in.ub.tokenValue = remote->mr.tokenValue;
        importInfo.in.ub.flags.bs.cacheable = remote->mr.cacheable;
        importInfo.in.ub.flags.bs.access = remote->mr.access;
        void* remoteHandle = nullptr;
        ret = loader_.RaCtxRmemImport(ctxHandleByEid_[localEid], &importInfo, &remoteHandle);
        if (ret != 0 || remoteHandle == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }
        remoteMemHandles_[peer] = remoteHandle;
    }
    return TILEXR_SUCCESS;
}

int TileXRUDMATransport::UnregisterMemory(GM_ADDR localPtr)
{
    if (localPtr == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    CleanupMemory();
    registeredPtr_ = nullptr;
    localMemInfoByEid_.clear();
    if (available_) {
        return RefreshUDMAInfo();
    }
    return TILEXR_SUCCESS;
}

void TileXRUDMATransport::CleanupMemory()
{
    for (int peer = 0; peer < static_cast<int>(remoteMemHandles_.size()); ++peer) {
        if (peer == options_.rank || remoteMemHandles_[peer] == nullptr) {
            continue;
        }
        const uint32_t localEid = peerLocalEid_[peer];
        loader_.RaCtxRmemUnimport(ctxHandleByEid_[localEid], remoteMemHandles_[peer]);
        remoteMemHandles_[peer] = nullptr;
    }
    for (const auto& mrEntry : registeredMem_) {
        for (const auto& eidMr : mrEntry.second) {
            const uint32_t eidIndex = eidMr.first;
            if (ctxHandleByEid_.count(eidIndex) != 0 && eidMr.second.lmemHandle != nullptr) {
                loader_.RaCtxLmemUnregister(ctxHandleByEid_[eidIndex], eidMr.second.lmemHandle);
            }
        }
    }
    registeredMem_.clear();
}

void TileXRUDMATransport::CleanupQueues()
{
    for (auto& stateEntry : states_) {
        auto& state = stateEntry.second;
        for (void* remoteQp : state.remoteQpHandles) {
            if (remoteQp != nullptr && state.ctxHandle != nullptr) {
                loader_.RaCtxQpUnimport(state.ctxHandle, remoteQp);
            }
        }
        if (state.qpHandle != nullptr) {
            loader_.RaCtxQpDestroy(state.qpHandle);
        }
        if (state.cqHandle != nullptr && state.ctxHandle != nullptr) {
            loader_.RaCtxCqDestroy(state.ctxHandle, state.cqHandle);
        }
        if (state.chanHandle != nullptr && state.ctxHandle != nullptr) {
            loader_.RaCtxChanDestroy(state.ctxHandle, state.chanHandle);
        }
        FreeDeviceScalar(state.cqPiAddr);
        FreeDeviceScalar(state.cqCiAddr);
        FreeDeviceScalar(state.sqPiAddr);
        FreeDeviceScalar(state.sqCiAddr);
        FreeDeviceScalar(state.wqeCntAddr);
        FreeDeviceScalar(state.amoAddr);
    }
    states_.clear();
    queuesImported_ = false;
}

void TileXRUDMATransport::CleanupContexts()
{
    for (const auto& tokenEntry : tokenHandleByEid_) {
        const uint32_t eidIndex = tokenEntry.first;
        if (ctxHandleByEid_.count(eidIndex) != 0 && tokenEntry.second != nullptr) {
            loader_.RaCtxTokenIdFree(ctxHandleByEid_[eidIndex], tokenEntry.second);
        }
    }
    tokenHandleByEid_.clear();

    for (const auto& ctxEntry : ctxHandleByEid_) {
        if (ctxEntry.second != nullptr) {
            loader_.RaCtxDeinit(ctxEntry.second);
        }
    }
    ctxHandleByEid_.clear();

    if (raInitialized_) {
        RaInitConfig deinitConfig {};
        deinitConfig.phyId = logicDevId_ + deviceIdOffset_;
        deinitConfig.nicPosition = NETWORK_OFFLINE;
        deinitConfig.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
        deinitConfig.enableHdcAsync = 1;
        loader_.RaDeinit(&deinitConfig);
        raInitialized_ = false;
    }
    if (tsdOpened_) {
        loader_.TsdProcessClose(logicDevId_, subPid_);
        tsdOpened_ = false;
        subPid_ = 0;
    }
}

void TileXRUDMATransport::Shutdown()
{
    available_ = false;
    CleanupMemory();
    CleanupQueues();
    CleanupContexts();
    if (udmaInfoDev_ != nullptr) {
        aclrtFree(udmaInfoDev_);
        udmaInfoDev_ = nullptr;
    }
    if (eidTableDev_ != nullptr) {
        aclrtFree(eidTableDev_);
        eidTableDev_ = nullptr;
    }
    localEidByEid_.clear();
    peerLocalEid_.clear();
    peerRemoteEid_.clear();
    localMemInfoByEid_.clear();
    remoteMemHandles_.clear();
    loader_.Unload();
}

bool TileXRUDMATransport::IsAvailable() const
{
    return available_ && udmaInfoDev_ != nullptr;
}

GM_ADDR TileXRUDMATransport::GetUDMAInfoDev() const
{
    return udmaInfoDev_;
}

} // namespace TileXR
