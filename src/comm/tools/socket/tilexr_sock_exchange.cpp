/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tilexr_sock_exchange.h"

#include <unistd.h>
#include <sys/types.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <set>
#include <array>
#include <string>
#include <fstream>
#include <sstream>

#include <sys/socket.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <acl/acl.h>
#include <securec.h>

using namespace std;
namespace TileXR {
const string TILEXR_LOCAL_SOCK_IP = "127.0.0.1";
constexpr uint16_t TILEXR_DEFAULT_SOCK_PORT = 10067;
constexpr uint32_t TILEXR_MAX_BACK_LOG = 65535;

int ParseIpAndPort(const char* input, string &ip, uint16_t &port)
{
    if (input == nullptr) {
        return TILEXR_INVALID_VALUE;
    }
    string inputStr(input);
    size_t colonPos = inputStr.find(':');
    if (colonPos == string::npos) {
        TILEXR_LOG(ERROR) << "Input string does not contain a colon separating IP and port.";
        return TILEXR_ERROR_INTERNAL;
    }

    ip = inputStr.substr(0, colonPos);
    std::string portStr = inputStr.substr(colonPos + 1);

    std::istringstream portStream(portStr);
    portStream >> port;
    if (portStream.fail() || portStream.bad()) {
        TILEXR_LOG(ERROR) << "Invalid port number.";
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}


TileXRSockExchange::~TileXRSockExchange()
{
    Cleanup();
}

TileXRSockExchange::TileXRSockExchange(int rank, int rankSize, int commDomain)
    : rank_(rank), rankSize_(rankSize), commDomain_(commDomain)
{
}

TileXRSockExchange::TileXRSockExchange(int rank, int rankSize, TileXRUniqueId tilexrCommId)
    : rank_(rank), rankSize_(rankSize)
{
    tilexrCommId_.uid = tilexrCommId;
    char ip[INET_ADDRSTRLEN] = {};
    inet_ntop(AF_INET, &(tilexrCommId_.handle.addr.sin.sin_addr), ip, INET_ADDRSTRLEN);
    ip_ = ip;
    port_ = ntohs(tilexrCommId_.handle.addr.sin.sin_port);
    TILEXR_LOG(INFO) << "TileXRSockExchange using UniqueId mode "
        << ip_ << ":" << port_ << " ";
}

string GetUUID()
{
    const string filePath = "/proc/sys/kernel/random/boot_id";
    ifstream fileStream(filePath);
    stringstream buffer;
    if (fileStream) {
        buffer << fileStream.rdbuf();
        fileStream.close();
    }
    const string uuid = buffer.str();
    return uuid;
}

int TileXRSockExchange::GetNodeNum()
{
    constexpr size_t uuidBytes = 64;
    std::array<char, uuidBytes> localUuid {};
    const string uuid = GetUUID();
    std::copy_n(uuid.data(), std::min(uuid.size(), uuidBytes - 1), localUuid.data());
    std::vector<char> allUuids(static_cast<size_t>(rankSize_) * uuidBytes);
    if (AllGather(localUuid.data(), uuidBytes, allUuids.data()) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }
    set<string> uuidSet {};
    for (int rank = 0; rank < rankSize_; ++rank) {
        uuidSet.emplace(allUuids.data() + static_cast<size_t>(rank) * uuidBytes);
    }
    TILEXR_LOG(DEBUG) << "nodeNum:" << uuidSet.size();
    return static_cast<int>(uuidSet.size());
}

void TileXRSockExchange::GetIpAndPort()
{
    const char* env = std::getenv("TILEXR_COMM_ID");

    if (env == nullptr or ParseIpAndPort(env, ip_, port_) != TILEXR_SUCCESS) {
        ip_ = TILEXR_LOCAL_SOCK_IP;
        port_ = TILEXR_DEFAULT_SOCK_PORT;
    }
    port_ += commDomain_;
    tilexrCommId_.handle.addr.sin.sin_family = AF_INET;
    tilexrCommId_.handle.addr.sin.sin_addr.s_addr = inet_addr(ip_.c_str());
    tilexrCommId_.handle.addr.sin.sin_port = htons(port_);
    TILEXR_LOG(DEBUG) << "curRank: " << rank_ << " commDomain: " << commDomain_ << " ip: " << ip_ << " port: " << port_;
}

int TileXRSockExchange::ConfigureHierarchy()
{
    hierarchical_ = false;
    const char* enabled = std::getenv("TILEXR_SOCKET_HIERARCHICAL");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
        return TILEXR_SUCCESS;
    }
    const char* groupSizeEnv = std::getenv("TILEXR_SOCKET_GROUP_SIZE");
    const char* ranksPerHostEnv = std::getenv("TILEXR_SOCKET_RANKS_PER_HOST");
    hierarchyGroupSize_ = groupSizeEnv == nullptr ? 64 : std::atoi(groupSizeEnv);
    hierarchyRanksPerHost_ = ranksPerHostEnv == nullptr ? 8 : std::atoi(ranksPerHostEnv);
    if (hierarchyGroupSize_ <= 0 || hierarchyRanksPerHost_ <= 0 ||
        hierarchyGroupSize_ % hierarchyRanksPerHost_ != 0) {
        TILEXR_LOG(ERROR) << "Invalid hierarchical socket group configuration";
        return TILEXR_ERROR_INTERNAL;
    }

    const char* hostsEnv = std::getenv("TILEXR_COMM_HOSTS");
    if (hostsEnv == nullptr || hostsEnv[0] == '\0') {
        TILEXR_LOG(ERROR) << "TILEXR_COMM_HOSTS is required for hierarchical socket exchange";
        return TILEXR_ERROR_INTERNAL;
    }
    hierarchyHosts_.clear();
    std::stringstream hostStream(hostsEnv);
    string host;
    while (std::getline(hostStream, host, ',')) {
        if (!host.empty()) {
            hierarchyHosts_.push_back(host);
        }
    }
    const size_t requiredHosts =
        (static_cast<size_t>(rankSize_) + hierarchyRanksPerHost_ - 1) /
        static_cast<size_t>(hierarchyRanksPerHost_);
    if (hierarchyHosts_.size() < requiredHosts) {
        TILEXR_LOG(ERROR) << "Hierarchical socket host list too short: "
                          << hierarchyHosts_.size() << " < " << requiredHosts;
        return TILEXR_ERROR_INTERNAL;
    }

    groupLayout_ = BuildSockExchangeGroupLayout(rank_, rankSize_, hierarchyGroupSize_);
    if (groupLayout_.groupCount == 0) {
        return TILEXR_ERROR_INTERNAL;
    }
    const size_t leaderHost = static_cast<size_t>(groupLayout_.groupLeader) /
        static_cast<size_t>(hierarchyRanksPerHost_);
    listenAddr_ = tilexrCommId_.handle.addr.sin;
    listenAddr_.sin_addr.s_addr = inet_addr(hierarchyHosts_[leaderHost].c_str());
    if (listenAddr_.sin_addr.s_addr == INADDR_NONE) {
        TILEXR_LOG(ERROR) << "Invalid hierarchical leader host " << hierarchyHosts_[leaderHost];
        return TILEXR_ERROR_INTERNAL;
    }
    hierarchical_ = groupLayout_.groupCount > 1;
    TILEXR_LOG(INFO) << "Hierarchical socket exchange rank=" << rank_
                     << " group=" << groupLayout_.groupIndex
                     << " leader=" << groupLayout_.groupLeader
                     << " range=[" << groupLayout_.groupBegin << ","
                     << groupLayout_.groupEnd << ") groups=" << groupLayout_.groupCount;
    return TILEXR_SUCCESS;
}

int TileXRSockExchange::Prepare()
{
    if (tilexrCommId_.handle.magic != TILEXR_MAGIC) {
        GetIpAndPort();
    }
    if (ConfigureHierarchy() != TILEXR_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }
    clientFds_.resize(rankSize_, -1);
    if (hierarchical_) {
        if (!IsGroupLeader()) {
            sockaddr_in leaderAddr = tilexrCommId_.handle.addr.sin;
            const size_t leaderHost = static_cast<size_t>(groupLayout_.groupLeader) /
                static_cast<size_t>(hierarchyRanksPerHost_);
            leaderAddr.sin_addr.s_addr = inet_addr(hierarchyHosts_[leaderHost].c_str());
            return ConnectTo(leaderAddr);
        }
        if (Listen() != TILEXR_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }
        if (rank_ != 0 && Connect() != TILEXR_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }
        return AcceptHierarchical();
    }
    if (!IsServer()) {
        return Connect();
    }
    if (Listen() != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "Listen Failed!";
        return TILEXR_ERROR_INTERNAL;
    }

    if (Accept() != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "Accept Failed!";
        return TILEXR_ERROR_INTERNAL;
    }

    return TILEXR_SUCCESS;
}

int TileXRSockExchange::Listen()
{
    listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) {
        TILEXR_LOG(ERROR) << "Server side create socket failed";
        return TILEXR_ERROR_INTERNAL;
    }

    int reuse = 1;
    if (setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(int)) < 0) {
        TILEXR_LOG(ERROR) << "Server side set reuseaddr failed";
        return TILEXR_ERROR_INTERNAL;
    }

    sockaddr_in bindAddr = hierarchical_ ? listenAddr_ : tilexrCommId_.handle.addr.sin;
    struct sockaddr *addrPtr = reinterpret_cast<struct sockaddr*>(&bindAddr);
    if (bind(listenFd_, addrPtr, sizeof(struct sockaddr)) < 0) {
        TILEXR_LOG(ERROR) << "Server side bind " << ntohs(tilexrCommId_.handle.addr.sin.sin_port) << " failed";
        return TILEXR_ERROR_INTERNAL;
    }

    /*
     * kernel would silently truncate backlog to the value defined in
     * /proc/sys/net/core/somaxconn if it is less than 65535.
     */
    if (listen(listenFd_, TILEXR_MAX_BACK_LOG) < 0) {
        TILEXR_LOG(ERROR) << "Server side listen " << ntohs(tilexrCommId_.handle.addr.sin.sin_port) << " failed";
        return TILEXR_ERROR_INTERNAL;
    }
    TILEXR_LOG(INFO) << "The server is listening! ip: "<< inet_ntoa(bindAddr.sin_addr)
        << " port: " << ntohs(bindAddr.sin_port);

    return TILEXR_SUCCESS;
}

int TileXRSockExchange::AcceptConnection(int fd, sockaddr_in& clientAddr, socklen_t *sinSize) const
{
    int clientFd;
    TileXRSocketAddress clientAddrPtr;
    clientAddrPtr.sin = clientAddr;

    do {
        clientFd = accept(fd, &clientAddrPtr.sa, sinSize);
        if (clientFd < 0) {
            if (!CheckErrno(errno)) {
                TILEXR_LOG(ERROR) << "Server side accept failed" << strerror(errno);
                return -1;
            }
            TILEXR_LOG(DEBUG) << "accept failed: " << strerror(errno);
            continue;
        }
        break;
    } while (true);

    return clientFd;
}

int TileXRSockExchange::Accept()
{
    struct sockaddr_in clientAddr;
    socklen_t sinSize = sizeof(struct sockaddr_in);

    for (int i = 1; i < rankSize_; ++i) {
        int fd = AcceptConnection(listenFd_, clientAddr, &sinSize);
        if (fd < 0) {
            TILEXR_LOG(ERROR) << "AcceptConnection failed";
            return TILEXR_ERROR_INTERNAL;
        }

        int rank = 0;
        if (Recv(fd, &rank, sizeof(rank), 0) <= 0) {
            TILEXR_LOG(ERROR) << "Server side recv rank id failed";
            return TILEXR_ERROR_INTERNAL;
        }

        if (rank >= rankSize_ || rank <= 0 || clientFds_[rank] >= 0) {
            TILEXR_LOG(ERROR) << "Server side recv invalid rank id " << rank;
            return TILEXR_ERROR_INTERNAL;
        }

        TILEXR_LOG(DEBUG) << "Server side recv rank id " << rank;
        clientFds_[rank] = fd;
    }

    return TILEXR_SUCCESS;
}

int TileXRSockExchange::AcceptHierarchical()
{
    const int localMembers = groupLayout_.groupEnd - groupLayout_.groupBegin - 1;
    const int childLeaders = rank_ == 0 ? groupLayout_.groupCount - 1 : 0;
    const int expected = localMembers + childLeaders;
    struct sockaddr_in clientAddr {};
    socklen_t sinSize = sizeof(clientAddr);
    for (int accepted = 0; accepted < expected; ++accepted) {
        int clientFd = AcceptConnection(listenFd_, clientAddr, &sinSize);
        int peerRank = -1;
        if (clientFd < 0 || Recv(clientFd, &peerRank, sizeof(peerRank), 0) <= 0) {
            TILEXR_LOG(ERROR) << "Hierarchical accept rank handshake failed";
            return TILEXR_ERROR_INTERNAL;
        }
        const bool localMember = peerRank >= groupLayout_.groupBegin &&
            peerRank < groupLayout_.groupEnd && peerRank != rank_;
        const bool childLeader = rank_ == 0 && peerRank > 0 &&
            peerRank < rankSize_ && peerRank % hierarchyGroupSize_ == 0;
        if ((!localMember && !childLeader) || clientFds_[peerRank] >= 0) {
            TILEXR_LOG(ERROR) << "Hierarchical accept invalid peer rank " << peerRank;
            Close(clientFd);
            return TILEXR_ERROR_INTERNAL;
        }
        clientFds_[peerRank] = clientFd;
    }
    return TILEXR_SUCCESS;
}

void TileXRSockExchange::Close(int &fd) const
{
    if (fd == -1) {
        return;
    }

    if (close(fd) < 0) {
        TILEXR_LOG(WARN) << "failed to close fd:" << fd;
        return;
    }

    fd = -1;
}

int TileXRSockExchange::Connect()
{
    return ConnectTo(tilexrCommId_.handle.addr.sin);
}

int TileXRSockExchange::ConnectTo(const sockaddr_in &serverAddr)
{
    TILEXR_LOG(DEBUG) << "Client side " << rank_ << " begin to connect";

    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        TILEXR_LOG(ERROR) << "Client side " << rank_ << " create socket failed";
        return TILEXR_ERROR_INTERNAL;
    }

    int sleepTimeS = 1;
    int maxRetryCount = 180;
    int retryCount = 0;
    bool success = false;
    const struct sockaddr *addrPtr = reinterpret_cast<const struct sockaddr*>(&serverAddr);
    while (retryCount < maxRetryCount) {
        if (connect(fd_, addrPtr, sizeof(struct sockaddr)) < 0) {
            if (errno == ECONNREFUSED) {
                TILEXR_LOG(DEBUG) << "Client side " << rank_ << " try connect " << (retryCount + 1) << " times refused";
                retryCount++;
                sleep(sleepTimeS);
                continue;
            }
            if (errno != EINTR) {
                TILEXR_LOG(ERROR) << "Client side " << rank_ << " connect failed: " << strerror(errno);
                break;
            }
            TILEXR_LOG(DEBUG) << "Client side " << rank_ << " try connect failed: " << strerror(errno);
            continue;
        }
        success = true;
        break;
    }

    if (!success) {
        TILEXR_LOG(ERROR) << "Client side " << rank_ << " connect failed";
        return TILEXR_ERROR_INTERNAL;
    }

    if (Send(fd_, &rank_, sizeof(rank_), 0) <= 0) {
        TILEXR_LOG(ERROR) << "Client side " << rank_ << " send rank failed";
        return TILEXR_ERROR_INTERNAL;
    }

    return TILEXR_SUCCESS;
}

bool TileXRSockExchange::IsServer() const
{
    return rank_ == 0;
}

void TileXRSockExchange::Cleanup()
{
    if (fd_ >= 0) {
        Close(fd_);
        fd_ = -1;
    }
    if (listenFd_ >= 0) {
        Close(listenFd_);
    }

    if (clientFds_.empty()) {
        return;
    }

    for (int i = 1; i < rankSize_; ++i) {
        if (clientFds_[i] >= 0) {
            Close(clientFds_[i]);
        }
    }
}

int GetAddrFromString(TileXRSocketAddress* ua, const char* ipPortPair)
{
    std::string ip;
    uint16_t port;
    int ret = ParseIpAndPort(ipPortPair, ip, port);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr ParseIpAndPort failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    ua->sin.sin_family = AF_INET;
    ua->sin.sin_addr.s_addr = inet_addr(ip.c_str());
    ua->sin.sin_port = htons(port);
    return TILEXR_SUCCESS;
}

bool IsValidInterface(const std::string& interfaceName)
{
    // 判断字符串是否以 "lo"、"virbr" 或 "docker" 开头
    if (interfaceName.find("lo") == 0 ||
        interfaceName.find("virbr") == 0 ||
        interfaceName.find("docker") == 0) {
        return false; // 如果以这些前缀开头，返回 false
    }
    return true; // 否则返回 true
}

int BootstrapGetServerIp(TileXRSocketAddress& handle)
{
    struct ifaddrs *ifa = nullptr;
    struct ifaddrs* ifaddr = nullptr;
    int s;
    char host[NI_MAXHOST] = "127.0.0.1";

    // 获取网络接口列表
    if (getifaddrs(&ifaddr) == -1) {
        TILEXR_LOG(ERROR) << "Failed to getifaddrs in BootstrapGetServerIp.";
        return TILEXR_ERROR_INTERNAL;
    }

    // 遍历网络接口列表
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }

        if (ifa->ifa_addr->sa_family == AF_INET) {
            s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                            host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
            if (s != 0) {
                TILEXR_LOG(WARN) << "getnameinfo() failed: " << gai_strerror(s);
                continue;
            }
            TILEXR_LOG(DEBUG) << "Interface: " << ifa->ifa_name << " Address: " << host;
            if (IsValidInterface(string(ifa->ifa_name))) {
                break;
            }
        }
    }
    if (ifa) {
        TILEXR_LOG(INFO) << "Interface: " << ifa->ifa_name << " Address: " << host;
    }
    freeifaddrs(ifaddr);

    // 填充 sockaddr_in 结构体
    // Zero out handle
    for (size_t i = 0; i < sizeof(handle); ++i) {
        ((uint8_t*)&handle)[i] = 0;
    }
    int ret = 0; // EOK
    handle.sin.sin_family = AF_INET;
    handle.sin.sin_addr.s_addr = inet_addr(host); // 将IP地址填入sockaddr_in
    handle.sin.sin_port = 0;

    return TILEXR_SUCCESS;
}

int BootstrapGetUniqueId(TileXRBootstrapHandle *handle, int commDomain)
{
    // Zero out handle
    for (size_t i = 0; i < sizeof(TileXRBootstrapHandle); ++i) {
        ((uint8_t*)handle)[i] = 0;
    }
    int ret = 0; // EOK

    const char* env = std::getenv("TILEXR_COMM_ID");
    if (env) {
        TILEXR_LOG(INFO) << "TILEXR_COMM_ID set by environment to " << env;
        if (GetAddrFromString(&handle->addr, env) != TILEXR_SUCCESS) {
            TILEXR_LOG(WARN) << ("Invalid TILEXR_COMM_ID, please use format: <ipv4>:<port>");
            return TILEXR_INVALID_VALUE;
        }
    } else {
        int bootRet = BootstrapGetServerIp(handle->addr);
        if (bootRet != TILEXR_SUCCESS) {
            TILEXR_LOG(ERROR) << "tilexr BootstrapGetIpPort failed!";
            return TILEXR_ERROR_INTERNAL;
        }
    }
    int dev;
    int aclRet = aclrtGetDevice(&dev);
    if (aclRet != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "ERROR: GetDevice.";
        return TILEXR_ERROR_INTERNAL;
    }
    handle->addr.sin.sin_port = htons(TILEXR_DEFAULT_SOCK_PORT + dev + commDomain);
    handle->magic = TILEXR_MAGIC;
    return TILEXR_SUCCESS;
}
} // namespace TileXR
