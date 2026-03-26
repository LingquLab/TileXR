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
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <set>
#include <string>
#include <fstream>
#include <sstream>

#include <sys/file.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <acl/acl.h>
#include <securec.h>
#include <mki/utils/env/env.h>

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
        MKI_LOG(ERROR) << "Input string does not contain a colon separating IP and port.";
        return TILEXR_ERROR_INTERNAL;
    }

    ip = inputStr.substr(0, colonPos);
    std::string portStr = inputStr.substr(colonPos + 1);

    std::istringstream portStream(portStr);
    portStream >> port;
    if (portStream.fail() || portStream.bad()) {
        MKI_LOG(ERROR) << "Invalid port number.";
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
    MKI_LOG(INFO) << "TileXRSockExchange using UniqueId mode "
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
    if (!isInit_ && Prepare() != TILEXR_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }
    isInit_ = true;
    string uuid = GetUUID();
    MKI_LOG(DEBUG) << "rank:" << rank_ << " UUID " << uuid;

    set<string> uuidSet {};
    uuidSet.insert(uuid);
    int nodeNum = -1;
    if (IsServer()) {
        for (int i = 1; i < rankSize_; ++i) {
            if (Recv(clientFds_[i], &uuid[0], uuid.size(), 0) <= 0) {
                MKI_LOG(ERROR) << "Server side recv rank " << i << " buffer failed";
                return TILEXR_ERROR_INTERNAL;
            }
            uuidSet.insert(uuid);
        }
        nodeNum = static_cast<int>(uuidSet.size());
        MKI_LOG(DEBUG) << "nodeNum:" << nodeNum;
        for (int i = 1; i < rankSize_; ++i) {
            if (Send(clientFds_[i], &nodeNum, sizeof(int), 0) <= 0) {
                MKI_LOG(ERROR) << "Server side send rank " << i << " buffer failed";
                return TILEXR_ERROR_INTERNAL;
            }
        }
    } else {
        if (Send(fd_, uuid.data(), uuid.size(), 0) <= 0) {
            MKI_LOG(ERROR) << "Client side " << rank_ << " send buffer failed";
            return TILEXR_ERROR_INTERNAL;
        }
        if (Recv(fd_, &nodeNum, sizeof(int), 0) <= 0) {
            MKI_LOG(ERROR) << "Client side " << rank_ << " recv buffer failed ";
            return TILEXR_ERROR_INTERNAL;
        }
    }
    return nodeNum;
}

void TileXRSockExchange::GetIpAndPort()
{
    const char* env = Mki::GetEnv("TILEXR_COMM_ID");

    if (env == nullptr or ParseIpAndPort(env, ip_, port_) != TILEXR_SUCCESS) {
        ip_ = TILEXR_LOCAL_SOCK_IP;
        port_ = TILEXR_DEFAULT_SOCK_PORT;
    }
    port_ += commDomain_;
    tilexrCommId_.handle.addr.sin.sin_family = AF_INET;
    // only connect localhost for safety
    tilexrCommId_.handle.addr.sin.sin_addr.s_addr = inet_addr(TILEXR_LOCAL_SOCK_IP.c_str()); // ip_.c_str()
    tilexrCommId_.handle.addr.sin.sin_port = htons(port_);
    MKI_LOG(DEBUG) << "curRank: " << rank_ << " commDomain: " << commDomain_ << " ip: " << ip_ << " port: " << port_;
}

bool IsLocalIp(const std::string& ip)
{
    struct ifaddrs* ifaddr = nullptr;
    char host[NI_MAXHOST];
    bool found = false;

    // 获取网络接口列表
    if (getifaddrs(&ifaddr) == -1) {
        MKI_LOG(WARN) << "Failed to getifaddrs in IsLocalIp.";
        return false;
    }

    // 遍历网络接口列表
    for (ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }

        // 只处理IPv4地址
        if (ifa->ifa_addr->sa_family == AF_INET) {
            // 初始化 host 缓冲区
            for (size_t i = 0; i < NI_MAXHOST; ++i) {
                host[i] = 0;
            }
            int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                                host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
            if (s != 0) {
                MKI_LOG(WARN) << "getnameinfo() failed: " << gai_strerror(s);
                continue;
            }
            // 比较输入的IP和当前接口的IP
            if (ip == host) {
                found = true;
                break;
            }
        }
    }
    freeifaddrs(ifaddr);
    return found;
}

int TileXRSockExchange::StartSecureTunnel()
{
    if (IsLocalIp(ip_)) {
        MKI_LOG(DEBUG) << "no need to start tunnel.";
        return TILEXR_SUCCESS;
    }
    // 生成一个唯一的锁文件路径，通常基于服务器IP或主机名
    std::string lockFilePath = "/tmp/secure_tunnel_lock_" + ip_;

    // 打开锁文件
    lockFileDescriptor_ = open(lockFilePath.c_str(), O_CREAT | O_RDWR,
                               S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    if (lockFileDescriptor_ < 0) {
        MKI_LOG(ERROR) << "StartSecureTunnel Failed: Unable to open lock file";
        return TILEXR_ERROR_INTERNAL;
    }

    // 尝试获取独占锁
    if (flock(lockFileDescriptor_, LOCK_EX | LOCK_NB) < 0) {
        // 如果获取锁失败，说明已经有其他进程在运行
        MKI_LOG(INFO) << "StartSecureTunnel: Another instance is already running on this server";
        close(lockFileDescriptor_);
        lockFileDescriptor_ = -1;
        return TILEXR_SUCCESS;
    }
    string port = to_string(port_) + ":localhost:" + to_string(port_);
    string cmd = "/usr/bin/ssh -N -L " + port + " " + ip_;
    MKI_LOG(INFO) << rank_ <<" StartSecureTunnel : " << cmd;

    pipe_ = popen(cmd.c_str(), "r");  // "r" 模式打开管道
    if (!pipe_) {
        MKI_LOG(ERROR) << "StartSecureTunnel Failed: popen error";
        return TILEXR_ERROR_INTERNAL;
    }

    MKI_LOG(DEBUG) << "StartSecureTunnel success!";
    return TILEXR_SUCCESS;
}

int TileXRSockExchange::Prepare()
{
    if (tilexrCommId_.handle.magic != TILEXR_MAGIC) {
        GetIpAndPort();
    }
    if (!IsServer()) {
        StartSecureTunnel();
        return Connect();
    }

    clientFds_.resize(rankSize_, -1);
    if (Listen() != TILEXR_SUCCESS) {
        MKI_LOG(ERROR) << "Listen Failed!";
        return TILEXR_ERROR_INTERNAL;
    }

    if (Accept() != TILEXR_SUCCESS) {
        MKI_LOG(ERROR) << "Accept Failed!";
        return TILEXR_ERROR_INTERNAL;
    }

    return TILEXR_SUCCESS;
}

int TileXRSockExchange::Listen()
{
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        MKI_LOG(ERROR) << "Server side create socket failed";
        return TILEXR_ERROR_INTERNAL;
    }

    int reuse = 1;
    if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(int)) < 0) {
        MKI_LOG(ERROR) << "Server side set reuseaddr failed";
        return TILEXR_ERROR_INTERNAL;
    }

    struct sockaddr *addrPtr = &tilexrCommId_.handle.addr.sa;
    if (bind(fd_, addrPtr, sizeof(struct sockaddr)) < 0) {
        MKI_LOG(ERROR) << "Server side bind " << ntohs(tilexrCommId_.handle.addr.sin.sin_port) << " failed";
        return TILEXR_ERROR_INTERNAL;
    }

    /*
     * kernel would silently truncate backlog to the value defined in
     * /proc/sys/net/core/somaxconn if it is less than 65535.
     */
    if (listen(fd_, TILEXR_MAX_BACK_LOG) < 0) {
        MKI_LOG(ERROR) << "Server side listen " << ntohs(tilexrCommId_.handle.addr.sin.sin_port) << " failed";
        return TILEXR_ERROR_INTERNAL;
    }
    MKI_LOG(INFO) << "The server is listening! ip: "<< inet_ntoa(tilexrCommId_.handle.addr.sin.sin_addr)
        << " port: " << ntohs(tilexrCommId_.handle.addr.sin.sin_port);

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
                MKI_LOG(ERROR) << "Server side accept failed" << strerror(errno);
                return -1;
            }
            MKI_LOG(DEBUG) << "accept failed: " << strerror(errno);
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
        int fd = AcceptConnection(fd_, clientAddr, &sinSize);
        if (fd < 0) {
            MKI_LOG(ERROR) << "AcceptConnection failed";
            return TILEXR_ERROR_INTERNAL;
        }

        int rank = 0;
        if (Recv(fd, &rank, sizeof(rank), 0) <= 0) {
            MKI_LOG(ERROR) << "Server side recv rank id failed";
            return TILEXR_ERROR_INTERNAL;
        }

        if (rank >= rankSize_ || rank <= 0 || clientFds_[rank] >= 0) {
            MKI_LOG(ERROR) << "Server side recv invalid rank id " << rank;
            return TILEXR_ERROR_INTERNAL;
        }

        MKI_LOG(DEBUG) << "Server side recv rank id " << rank;
        clientFds_[rank] = fd;
    }

    return TILEXR_SUCCESS;
}

void TileXRSockExchange::Close(int &fd) const
{
    if (fd == -1) {
        return;
    }

    if (close(fd) < 0) {
        MKI_LOG(WARN) << "failed to close fd:" << fd;
        return;
    }

    fd = -1;
}

int TileXRSockExchange::Connect()
{
    MKI_LOG(DEBUG) << "Client side " << rank_ << " begin to connect";

    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        MKI_LOG(ERROR) << "Client side " << rank_ << " create socket failed";
        return TILEXR_ERROR_INTERNAL;
    }

    int sleepTimeS = 1;
    int maxRetryCount = 180;
    int retryCount = 0;
    bool success = false;
    struct sockaddr *addrPtr = &tilexrCommId_.handle.addr.sa;
    while (retryCount < maxRetryCount) {
        if (connect(fd_, addrPtr, sizeof(struct sockaddr)) < 0) {
            if (errno == ECONNREFUSED) {
                MKI_LOG(DEBUG) << "Client side " << rank_ << " try connect " << (retryCount + 1) << " times refused";
                retryCount++;
                sleep(sleepTimeS);
                continue;
            }
            if (errno != EINTR) {
                MKI_LOG(ERROR) << "Client side " << rank_ << " connect failed: " << strerror(errno);
                break;
            }
            MKI_LOG(DEBUG) << "Client side " << rank_ << " try connect failed: " << strerror(errno);
            continue;
        }
        success = true;
        break;
    }

    if (!success) {
        MKI_LOG(ERROR) << "Client side " << rank_ << " connect failed";
        return TILEXR_ERROR_INTERNAL;
    }

    if (Send(fd_, &rank_, sizeof(rank_), 0) <= 0) {
        MKI_LOG(ERROR) << "Client side " << rank_ << " send rank failed";
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

    if (lockFileDescriptor_ >= 0) {
        Close(lockFileDescriptor_);
        lockFileDescriptor_ = -1;
    }
    if (pipe_) {
        pclose(pipe_);
        pipe_ = nullptr;
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
        MKI_LOG(ERROR) << "tilexr ParseIpAndPort failed!";
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
        MKI_LOG(ERROR) << "Failed to getifaddrs in BootstrapGetServerIp.";
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
                MKI_LOG(WARN) << "getnameinfo() failed: " << gai_strerror(s);
                continue;
            }
            MKI_LOG(DEBUG) << "Interface: " << ifa->ifa_name << " Address: " << host;
            if (IsValidInterface(string(ifa->ifa_name))) {
                break;
            }
        }
    }
    if (ifa) {
        MKI_LOG(INFO) << "Interface: " << ifa->ifa_name << " Address: " << host;
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

    const char* env = Mki::GetEnv("TILEXR_COMM_ID");
    if (env) {
        MKI_LOG(INFO) << "TILEXR_COMM_ID set by environment to " << env;
        if (GetAddrFromString(&handle->addr, env) != TILEXR_SUCCESS) {
            MKI_LOG(WARN) << ("Invalid TILEXR_COMM_ID, please use format: <ipv4>:<port>");
            return TILEXR_INVALID_VALUE;
        }
    } else {
        int bootRet = BootstrapGetServerIp(handle->addr);
        if (bootRet != TILEXR_SUCCESS) {
            MKI_LOG(ERROR) << "tilexr BootstrapGetIpPort failed!";
            return TILEXR_ERROR_INTERNAL;
        }
    }
    int dev;
    int aclRet = aclrtGetDevice(&dev);
    if (aclRet != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "ERROR: GetDevice.";
        return TILEXR_ERROR_INTERNAL;
    }
    handle->addr.sin.sin_port = htons(TILEXR_DEFAULT_SOCK_PORT + dev + commDomain);
    handle->magic = TILEXR_MAGIC;
    return TILEXR_SUCCESS;
}
} // namespace TileXR
