/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEXR_SOCK_EXCHANGE_H
#define TILEXR_SOCK_EXCHANGE_H

#include <algorithm>
#include <limits>
#include <new>
#include <vector>
#include <string>
#include <memory>

#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>

#include "tilexr_log.h"

#include "tilexr_types.h"
#include "tilexr_api.h"

namespace TileXR {
/* Common socket address storage structure for IPv4/IPv6 */
union TileXRSocketAddress {
    struct sockaddr sa;
    struct sockaddr_in sin;
    struct sockaddr_in6 sin6;
};

constexpr uint64_t TILEXR_MAGIC = 0xdddd0000dddd0000;

struct TileXRBootstrapHandle {
    uint64_t magic;
    union TileXRSocketAddress addr;
};
union TileXRBootstrap {
    TileXRBootstrapHandle handle;
    TileXRUniqueId uid;
};

int BootstrapGetUniqueId(TileXRBootstrapHandle *handle, int commDomain);

class TileXRSockExchange {
public:
    TileXRSockExchange(int rank, int rankSize, int commDomain);
    TileXRSockExchange(int rank, int rankSize, TileXRUniqueId tilexrCommId);
    ~TileXRSockExchange();

    /* *
     * @brief All gather data from @ref sendBuf to @ref recvBuf
     *
     * @note recvBuf's space must larger than sendSize * rankSize_
     * @return TILEXR_SUCCESS for success, other for failed
     */
    template <typename T> int AllGather(const T *sendBuf, size_t sendCount, T *recvBuf)
    {
        if (!isInit_ && Prepare() != TILEXR_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }
        isInit_ = true;

        if (!IsServer()) {
            return ClientSendRecv(sendBuf, sendCount, recvBuf);
        } else {
            return ServerRecvSend(sendBuf, sendCount, recvBuf);
        }
    }

    /* sendBuf is [destination rank][element], recvBuf is [source rank][element]. */
    template <typename T> int AllToAll(const T *sendBuf, size_t sendCountPerRank, T *recvBuf)
    {
        if (!isInit_ && Prepare() != TILEXR_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }
        isInit_ = true;

        if (!IsServer()) {
            return ClientSendRecvAllToAll(sendBuf, sendCountPerRank, recvBuf);
        }
        return ServerRecvSendAllToAll(sendBuf, sendCountPerRank, recvBuf);
    }

    int GetNodeNum();

    static bool CheckValid(TileXRUniqueId tilexrCommId)
    {
        TileXRBootstrap id {};
        id.uid = tilexrCommId;
        return id.handle.magic == TILEXR_MAGIC;
    }

private:
    void GetIpAndPort();
    int Prepare();
    int Listen();
    int Accept();
    void Close(int &fd) const;
    int Connect();
    int AcceptConnection(int fd, sockaddr_in &clientAddr, socklen_t *sinSize) const;
    void Cleanup();
    bool IsServer() const;
    static bool CheckErrno(int ioErrno)
    {
        return ((ioErrno == EAGAIN) || (ioErrno == EWOULDBLOCK) || (ioErrno == EINTR));
    }

    template <typename T> ssize_t Send(int fd, const T *sendBuf, size_t sendSize, int flag) const
    {
        const char *data = reinterpret_cast<const char *>(sendBuf);
        size_t sent = 0;
        while (sent < sendSize) {
            auto ret = send(fd, data + sent, sendSize - sent, flag);
            if (ret < 0) {
                if (CheckErrno(errno)) {
                    TILEXR_LOG(ERROR) << "send failed: " << strerror(errno);
                    continue;
                }
                TILEXR_LOG(DEBUG) << "Send failed: " << strerror(errno);
                return ret;
            }
            if (ret == 0) {
                TILEXR_LOG(DEBUG) << "Send returned zero before buffer completion";
                return -1;
            }
            sent += static_cast<size_t>(ret);
        }
        return static_cast<ssize_t>(sent);
    }

    template <typename T> ssize_t Recv(int fd, T *recvBuf, size_t recvSize, int flag) const
    {
        char *data = reinterpret_cast<char *>(recvBuf);
        size_t received = 0;
        while (received < recvSize) {
            auto ret = recv(fd, data + received, recvSize - received, flag);
            if (ret < 0) {
                if (CheckErrno(errno)) {
                    TILEXR_LOG(ERROR) << "recv failed: " << strerror(errno);
                    continue;
                }
                TILEXR_LOG(DEBUG) << "recv failed: " << strerror(errno);
                return ret;
            }
            if (ret == 0) {
                TILEXR_LOG(DEBUG) << "Recv reached EOF before buffer completion";
                return -1;
            }
            received += static_cast<size_t>(ret);
        }
        return static_cast<ssize_t>(received);
    }

    template <typename T> int ClientSendRecv(const T *sendBuf, size_t sendSize, T *recvBuf)
    {
        if (Send(fd_, sendBuf, sendSize * sizeof(T), 0) <= 0) {
            TILEXR_LOG(ERROR) << "Client side " << rank_ << " send buffer failed";
            return TILEXR_ERROR_INTERNAL;
        }

        if (Recv(fd_, recvBuf, sendSize * rankSize_ * sizeof(T), MSG_WAITALL) <= 0) {
            TILEXR_LOG(ERROR) << "Client side " << rank_ << " recv buffer failed ";
            return TILEXR_ERROR_INTERNAL;
        }

        return TILEXR_SUCCESS;
    }

    template <typename T> int ServerRecvSend(const T *sendBuf, size_t sendSize, T *recvBuf)
    {
        for (int i = 0; i < sendSize; ++i) {
            recvBuf[i] = sendBuf[i];
        }

        for (int i = 1; i < rankSize_; ++i) {
            if (Recv(clientFds_[i], recvBuf + i * sendSize, sendSize * sizeof(T), MSG_WAITALL) <= 0) {
                TILEXR_LOG(ERROR) << "Server side recv rank " << i << " buffer failed";
                return TILEXR_ERROR_INTERNAL;
            }
        }

        for (int i = 1; i < rankSize_; ++i) {
            if (Send(clientFds_[i], recvBuf, sendSize * rankSize_ * sizeof(T), 0) <= 0) {
                TILEXR_LOG(ERROR) << "Server side send rank " << i << " buffer failed";
                return TILEXR_ERROR_INTERNAL;
            }
        }

        return TILEXR_SUCCESS;
    }

    template <typename T> bool AllToAllSizes(size_t sendCountPerRank, size_t& rowCount, size_t& totalCount) const
    {
        if (rankSize_ <= 0 || sendCountPerRank == 0 ||
            sendCountPerRank > std::numeric_limits<size_t>::max() / static_cast<size_t>(rankSize_)) {
            return false;
        }
        rowCount = sendCountPerRank * static_cast<size_t>(rankSize_);
        if (rowCount > std::numeric_limits<size_t>::max() / sizeof(T) ||
            rowCount > std::numeric_limits<size_t>::max() / static_cast<size_t>(rankSize_)) {
            return false;
        }
        totalCount = rowCount * static_cast<size_t>(rankSize_);
        return true;
    }

    template <typename T> int ClientSendRecvAllToAll(
        const T *sendBuf, size_t sendCountPerRank, T *recvBuf)
    {
        size_t rowCount = 0;
        size_t totalCount = 0;
        if (!AllToAllSizes<T>(sendCountPerRank, rowCount, totalCount)) {
            return TILEXR_ERROR_INTERNAL;
        }
        (void)totalCount;
        const size_t rowBytes = rowCount * sizeof(T);
        if (Send(fd_, sendBuf, rowBytes, 0) <= 0) {
            TILEXR_LOG(ERROR) << "Client side " << rank_ << " send alltoall buffer failed";
            return TILEXR_ERROR_INTERNAL;
        }
        if (Recv(fd_, recvBuf, rowBytes, MSG_WAITALL) <= 0) {
            TILEXR_LOG(ERROR) << "Client side " << rank_ << " recv alltoall buffer failed";
            return TILEXR_ERROR_INTERNAL;
        }
        return TILEXR_SUCCESS;
    }

    template <typename T> int ServerRecvSendAllToAll(
        const T *sendBuf, size_t sendCountPerRank, T *recvBuf)
    {
        size_t rowCount = 0;
        size_t totalCount = 0;
        if (!AllToAllSizes<T>(sendCountPerRank, rowCount, totalCount)) {
            return TILEXR_ERROR_INTERNAL;
        }
        const size_t rowBytes = rowCount * sizeof(T);
        std::vector<T> gathered;
        std::vector<T> destination;
        try {
            gathered.resize(totalCount);
            destination.resize(rowCount);
        } catch (const std::bad_alloc&) {
            TILEXR_LOG(ERROR) << "Server side allocate alltoall exchange buffer failed";
            return TILEXR_ERROR_INTERNAL;
        }
        std::copy_n(sendBuf, rowCount, gathered.data());
        for (int source = 1; source < rankSize_; ++source) {
            if (Recv(clientFds_[source], gathered.data() + static_cast<size_t>(source) * rowCount,
                    rowBytes, MSG_WAITALL) <= 0) {
                TILEXR_LOG(ERROR) << "Server side recv alltoall rank " << source << " failed";
                return TILEXR_ERROR_INTERNAL;
            }
        }

        for (int target = 0; target < rankSize_; ++target) {
            for (int source = 0; source < rankSize_; ++source) {
                const size_t sourceOffset =
                    (static_cast<size_t>(source) * rankSize_ + target) * sendCountPerRank;
                std::copy_n(gathered.data() + sourceOffset, sendCountPerRank,
                    destination.data() + static_cast<size_t>(source) * sendCountPerRank);
            }
            if (target == 0) {
                std::copy_n(destination.data(), rowCount, recvBuf);
            } else if (Send(clientFds_[target], destination.data(), rowBytes, 0) <= 0) {
                TILEXR_LOG(ERROR) << "Server side send alltoall rank " << target << " failed";
                return TILEXR_ERROR_INTERNAL;
            }
        }
        return TILEXR_SUCCESS;
    }
    int rank_ = 0;
    int rankSize_ = 0;
    int fd_ = -1;
    std::vector<int> clientFds_ = {};
    bool isInit_ = false;
    int commDomain_ = -1;
    std::string ip_;
    uint16_t port_ = 0;
    TileXRBootstrap tilexrCommId_ = {};
};
} // namespace TileXR

#endif
