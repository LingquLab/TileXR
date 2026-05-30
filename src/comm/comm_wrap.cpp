/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tilexr_api.h"
#include <vector>
#include <thread>
#include <atomic>
#include <acl/acl_rt.h>

#include "tilexr_comm.h"
#include "tilexr_log.h"
#include "tools/socket/tilexr_sock_exchange.h"

using namespace std;
using namespace TileXR;

int TileXRCommInitRankLocal(int rankSize, int rank, TileXRCommPtr *comm)
{
    TILEXR_LOG(INFO) << "using tilexr c++ api! rank" << rank;
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm ptr is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    auto* c = new (std::nothrow) TileXRComm(rank, rankSize);
    if (c == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    *comm = c;
    int ret = c->Init();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr init failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRGetUniqueId(TileXRUniqueId *uniqueId, int commDomain)
{
    if (uniqueId == nullptr) {
        TILEXR_LOG(ERROR) << "uniqueId is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    int res = BootstrapGetUniqueId(reinterpret_cast<struct TileXRBootstrapHandle *>(uniqueId), commDomain);
    if (res != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr BootstrapGetUniqueId failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRCommInitRank(TileXRUniqueId commId, int rankSize, int rank, TileXRCommPtr *comm)
{
    TILEXR_LOG(INFO) << "using tilexr c++ api! rank" << rank;
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm ptr is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    auto* c = new (std::nothrow) TileXRComm(rank, rankSize, commId);
    if (c == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    *comm = c;
    int ret = c->Init();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr init failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRCommInitRankWithCustDomainSize(int commDomain, int bufferSize, int rankSize, int rank, TileXRCommPtr *comm)
{
    TILEXR_LOG(INFO) << "using tilexr c++ api! rank" << rank;
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm ptr is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }

    constexpr int minBufferSize = TILEXR_COMM_BUFFER_SIZE;
    if (bufferSize < minBufferSize) {
        TILEXR_LOG(ERROR) << "tilexr comm buffer size " << bufferSize << " MBytes should not be less than " <<
            minBufferSize << " MBytes!";
        return TILEXR_ERROR_INTERNAL;
    }

    auto* c = new TileXRComm(rank, rankSize, commDomain, bufferSize);
    *comm = c;
    int ret = c->Init();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr init failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRCommInitRankWithDomain(int commDomain, int rankSize, int rank, TileXRCommPtr *comm)
{
    constexpr int minBufferSize = TILEXR_COMM_BUFFER_SIZE;
    return TileXRCommInitRankWithCustDomainSize(commDomain, minBufferSize, rankSize, rank, comm);
}

int TileXRGetCommArgsDev(TileXRCommPtr comm, GM_ADDR &commArgsPtr)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    auto* tilexr = static_cast<TileXRComm *>(comm);
    commArgsPtr = tilexr->GetCommArgsPtr();
    return TILEXR_SUCCESS;
}

int TileXRGetCommArgsHost(TileXRCommPtr comm, TileXR::CommArgs *&commArgsPtr)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    commArgsPtr = c->GetCommArgs();
    return TILEXR_SUCCESS;
}

int TileXRCommNextMagic(TileXRCommPtr comm, int64_t *magic)
{
    if (comm == nullptr || magic == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRCommNextMagic invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    *magic = c->NextMagic();
    return TILEXR_SUCCESS;
}

int TileXRUDMARegister(TileXRCommPtr comm, GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle *handle)
{
    if (comm == nullptr || localPtr == nullptr || handle == nullptr || bytes == 0) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    return c->RegisterUDMAMemory(localPtr, bytes, handle);
}

int TileXRUDMAUnregister(TileXRCommPtr comm, TileXRUDMAMemHandle handle)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRUDMAUnregister invalid comm";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    return c->UnregisterUDMAMemory(handle);
}

int TileXRGetUDMARegistryDev(TileXRCommPtr comm, GM_ADDR &registryPtr)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRGetUDMARegistryDev invalid comm";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    registryPtr = c->GetUDMARegistryPtr();
    return TILEXR_SUCCESS;
}

void TileXRPrintDFX2Log(TileXRCommPtr comm)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm is nullptr!";
        return;
    }
    auto* tilexr = static_cast<TileXRComm *>(comm);
    TILEXR_LOG(INFO) << tilexr->PrintDFX();
}

int TileXRCommInit(int rank, int rankSize, TileXRCommPtr *comms)
{
    if (comms == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comms is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    *comms = new (std::nothrow) TileXRComm(rank, rankSize);
    if (*comms == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRCommInitAll(uint32_t ndev, int32_t *devices, TileXRCommPtr *comms)
{
    if (comms == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comms is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    if (devices == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr devices is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    static int commDomain = 0;
    commDomain++;
    for (uint32_t i = 0; i < ndev; ++i) {
        comms[i] = new (std::nothrow) TileXRComm(i, ndev, commDomain, TILEXR_COMM_BUFFER_SIZE);
        if (comms[i] == nullptr) {
            TILEXR_LOG(ERROR) << "TileXRComm create failed. dev : " << i << ", ndev : " << ndev;
            return TILEXR_ERROR_INTERNAL;
        }
    }
    static atomic<int> uid;
    uid++;
    vector<unique_ptr<thread>> threads;
    int error = TILEXR_SUCCESS;
    for (uint32_t r = 0; r < ndev; r++) {
        threads.emplace_back(make_unique<thread>(
            [&](int rank) {
                aclrtSetDevice(devices[rank]);
                auto* c = static_cast<TileXRComm *>(comms[rank]);
                int ret = c->InitThread("uid" + to_string(uid));
                if (ret != TILEXR_SUCCESS) {
                    error = ret;
                }
            },
            r));
    }
    for (auto &t : threads) {
        t->join();
    }
    threads.clear();
    return error;
}

int TileXRCommInitThread(int rank, int rankSize, const char *uid, TileXRCommPtr *comms)
{
    if (uid == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr uid is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    if (comms == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comms is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    if (rank >= rankSize) {
        TILEXR_LOG(ERROR) << "tilexr rank : " << rank << " rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    *comms = new (std::nothrow) TileXRComm(rank, rankSize);
    if (*comms == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    auto* c = static_cast<TileXRComm *>(*comms);
    return c->InitThread(string(uid));
}

int TileXRCommDestroy(TileXRCommPtr comm)
{
    if (comm == nullptr) {
        return TILEXR_INVALID_VALUE;
    }
    auto *c = static_cast<TileXRComm *>(comm);

    delete c;
    return TILEXR_SUCCESS;
}
