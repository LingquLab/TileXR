/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEXR_COMM_H
#define TILEXR_COMM_H

#include <vector>
#include <string>
#include "acl/acl_rt.h"
#include "../include/tilexr_types.h"
#include "../include/tilexr_api.h"
#include "../include/comm_args.h"

namespace TileXR {
constexpr int IPC_NAME_SIZE = 65;

class TileXRSockExchange;
class TileXRComm {
public:
    TileXRComm(int rank, int rankSize);
    TileXRComm(int rank, int rankSize, int commDomain, int bufferSize);
    TileXRComm(int rank, int rankSize, TileXRUniqueId commId);
    TileXRComm(int rank, int rankSize, TileXRUniqueId commId, uint64_t hostAddr, uint64_t hostMemAddr);
    ~TileXRComm();
    TileXRComm(const TileXRComm &) = delete;
    TileXRComm &operator=(const TileXRComm &) = delete;
    int Init();
    int InitThread(const std::string &uid = "default");
    int GetRank() const;
    int GetRankSize() const;
    int GetCommSize() const;
    const PhysicalInfo &GetPhysicalInfo() const;
    GM_ADDR GetCommArgsPtr();
    CommArgs* GetCommArgs();
    std::string PrintDFX();
    friend class Lccl;
    friend class Lcoc;
    friend class LcclTest;

private:
    int SetMemoryName(std::string &name);
    int SetIpcPidSdid(std::string &name, const uint32_t *pids, const int64_t *sdids) const;
    int OpenIpcMem(const char names[TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE]);
    int GetDev();
    int GetDevThread(const std::string &uid = "");
    int EnablePeerAccess();
    int InitCommMem();
    int InitHostMem();
    int InitCommon();
    void CloseIpcMem();
    void FreePeerMem(GM_ADDR &mem) const;
    int InitMem();
    int GetSidId(int64_t sdids[TILEXR_MAX_RANK_SIZE], int rankSize);
    int GetPid(uint32_t *pids);
    int GetName(std::string &name, char names[TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE]) const;
    int SyncCommArgs();
    int InitDumpAddr();

private:
    int rank_ = 0;  // global rank id
    int rankSize_ = 0;  // global rank size
    int commSize_ = 0;  // local TileXRComm size
    int localRank_ = -1;
    uint32_t localRankSize_ = 0;
    int devId_ = 0;
    int64_t magic_ = 1;
    void *hostMemAddr_ = nullptr;
    bool inited_ = false;
    bool ipcMemInited_ = false;
    std::string uid_ = {};
    std::vector<int> devList_ = {};
    int commDomain_ = {};
    int bufferSize_ = TILEXR_COMM_BUFFER_SIZE;

    // shared ping pong buff，这个地址就是一开始申请在HBM上的，所以host上可以取到，但不能直接修改。
    GM_ADDR peerMem_[TILEXR_MAX_RANK_SIZE] = {};
    GM_ADDR hostMappingAddr_[TILEXR_MAX_RANK_SIZE] = {};
    aclrtMemFabricHandle share_handles[TILEXR_MAX_RANK_SIZE] = {};
    PhysicalInfo physicalInfo_ = {};
    CommArgs commArgs_ = {};    // host侧
    GM_ADDR commArgsPtr_ = nullptr; // device侧
    TileXRUniqueId commId_ = {};
    TileXRSockExchange *socketExchange_ = nullptr;
    bool isEnableMsprofOp_ = false;
};
} // TileXR

#endif // TILEXR_COMM_H
