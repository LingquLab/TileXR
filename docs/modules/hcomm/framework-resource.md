# 模块：hcomm/framework-resource（资源管理层）

## 1. 模块作用

管理通信所需的所有运行时资源：socket 连接（ibverbs/TCP）、transport 对象（P2P/RDMA）、CCL buffer、workspace 内存、stream、notify/event 队列。是 `HcclCommunicator` 与底层网络/内存之间的资源调度中心。

---

## 2. 目录结构

```
src/framework/communicator/impl/resource_manager/
├── transport_manager.h/.cc            # Transport 创建与管理（核心）
├── hccl_socket_manager.h/.cc          # Socket 生命周期管理（36KB）
├── ccl_buffer_manager.h/.cc           # CCL buffer 分配与管理
├── share_ccl_buffer_manager.h/.cc     # 共享 CCL buffer 管理
├── workspace_mem.h/.cc                # Workspace 内存基础
├── workspace_resource.h/.cc           # Workspace 资源封装
├── workspace_resource_impl.h/.cc      # Workspace 资源实现（14KB）
├── offload_stream_manager.h/.cc       # Offload stream 管理
├── offload_stream_manager_pub.h       # Offload stream 公共接口
├── op_base_stream_manager.h/.cc       # OpBase stream 管理
├── op_base_stream_manager_pub.h
├── queue_notify_manager.h/.cc         # 队列/通知（notify/event）管理
└── stream_active_manager.h/.cc        # Stream 活跃状态管理
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `transport_manager.cc` | 核心：`CreateDestSockets()`（创建目标端 socket）、`UpdateIsInterRdma()`（P2P vs RDMA 决策）、`GetTransportByRank()`（获取已建立的 transport 对象） |
| `hccl_socket_manager.cc` | `CreateSingleLinkSocket()`（建立单条链路 socket）、`ServerInit()`（启动监听）；Line 345：`HcclNetDevGetNicType(netDevCtx, ...)` 是 `netDevCtx=nullptr` crash 的位置 |
| `ccl_buffer_manager.cc` | 管理 CCL 通信 buffer 的分配、复用和释放；支持多 stream 并发使用不同 buffer slot |
| `workspace_resource_impl.cc` | Workspace 内存的分配策略：按算子类型预分配固定大小（`COMM_MAX_WORK_SPACE_SIZE=16MB`）或动态按需分配 |
| `queue_notify_manager.cc` | 管理 notify/event 对象池，为通信算子提供同步原语 |

---

## 4. 核心函数 / 类 / 接口

### TransportManager

```cpp
class TransportManager {
    // 决定是否走 RDMA 路径
    // forceRdma=true 时强制 RDMA（MC2 ibverbs 场景）
    void UpdateIsInterRdma(RankId remoteRank, bool& isInterRdma, bool forceRdma);
    // 单机：isInterRdma = (isUsedRdmaLevel0_ && linkType==PXI_TYPE) || forceRdma

    // 为目标 rank 创建 socket（ibverbs 或 TCP）
    HcclResult CreateDestSockets(const std::string& tag, RankId remoteRank,
        u64 taskNum, std::vector<shared_ptr<HcclSocket>>& sockets,
        HcclNetDevCtx& netDevCtx, bool& isInterRdma, bool forceRdma,
        bool isBackup, u32 subCommIndex, TransportLinkType linkType);
    // 关键路径：netDevCtx = netDevCtxMap_[devIpAddr_[0]]（若为空则 nullptr）

    // 为每个有效 rank 并行创建 Transport 链路线程
    HcclResult createSubCommLinkThreads(const std::string& tag,
        std::vector<u32>& rankList, CommInfo& commInfo, ...);
    // 跳过条件：无效 rank / 已存在链路 / 非 RDMA 的备份链路
    // 每个有效 rank 创建独立线程调用 CreateLink(rankId, ...)

    // 获取已建立的 transport 对象
    std::shared_ptr<Transport> GetTransportByRank(RankId rank) const;

    // 关键常量
    // AICPU_RETRY_BACKUP_PORT     = 16667
    // MASSIVE_IBV_CONNECTION_COUNT = 1000（大规模 ibverbs 连接阈值，超过后切换批量建连模式）
};
```

### HcclSocketManager

```cpp
class HcclSocketManager {
    // 启动监听（填充 netDevCtxMap_ 后调用）
    HcclResult ServerInit(HcclNetDevCtx netDevCtx, u32 port);

    // 建立单条链路 socket（ibverbs QP 或 TCP 连接）
    HcclResult CreateSingleLinkSocket(const std::string& tag,
        HcclNetDevCtx netDevCtx,                // Line 345 crash 点：此处不能为 nullptr
        const HcclRankLinkInfo& remoteLinkInfo,
        std::vector<shared_ptr<HcclSocket>>& sockets,
        bool isServer, bool isBackup);
};
```

### CCL Buffer Manager

```cpp
class CclBufferManager {
    // 分配通信 buffer（支持多 slot 并发）
    HcclResult AllocBuffer(u32 slotIdx, size_t size, DeviceMem& mem);
    HcclResult FreeBuffer(u32 slotIdx);

    // 查询可用 buffer slot
    u32 GetAvailableSlot() const;
};
```

### Workspace Resource

```cpp
class WorkspaceResourceImpl {
    // 按算子类型分配 workspace
    HcclResult AllocWorkspace(const std::string& tag, size_t size, DeviceMem& mem);

    // 释放（实际上是归还到池中，不立即 free）
    HcclResult FreeWorkspace(const std::string& tag);
};
```

---

## 5. 数据流向

```
CommBase（通信域基类）
  └── TransportManager::CreateDestSockets()
        ├── UpdateIsInterRdma()        → 决定 P2P 或 RDMA
        ├── MakeRemoteLinkInfo()       → 构造目标端 IP/port
        └── socketManager_->CreateSingleLinkSocket()
              ├── netDevCtx = netDevCtxMap_[devIpAddr_[0]]
              └── HcclNetDevGetNicType(netDevCtx, ...)  ← Line 345

算子执行时
  ├── CclBufferManager::AllocBuffer()  → 获取通信 buffer
  ├── WorkspaceResourceImpl::AllocWorkspace() → 获取 workspace
  ├── QueueNotifyManager::GetNotify()  → 获取同步 notify
  └── OffloadStreamManager::GetStream() → 获取 offload stream
```

---

## 6. 关键业务逻辑

### P2P vs RDMA 路径决策（UpdateIsInterRdma）

```
单机场景：
  isInterRdma = (isUsedRdmaLevel0_ && linkType == PXI_TYPE) || forceRdma

多机场景：
  isInterRdma = isInterServer || isInterSuperPod || forceRdma
```

`forceRdma=true` 由 `CommParaInfo.forceRdma` 传入，MC2 双通道场景下 `commLevel0Rdma` 创建时设置。

### netDevCtx 查找逻辑
```cpp
// CreateDestSockets 中
netDevCtx = nicDeployment_ == NIC_DEPLOYMENT_DEVICE
    ? netDevCtxMap_[devIpAddr_[0]]   // 设备 NIC
    : netDevCtxMap_[hostIp_];        // 主机 NIC
```
若 `devIpAddr_` 为空（单机场景未填充），`netDevCtxMap_[devIpAddr_[0]]` 返回默认构造的 `nullptr`，导致 Line 345 crash。

### 大规模连接优化
当 ibverbs 连接数超过 `MASSIVE_IBV_CONNECTION_COUNT=1000` 时，`TransportManager` 切换到批量建连模式，减少握手轮次。

### Buffer Slot 复用
`CclBufferManager` 维护固定数量的 buffer slot（通常 2-4 个），算子执行时轮转使用，避免频繁分配/释放。

---

## 7. 开发注意事项

- `hccl_socket_manager.cc:345` 的 `HcclNetDevGetNicType(netDevCtx, ...)` 是已知 crash 点，根本原因是 `netDevCtxMap_` 未初始化（`InitNic` 未被调用）。
- `TransportManager` 持有 `netDevCtxMap_` 的引用（非拷贝），`HcclCommunicator` 中对 `netDevCtxMap_` 的修改对 `TransportManager` 立即可见。
- `CclBufferManager` 的 slot 数量在编译期确定，若算子并发度超过 slot 数会阻塞等待，需根据实际并发场景调整。
- `WorkspaceResourceImpl` 的 workspace 不会立即释放，而是缓存到池中，内存占用可能持续增长，需关注内存泄漏。

---

## 8. 未来可扩展点

- **Transport 缓存**：`GetTransportByRank` 目前每次 MC2 调用都重建，可增加 tag-based 缓存，同 tag 复用已建立的 transport。
- **动态 buffer 大小**：`CclBufferManager` 的 slot 大小目前固定，可改为按算子数据量动态调整。
- **连接池**：ibverbs QP 建立开销大，可实现 QP 连接池，跨算子复用已建立的连接。
- **Workspace 压缩**：多个算子的 workspace 可以时分复用同一块内存，减少总内存占用。
