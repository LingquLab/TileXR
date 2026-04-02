# 模块：hcomm-framework（HcclCommunicator 框架层）

## 1. 模块作用

HCCL（Huawei Collective Communication Library）的 host 侧框架层，负责：
- Communicator 对象的完整生命周期管理
- 网卡初始化（`InitNic`）与 socket 管理
- 通信资源分配（`SetCommResource`）：P2P 内存窗口（`windowsIn/Out`）和 ibverbs RDMA 数据（`ibverbsData`）
- MC2 算子的 Context 构建与 kernel launch（`Mc2CreateAndLaunchContext`）
- Communicator 属性（topo、rank 信息、设备类型等）的收集与缓存

---

## 2. 目录结构

```
src/framework/communicator/impl/
├── hccl_communicator.h/.cc           # 基类接口与通用实现
├── hccl_communicator_host.cc         # 主体实现（~470KB，host 侧核心逻辑）
├── hccl_communicator_device.cc       # 设备侧特化实现
├── hccl_communicator_attrs.h/.cc     # 属性基类
├── hccl_communicator_attrs_host.cc   # Host 侧属性收集
├── hccl_communicator_attrs_device.cc # Device 侧属性收集
├── comm_topo_desc.h/.cc              # 拓扑描述
├── task_abort_handler.h/.cc          # 任务中止处理
└── resource_manager/                 # 资源管理子模块
    ├── hccl_socket_manager.h/.cc     # Socket 生命周期管理（ibverbs/TCP）
    ├── transport_manager.h/.cc       # Transport 创建与管理
    ├── ccl_buffer_manager.h/.cc      # CCL buffer 分配
    ├── share_ccl_buffer_manager.h/.cc
    ├── offload_stream_manager.h/.cc  # Offload stream 管理
    ├── op_base_stream_manager.h/.cc  # OpBase stream 管理
    ├── workspace_resource_impl.h/.cc # Workspace 内存管理
    ├── queue_notify_manager.h/.cc    # 队列/通知管理
    └── stream_active_manager.h/.cc
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `hccl_communicator_host.cc` | 整个框架层最核心的文件，包含 `Init`、`InitNic`、`InitRaResource`、`SetCommResource`、`Mc2CreateAndLaunchContext`、`CreateCommAndStreamRes` 等所有关键路径 |
| `hccl_communicator_attrs_host.cc` | 收集 rank 信息、拓扑类型、设备物理 ID 等属性，为 `Init` 提供数据 |
| `resource_manager/transport_manager.cc` | `CreateDestSockets`（创建目标端 socket）、`UpdateIsInterRdma`（决定 P2P vs RDMA 路径） |
| `resource_manager/hccl_socket_manager.cc` | `CreateSingleLinkSocket`、`ServerInit`，实际 socket 连接建立（ibverbs 或 TCP） |

---

## 4. 核心函数 / 类 / 接口

### `HcclCommunicator` 生命周期

```cpp
// Init 流程（顺序调用）
HcclResult Init(HcclCommParams params, HcclRootHandle handle);
  └── InitRaResource()        // 网络资源初始化（HcclNetInit、InitSocketManager）
        └── InitNic()         // 网卡初始化，填充 netDevCtxMap_
  └── InitTransportManager()  // TransportManager 初始化
  └── InitHcclAlg()           // 算法层初始化（hcclImpl）
```

### MC2 双传输通道资源创建

```cpp
// MC2 场景资源创建入口（在 CreateCommAndStreamRes 中调用）
HcclResult SetCommResource(HcclA2CombineOpParam& param);
  ├── 获取 commLevel0（P2P transport）
  ├── 获取 commLevel0Rdma（ibverbs transport，forceRdma=true 创建）
  ├── 填充 transDevIbverbsDataMem_（ibverbs 传输地址 + rkey）
  └── 为每个 rank 同时填充 P2P window 和 ibverbs data

// MC2 Context 构建与 launch
HcclResult Mc2CreateAndLaunchContext(HcclA2CombineOpParam& param);
  ├── 多机场景：H2D 拷贝 windowsIn/Out
  ├── 单机场景：H2D 拷贝 transDevIbverbsDataMem_ 到 device
  └── 设置 combinOparaPtr->ibverbsData / ibverbsDataSize
```

### 网卡初始化

```cpp
HcclResult InitNic(bool isMC2ReInit = false);
  // 早退条件（单机非 MC2 场景）：
  // !GetExternalInputIntraRoceSwitch() && servRankInfo_.size()==1
  //   && isDiffDeviceModule_ && !isMC2ReInit
  //
  // 主流程：
  //   1. 遍历 devIpAddr_，为每个 IP 调用 HcclNetOpenDev
  //   2. 调用 socketManager_->ServerInit(netDevCtx, port)
  //   3. 填充 netDevCtxMap_[ip] = netDevCtx
```

### TransportManager 关键函数

```cpp
// 决定某 rank 对是否走 RDMA（ibverbs）路径
void UpdateIsInterRdma(RankId remoteRank, bool& isInterRdma, bool forceRdma);
  // forceRdma=true 时强制 RDMA，否则按拓扑判断

// 为目标 rank 创建 socket（ibverbs 或 TCP）
HcclResult CreateDestSockets(const std::string& tag, RankId remoteRank,
    u64 taskNum, std::vector<shared_ptr<HcclSocket>>& sockets,
    HcclNetDevCtx& netDevCtx, bool& isInterRdma, bool forceRdma,
    bool isBackup, u32 subCommIndex, TransportLinkType linkType);
  // 关键：netDevCtx = netDevCtxMap_[devIpAddr_[0]]（若为空则 nullptr）
```

---

## 5. 数据流向

```
HcclCommunicator::Init()
  │
  ├── InitRaResource()
  │     └── InitNic() → 填充 netDevCtxMap_[ip] = netDevCtx
  │
  ├── InitHcclAlg() → hcclImpl::CreateCommByAlg()
  │     ├── 创建 commLevel0（P2P，CreateCommPlane）
  │     └── 创建 commLevel0Rdma（forceRdma=true）
  │           └── CommFactory::CreateCommPlane()
  │                 └── TransportManager::CreateDestSockets()
  │                       └── netDevCtxMap_[devIpAddr_[0]]
  │                             （需 InitNic 已完成）
  │
  └── CreateCommAndStreamRes()
        ├── SetCommResource()
        │     ├── 遍历 rank → P2P window 地址
        │     └── 遍历 rank → ibverbs rkey/addr（commLevel0Rdma）
        └── Mc2CreateAndLaunchContext()
              └── H2D 拷贝 → combinOpara.ibverbsData
```

---

## 6. 关键业务逻辑

### 双传输通道（Dual Transport）
MC2 单机场景同时需要两种 transport：
- `commLevel0`：SDMA/P2P，用于填充 `windowsIn/Out`
- `commLevel0Rdma`：ibverbs，用于填充 `ibverbsData`

`CommParaInfo.forceRdma = true` 强制 `CommFactory::GetIsUsedRdma` 返回 `true`，保证 ibverbs transport 被正确创建。

### InitNic 调用时序
`InitNic` 必须在 `CreateCommByAlg`（创建 ibverbs transport）之前完成，否则 `netDevCtxMap_` 为空，导致 `CreateDestSockets` 中 `netDevCtx = nullptr`，引发 `hccl_socket_manager.cc:345` 的 crash。

`CreateCommAndStreamRes` 中的 `InitNic(true)` 调用用于单机 ibverbs 场景，传 `isMC2ReInit=true` 绕过单机早退逻辑。

### `netDevCtxMap_` 的构建
- **正常路径**：`InitRaResource` → `InitNic` → `HcclNetOpenDev` → `netDevCtxMap_[ip] = ctx`
- **MC2 单机路径**：`CreateCommAndStreamRes` 检测到 `transDevIbverbsDataMem_ != nullptr` → 填充 `devIpAddr_`（从 `rankInfoList_` 读取本 rank 的 NIC IP）→ 调用 `InitNic(true)`

---

## 7. 开发注意事项

- `hccl_communicator_host.cc` 极大（~470KB），修改时注意函数间的调用顺序依赖。
- `netDevCtxMap_` 是引用传递给 `CommFactory` 和 `TransportManager`，任何地方修改都会全局可见。
- `devIpAddr_` 在单机场景下默认为空（不做跨机通信），MC2 ibverbs 场景需手动从 `rankInfoList_` 填充。
- `commPortConfig_.devNicListen` 是预先监听的 socket（CANN 框架提前建立），若非空则优先用预监听路径，否则走 `HcclNetOpenDev` 路径。
- `[cwh]` 前缀的日志是诊断期间添加的，定位问题后可以清理。

---

## 8. 未来可扩展点

- **InitNic 提前调用**：将单机 ibverbs 场景的 `devIpAddr_` 填充 + `InitNic` 移到 `InitRaResource` 中，与正常路径统一，消除时序 bug。
- **Transport 抽象**：当前 P2P/ibverbs 双 transport 逻辑散落在 `SetCommResource` 中，可抽象为 `ITransportProvider` 接口统一管理。
- **资源复用**：`commLevel0Rdma` 每次 MC2 调用都重建，可缓存 transport 对象实现复用。
