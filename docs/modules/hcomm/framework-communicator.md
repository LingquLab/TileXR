# 模块：hcomm/framework-communicator（通信器核心框架）

## 1. 模块作用

`HcclCommunicator` 是 hcomm 的核心对象，管理单个通信域（communicator）的完整生命周期：初始化、资源分配、算子执行、MC2 Context 构建、以及销毁。它是 framework 层的主入口，向上对接 HCCL 公共 API，向下协调 algorithm 层（hcclImpl）和 resource_manager 层（TransportManager、SocketManager）。

---

## 2. 目录结构

```
src/framework/communicator/impl/
├── hccl_communicator.h/.cc            # 基类接口 + 通用实现（146KB）
├── hccl_communicator_host.cc          # Host 侧主体实现（470KB，核心文件）
├── hccl_communicator_device.cc        # Device 侧特化实现
├── hccl_communicator_attrs.h/.cc      # 属性基类
├── hccl_communicator_attrs_host.cc    # Host 侧属性收集（rank 信息、拓扑、设备类型）
├── hccl_communicator_attrs_device.cc  # Device 侧属性收集
├── comm_topo_desc.h/.cc               # 拓扑描述对象
└── task_abort_handler.h/.cc           # 任务中止处理
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `hccl_communicator.h` | 定义 `RemoteRes`（远端资源）、`AicpuOpTiling`（AICPU tiling 配置）、`InitTask` 结构体；声明 `TileXrInit()`/`TileXrDeInit()`/`TileXrDaemon()` 等接口；常量：`COMM_MAX_WORK_SPACE_SIZE=16MB`、`CACHEMAP_MAXSIZE=65536` |
| `hccl_communicator_host.cc` | 包含所有关键路径：`Init()`、`InitRaResource()`、`InitNic()`、`CreateCommAndStreamRes()`、`SetCommResource()`、`Mc2CreateAndLaunchContext()`；是整个 hcomm 改动最频繁的文件 |
| `hccl_communicator_attrs_host.cc` | 收集并缓存 rank 信息（`rankInfoList_`）、服务器分组（`servRankInfo_`）、设备物理 ID、NIC 部署方式（`nicDeployment_`）等属性 |
| `task_abort_handler.cc` | 处理算子执行中止：清理 stream、释放资源、通知其他 rank |

---

## 4. 核心函数 / 类 / 接口

### 生命周期

```cpp
class HcclCommunicator {
    // =========================================================
    // Init 重载 1：从完整 RankTable_t 初始化（主进程/world group）
    // =========================================================
    HcclResult Init(HcclCommParams& params, const RankTable_t& rankTable);
    // 调用顺序（约 20 步）：
    //   InitCommParams → attrCollector_.Init → InitRankInfo → InitNetResource
    //   → InitDebug → InitNotifyManager → InitStreamManager → InitProfiler
    //   → InitDispatcher → InitTransportManager → InitMemoryManager
    //   → InitCombinOpara → RegisterRanksToDca → RegistTaskExceptionHandler
    //   → InitPara → RegisterKernel → LoadAICPUKernel → LoadCustomKernel
    //   → InitHDCommunicate → InitOpRetry → InitOpResPara
    //   → InitOneSidedService → rankGraph_.Init → SaveTopoDesc → RegisterToSnapshot

    // =========================================================
    // Init 重载 2：从 rankList + WorldGroupInfo 初始化（子组/SubGroup）
    // =========================================================
    HcclResult Init(HcclCommParams& params,
        const std::vector<RankInfo>& rankList, WorldGroupInfo& groupCommonData);
    // 调用顺序（17 步，SubGroup 场景）：
    //   InitCommParams → attrCollector_.Init(SubGroup) → InitRankInfoSubGroup
    //   → InitDebugSubGroup → InitNotifyManager → InitDispatcher
    //   → InitStreamManager → InitRaResource → InitTransportManager
    //   → InitMemoryManagerSubGroup → InitHcclAlg → LoadAICPUKernel
    //   → LoadCustomKernel → InitHDCommunicate → InitOpRetry
    //   → InitOpResPara → RegisterRanksToDca
    //   → OrderLaunch::GetInstance(deviceLogicId_).RegisterOrderLaunch
    //   → rankGraph_.Init → SaveTopoDesc → RegisterToSnapshot

    // 销毁
    HcclResult Destroy();

    // TileXR 专用接口
    HcclResult TileXrInit(const HcclTopoInfo& topoInfo);
    HcclResult TileXrDeInit();
    HcclResult TileXrDaemon();   // 后台守护线程
};
```

### 网卡初始化

```cpp
// isMC2ReInit=true 时绕过单机早退逻辑
HcclResult InitNic(bool isMC2ReInit = false);
// 早退条件：!IntraRoceSwitch && servRankInfo_.size()==1 && isDiffDeviceModule_ && !isMC2ReInit
// 主流程：HcclNetOpenDev(NicType, ...) → netDevCtxMap_[ip] = ctx → socketManager_->ServerInit
// 完成后设置：isNeedInitNic_ = true; attrCollector_.SetNeedInitNicFlag(true); nicInitialized_++;
// DeinitNic：只在 nicInitialized_ - 1 <= 0 时才真正关闭 netDevCtx
```

### Rank 到 DCA 注册

```cpp
HcclResult RegisterRanksToDca();
// 最后一步调用 DetectConnectionAnomalies::GetInstance(deviceLogicId_).Init(rankInfoList_, isNeedInitNic_)
// 启动连接异常检测（仅在 isNeedInitNic_=true 时激活 ibverbs CQE 监听）
```

### MC2 资源创建

```cpp
// CreateCommResource：检测 tag 后缀判断是否多机
HcclResult CreateCommResource(const std::string& tag, ...);
// tag 含 HCCL_MC2_MULTISERVER_SUFFIX → isA2MC2MultiServer_ = true
// 单机 ibverbs 场景：调用 InitNic(isMC2ReInit=true) → CreateCommAndStreamRes → Mc2CreateAndLaunchContext
```

### MC2 双传输通道资源

```cpp
// 填充 P2P window 和 ibverbs data（双通道）
HcclResult SetCommResource(HcclA2CombineOpParam& param);
  ├── commLevel0（P2P）→ GetTransportByRank(i) → windowsIn/Out
  └── commLevel0Rdma（ibverbs）→ GetTransportByRank(i) → transDevIbverbsDataMem_

// 构建 kernel Context 并 launch AICPU 内核
HcclResult Mc2CreateAndLaunchContext(HcclA2CombineOpParam& param);
  ├── InitWorkSpace()
  ├── AICPU notify 初始化
  ├── 多机：H2D 拷贝 windowsIn/Out
  ├── 单机：H2D 拷贝 transDevIbverbsDataMem_ → combinOpara.ibverbsData
  ├── combinOparaPtr->tileXrContext = reinterpret_cast<u64>(tileXrCtxDev_.ptr())
  └── AiCpuKernelLaunch(stream, commContext_, "RunAicpuKfcResInit")
      // RunAicpuKfcResInitV2 用于部分场景
```

### 关键成员变量

```cpp
// 网络资源
std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap_;  // IP → NIC 上下文
std::vector<HcclIpAddress> devIpAddr_;                  // 本 rank 的 NIC IP 列表
bool isNeedInitNic_ = false;           // InitNic 是否已被调用
int  nicInitialized_ = 0;             // NIC 初始化引用计数

// 通信域
CommInfo commInfo_;                    // 持有 commLevel0 / commLevel0Rdma / commLevel1
std::unique_ptr<hcclImpl> hcclImpl_;   // 算法执行器

// MC2 双通道
TransportDeviceNormalData* transDevIbverbsDataMem_; // ibverbs transport 数据（host 侧）

// 属性
std::vector<RankInfo> rankInfoList_;   // 全局 rank 信息
std::map<std::string, std::vector<RankInfo>> servRankInfo_; // 服务器 → rank 分组
NICDeployment nicDeployment_;          // NIC 部署方式（DEVICE / HOST）
bool isA2MC2MultiServer_;              // 是否多机 MC2
bool isDiffDeviceModule_;              // 是否异构设备模块
```

---

## 5. 数据流向

```
HcclCommunicator::Init()
  │
  ├── InitRaResource()
  │     ├── HcclNetInit()              → 初始化网络子系统
  │     ├── InitSocketManager()        → 创建 HcclSocketManager
  │     └── InitNic()                  → 填充 netDevCtxMap_
  │
  ├── InitHcclAlg()
  │     └── hcclImpl::Init()           → 算法层初始化
  │
  └── CreateCommAndStreamRes()（MC2 场景）
        ├── hcclImpl::CreateCommByAlg()
        │     ├── commLevel0 线程（P2P）
        │     └── commLevel0Rdma 线程（ibverbs，forceRdma=true）
        ├── SetCommResource()
        │     ├── P2P window 地址填充
        │     └── ibverbs rkey/addr 填充
        └── Mc2CreateAndLaunchContext()
              └── H2D 拷贝 → kernel 可见的 combinOpara
```

---

## 6. 关键业务逻辑

### InitNic 调用时序（关键约束）
`InitNic` 必须在 `hcclImpl::CreateCommByAlg` 之前完成，否则 `netDevCtxMap_` 为空，导致 `TransportManager::CreateDestSockets` 中 `netDevCtx=nullptr`，引发 `hccl_socket_manager.cc:345` crash。

当前修复：`CreateCommAndStreamRes` 检测到 `transDevIbverbsDataMem_ != nullptr`（单机 ibverbs 场景）时，从 `rankInfoList_` 填充 `devIpAddr_` 并调用 `InitNic(true)`。

### 双传输通道（Dual Transport）
MC2 单机场景同时维护两个通信域：
- `commLevel0`（P2P/SDMA）：提供 `windowsIn/Out`（直接内存访问）
- `commLevel0Rdma`（ibverbs）：提供 `ibverbsData`（RDMA 传输地址和 rkey）

两者在 `SetCommResource` 中并行填充，在 `Mc2CreateAndLaunchContext` 中一起 H2D 拷贝到 kernel 可见的 `combinOpara`。

### AICPU 模式
`hccl_communicator.h` 中的 `AicpuOpTiling` 结构体和 `LoadAICPUKernel()` 支持算子在 AICPU 上执行（而非 AICore），适用于不支持 AICore 的场景或调试场景。

---

## 7. 开发注意事项

- `hccl_communicator_host.cc` 约 470KB，修改时务必先定位函数位置（用 `grep -n "函数名"`），避免在错误位置修改。
- `netDevCtxMap_` 是引用传递给 `CommFactory` 和 `TransportManager`，任何地方的修改全局可见。
- `devIpAddr_` 在单机场景默认为空，MC2 ibverbs 场景需手动从 `rankInfoList_` 填充（当前在 `CreateCommAndStreamRes` 中处理）。
- `[cwh]` 前缀的诊断日志是临时添加的，定位问题后应清理。
- `commPortConfig_.devNicListen` 是预监听 socket（CANN 框架提前建立），非空时优先使用，否则走 `HcclNetOpenDev` 路径。

---

## 8. 未来可扩展点

- **InitNic 提前调用**：将单机 ibverbs 场景的 `devIpAddr_` 填充 + `InitNic` 移到 `InitRaResource` 中，与正常路径统一，消除时序 bug。
- **Transport 抽象**：P2P/ibverbs 双 transport 逻辑可抽象为 `ITransportProvider` 接口，统一管理。
- **Communicator 池化**：当前每次 MC2 调用都重建 `commLevel0Rdma`，可增加 tag-based 缓存实现复用。
- **异步 Init**：`Init()` 目前是同步阻塞的，可改为异步模式，允许多个 communicator 并行初始化。
