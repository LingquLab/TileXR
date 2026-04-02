# 模块：hcomm/algorithm-comm（通信域与 CommFactory）

## 1. 模块作用

负责通信域（`CommBase` 及其子类）的创建、拓扑映射和 transport 绑定。`CommFactory` 是核心工厂类，根据 `CommParaInfo`（通信平面、拓扑类型、`forceRdma` 标志）创建对应的通信域对象（Ring/Mesh/Star/HD/P2P），并将 `netDevCtxMap_` 和 `rankInfoList_` 注入到每个通信域中。

---

## 2. 目录结构

```
src/algorithm/base/communicator/legacy/
├── comm_factory.cc/.h         # CommFactory 实现（核心工厂）
├── comm_factory_pub.h         # CommFactory 类声明（含 GetNetDevCtxMapSize）
├── comm_base.cc/.h            # CommBase 基类（所有通信域的父类）
├── comm_star.cc/.h            # Star 拓扑通信域
└── （其他拓扑实现文件）

src/algorithm/impl/legacy/
├── hccl_impl.cc/.h            # hcclImpl：算法执行器，管理通信域线程
└── （其他 impl 文件）
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `comm_factory.cc` | `CreateCommPlane()`：根据 `CommParaInfo` 选择拓扑类型并创建通信域；`GetIsUsedRdma()`：决定是否走 RDMA（含 `forceRdma` 修复） |
| `comm_factory_pub.h` | `CommFactory` 类声明：构造函数注入 `netDevCtxMap_`、`rankVector_`、`nicDeployment_` 等；`GetNetDevCtxMapSize()` 诊断接口 |
| `comm_base.cc` | `CommBase` 基类：持有 `netDevCtxMap_`、`paraVector_`（rank 信息）、`interSocketManager_`；`Init()` 调用 `TransportManager` 建立 socket |
| `comm_star.cc` | Star 拓扑：以 rank 0 为中心，其他 rank 直连 rank 0；适用于 Broadcast/Reduce |
| `hccl_impl.cc` | `CreateCommByAlg()`：并行启动 `commLevel0`/`commLevel0Rdma`/`commLevel1` 线程；`isA2MC2NeedIbverbs_` 控制单机 ibverbs 分支 |

---

## 4. 核心函数 / 类 / 接口

### CommFactory

```cpp
class CommFactory {
    // 构造：注入所有依赖
    CommFactory(const std::string& identifier, u32 userRank, u32 userRankSize,
        HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool>& notifyPool,
        std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap,  // 引用传递
        std::shared_ptr<TopoInfoExtractor> topoInfoEx,
        bool isUsedRdmaLevel0 = false,
        TopoType topoFlag = TOPO_TYPE_COMMON,
        DevType deviceType = DEV_TYPE_910,
        std::vector<RankInfo> rankVector = {},
        NICDeployment nicDeployment = NIC_DEPLOYMENT_DEVICE,
        ...);

    // 创建单层通信域（核心接口）
    HcclResult CreateCommPlane(const std::string& tag,
        const DeviceMem& inputMem, const DeviceMem& outputMem,
        const CommParaInfo& commParaInfo,
        std::vector<std::unique_ptr<CommBase>>& commVec,
        DeviceMem expMem = DeviceMem());

    // 诊断接口（cwh 新增）
    size_t GetNetDevCtxMapSize() const { return netDevCtxMap_.size(); }

    // 辅助查询
    u32 GetSubRootUserRank(u32 userRank, u32 rootUserRank);
    u32 GetLevel1CommRank(u32 ringIdx);

private:
    // RDMA 路径决策（已修复：含 forceRdma）
    HcclResult GetIsUsedRdma(const CommParaInfo& commParaInfo, bool& isUsedRdma);
    // isUsedRdma = isInterSuperPod || (isInterServer && !isUsedInterHccsMode_)
    //           || (isConnectedWithPcie && isUsedRdmaLevel0_)
    //           || commParaInfo.forceRdma;  ← 修复点

    // 各拓扑创建函数
    HcclResult CreateCommRing(...);   // Ring 拓扑
    HcclResult CreateCommHD(...);     // Halving-Doubling 拓扑
    HcclResult CreateCommStar(...);   // Star 拓扑
    HcclResult CreateCommMesh(...);   // Mesh 拓扑
    HcclResult CreateCommP2P(...);    // P2P 拓扑

    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap_;  // 引用，非拷贝
};
```

### CommParaInfo

```cpp
struct CommParaInfo {
    CommPlane commPlane;         // COMM_LEVEL0 / COMM_LEVEL1 / COMM_LEVEL2
    CommType  commType;          // RING / MESH / STAR / HD / P2P / ...
    u32       root;              // Broadcast/Reduce 的 root rank
    bool      forceRdma = false; // 强制 RDMA（MC2 双通道时设为 true）
    bool      isAicpuModeEn;
    bool      meshSinglePlane;
};
```

### hcclImpl

```cpp
class hcclImpl {
    // 根据算法类型创建所有通信域（并行线程）
    HcclResult CreateCommByAlg(const std::string& tag,
        const AlgType algType, CommInfo& commInfo,
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& expMem,
        u32 root, bool isAicpuModeEn, bool meshSinglePlane, bool isA2MC2MultiServer);

    // 单机 ibverbs 控制标志（由上层设置）
    bool isA2MC2NeedIbverbs_ = false;

    // 通信域线程指针
    std::unique_ptr<std::thread> commThreadPtrLevel0_;
    std::unique_ptr<std::thread> commThreadPtrLevel0Rdma_;
    std::unique_ptr<std::thread> commThreadPtrLevel1_;
};
```

---

## 5. 数据流向

```
HcclCommunicator::InitHcclAlg()
  └── hcclImpl::Init()
        └── CommFactory 初始化（注入 netDevCtxMap_、rankVector_ 等）

算子执行时
  hcclImpl::CreateCommByAlg()
    │
    ├── commLevel0 线程
    │     └── CommFactory::CreateCommPlane(forceRdma=false)
    │           └── GetIsUsedRdma() → isUsedRdma=false（单机 P2P）
    │                 └── CreateCommRing/Mesh/...
    │                       └── CommBase::Init()
    │                             └── TransportManager::CreateDestSockets(forceRdma=false)
    │
    └── commLevel0Rdma 线程（若 needLevel0Rdma）
          └── CommFactory::CreateCommPlane(forceRdma=true)
                └── GetIsUsedRdma() → isUsedRdma=true（因 forceRdma）
                      └── CreateCommRing/Mesh/...
                            └── CommBase::Init()
                                  └── TransportManager::CreateDestSockets(forceRdma=true)
                                        └── UpdateIsInterRdma → isInterRdma=true
                                              └── netDevCtxMap_[devIpAddr_[0]]
```

---

## 6. 关键业务逻辑

### GetIsUsedRdma 修复
原始逻辑只考虑跨机/PCIe 场景，忽略 `forceRdma`。修复后：

```cpp
isUsedRdma = (isInterSuperPod) ||
             (isInterServer && !isUsedInterHccsMode_) ||
             (isConnectedWithPcie && isUsedRdmaLevel0_) ||
             commParaInfo.forceRdma;  // ← 新增，保证单机 MC2 ibverbs 正确创建
```

### 通信域线程并行创建
`commLevel0` 和 `commLevel0Rdma` 线程并行启动，通过 `WaitCommThread()` 同步等待。若任一线程失败，`CreateCommByAlg` 返回错误码。

### 拓扑类型与通信域映射

| 拓扑类型 | 适用场景 | 创建函数 |
|---------|---------|---------|
| Ring | AllReduce（大数据量） | `CreateCommRing` |
| Halving-Doubling (HD) | AllReduce（小数据量） | `CreateCommHD` |
| Mesh | AllToAll、AllGather | `CreateCommMesh` |
| Star | Broadcast、Reduce | `CreateCommStar` |
| P2P | Send/Recv | `CreateCommP2P` |

### CommBase 中的 socket 建立
`CommBase::Init()` 调用 `interSocketManager_->CreateSockets()`，传入 `netDevCtxMap_[paraVector_[rank_].nicIp[0]]` 作为本端 NIC 上下文。若 `nicIp` 为空或 `netDevCtxMap_` 中无对应条目，返回 `nullptr`。

---

## 7. 开发注意事项

- `CommFactory` 持有 `netDevCtxMap_` 的引用，`HcclCommunicator` 中对该 map 的修改对 `CommFactory` 立即可见，反之亦然。
- `commParaInfo.forceRdma = true` 是 MC2 双通道的关键标志，创建 `commLevel0Rdma` 时必须设置，否则退化为 P2P transport。
- `hcclImpl::isA2MC2NeedIbverbs_` 必须在 `CreateCommByAlg` 调用前被设置为 `true`，否则 `commLevel0Rdma` 不会创建。
- 修改 `CommParaInfo` 结构体时，需检查所有传递点：`CreateCommPlane` → `CommBase` 构造 → `TransportManager`。

---

## 8. 未来可扩展点

- **Transport 缓存**：`CommBase` 每次创建都重建 transport，可增加 tag-based 缓存，同 tag 复用已建立的 transport。
- **拓扑自适应**：`CommFactory` 目前根据静态配置选择拓扑，可扩展为根据实时网络状态动态选择最优拓扑。
- **多级 ibverbs**：当前只有 level0 Rdma，`CommInfo` 中已有 `commLevel1Rdma` 字段，可扩展多机多级场景。
