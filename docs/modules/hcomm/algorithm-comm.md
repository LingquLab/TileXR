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

### CommBase（`comm_base_pub.h`）

```cpp
// 常量
constexpr u32 MC2_PLANE_MODE_HOST      = 0; // HOST 调度 RoCE
constexpr u32 MC2_PLANE_MODE_COMBINE   = 1; // 非层次链路模式
constexpr u32 MC2_PLANE_MODE_HIERARCHY = 2; // 层次链路模式

class CommBase {
    // 构造：注入所有依赖（collectiveId, userRank, rankSize, paraVector,
    //   topoFlag, dispatcher, notifyPool, netDevCtxMap, exchanger,
    //   inputMem, outputMem, isUsedRdmaLevel0, tag, nicDeployInner, ...)
    explicit CommBase(...);

    // 生命周期
    virtual HcclResult Init();
    virtual HcclResult DeInit();

    // Transport 访问
    std::shared_ptr<Transport> GetTransportByRank(u32 dstRank);
    HcclResult GetRankByUserRank(u32 userRank, u32& rank);
    HcclResult GetUserRankByRank(u32 rank, u32& userRank);

    // 异步建连
    HcclResult BuildAsync(u32& status);
    HcclResult BuildQuerry(u32& status);

    // 算法执行
    HcclResult RunTemplateAlg(std::unique_ptr<AlgTemplateBase>& tempAlg);
    HcclResult RunTemplateAlgStaged(std::unique_ptr<AlgTemplateBase>&, u32 stage);

    // MC2 支持查询
    HcclResult IsSupportMC2(const std::string& tag);

    // 中断所有连接
    void Break();

    // 关键成员（protected）
    std::vector<std::shared_ptr<Transport>> transportInfo_;  // 每 peer rank 一个 transport
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap_;  // 引用，非拷贝
    std::vector<RankInfo> paraVector_;                       // 子通信域 rank 信息
    std::shared_ptr<Transport> linkDummy_;                   // 无效 rank 查询时返回
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

### hcclImpl（`hccl_impl.h`）

```cpp
// 关键常量
constexpr u32 LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE = 2;  // 双 ring 场景 level0 平面数
constexpr u32 RDMA_PLANE_NUM_IN_NPRING_DOUBLE   = 2;  // 双 ring RDMA 平面数
constexpr f32 BASE_COMM_LATENCY = 13.0f;              // 基础通信延迟(μs)

// Pipeline 切片信息
struct PiplineSliceInfo {
    std::vector<Slice>              piplineDataSegsSlice;    // 每 stage 数据切片
    std::vector<std::vector<Slice>> piplineMultiStreamSlice; // 每 stream 每 stage 切片
    u64 count  {0};
    u64 offset {0};
};

class hcclImpl {
    // 初始化
    HcclResult Init(bool isHeterogComm = false);

    // 根据算法类型并行创建所有通信域（commLevel0/Rdma/Level1 各一线程）
    HcclResult CreateCommByAlg(const std::string& tag,
        const AlgType algType, CommInfo& commInfo,
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& expMem,
        u32 root, bool isAicpuModeEn, bool meshSinglePlane, bool isA2MC2MultiServer);

    // 多流资源创建（两个重载）
    HcclResult CreateMutiStreamRes(const std::string& tag, Stream& stream,
        const AlgType& algType, ...);
    HcclResult CreateMutiStreamRes(const std::string& tag, Stream& stream,
        level1StreamInfo_t& streamInfo, const AlgType& algType, ...);

    // 创建并缓存通信域（tag-based）
    HcclResult CreateComm(const std::string& tag,
        DeviceMem& inputMem, DeviceMem& outputMem,
        const AlgType& algType, CommInfo& commInfo, ...);

    // 释放 tag 对应的所有资源
    HcclResult ClearOpResource(const std::string& tag);

    // 中断所有通信
    void Break();

    // 检查 tag 是否已有通信资源
    bool IsExistCommRes(const std::string& tag);

    // 关键成员
    bool isA2MC2NeedIbverbs_ = false;    // 单机 ibverbs 控制标志（上层设置）
    u64  piplineSliceNum_ = 0;           // 流水线切片数（0=关闭，1=无流水，N=流水）
    bool isAlltoAllZCopyMode_ = false;   // 全局零拷贝 AllToAll 标志

    // 通信域线程（并行建连）
    std::unique_ptr<std::thread> commThreadPtrLevel0_;
    std::unique_ptr<std::thread> commThreadPtrLevel0Rdma_;
    std::unique_ptr<std::thread> commThreadPtrLevel1_;
    std::unique_ptr<std::thread> commThreadPtrLevel2_;

    // per-tag 通信域缓存（由 commLock_ 保护）
    tagCommInfo_t tagCommInfo_;
    std::mutex commLock_;

    // scratch 内存
    std::map<std::string, DeviceMem> scratchMemMap_;
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
