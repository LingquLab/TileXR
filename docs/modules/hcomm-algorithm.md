# 模块：hcomm-algorithm（算法层）

## 1. 模块作用

HCCL 算法层，负责：
- 集合通信算子（AllReduce、AllGather、ReduceScatter 等）的算法选择与执行
- `CommFactory`：根据拓扑类型和 `CommParaInfo` 创建各层通信域（CommBase）
- `hcclImpl`：算法执行器，管理 `commLevel0`/`commLevel1`/`commLevel0Rdma` 线程创建
- Transport 层对接（P2P vs ibverbs 路径选择）

---

## 2. 目录结构

```
src/algorithm/
├── base/
│   ├── alg_template/          # 集合通信算法模板（ring、halving-doubling 等）
│   ├── alg_aiv_template/      # AIV 算法模板
│   ├── communicator/
│   │   └── legacy/
│   │       ├── comm_factory.cc/.h       # CommFactory 实现
│   │       ├── comm_factory_pub.h       # CommFactory 类声明（含 GetNetDevCtxMapSize）
│   │       ├── comm_base.cc/.h          # CommBase 基类
│   │       └── comm_star.cc             # Star 拓扑通信域
│   ├── mc2_handler/           # MC2 算子处理器
│   └── inc/
├── impl/
│   ├── legacy/
│   │   ├── hccl_impl.cc/.h    # hcclImpl：算法执行器主体
│   │   └── hccl_impl.h
│   ├── coll_executor/         # 各算子的执行器
│   │   ├── coll_all_gather/
│   │   ├── coll_all_reduce/
│   │   ├── coll_reduce_scatter/
│   │   └── ... (共 11 个算子目录)
│   ├── resource_manager/
│   │   └── hccl_socket_manager.cc/.h   # Socket 管理
│   ├── alg_configurator.cc/.h
│   └── topo_matcher.cc/.h
└── pub_inc/
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `base/communicator/legacy/comm_factory.cc` | `CommFactory::CreateCommPlane()` 根据 `CommParaInfo`（含 `forceRdma`）创建通信域；`GetIsUsedRdma()` 决定是否走 RDMA |
| `impl/legacy/hccl_impl.cc` | `CreateCommByAlg()`：并行启动 `commLevel0`/`commLevel0Rdma`/`commLevel1` 线程；`isA2MC2NeedIbverbs_` 控制单机 ibverbs 分支 |
| `impl/resource_manager/hccl_socket_manager.cc` | `CreateSingleLinkSocket`、socket 生命周期，Line 345：`HcclNetDevGetNicType(netDevCtx, ...)` |

---

## 4. 核心函数 / 类 / 接口

### `CommFactory`（`comm_factory_pub.h` + `comm_factory.cc`）

```cpp
// 创建单层通信域
HcclResult CreateCommPlane(const std::string& tag,
    const DeviceMem& inputMem, const DeviceMem& outputMem,
    const CommParaInfo& commParaInfo,
    std::vector<std::unique_ptr<CommBase>>& commVec);

// 诊断接口（cwh 新增）
size_t GetNetDevCtxMapSize() const { return netDevCtxMap_.size(); }

// 内部：是否使用 RDMA transport
// commParaInfo.forceRdma = true 时强制返回 true
HcclResult GetIsUsedRdma(const CommParaInfo& commParaInfo, bool& isUsedRdma);
```

### `CommParaInfo`

```cpp
struct CommParaInfo {
    CommPlane commPlane;         // COMM_LEVEL0 / COMM_LEVEL1 / ...
    CommType  commType;          // RING / MESH / STAR / ...
    bool      forceRdma = false; // 强制 RDMA（MC2 双通道时设为 true）
    bool      isAicpuModeEn;
    // ...
};
```

### `hcclImpl`（`hccl_impl.cc`）

```cpp
// 根据算法类型创建所有通信域
HcclResult CreateCommByAlg(const std::string& tag,
    const AlgType algType, CommInfo& commInfo,
    DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& expMem,
    u32 root, bool isAicpuModeEn, bool meshSinglePlane, bool isA2MC2MultiServer);

// 单机 ibverbs 控制标志
bool isA2MC2NeedIbverbs_ = false;  // 由上层设置，触发 commLevel0Rdma 创建
```

### `CreateCommByAlg` 核心决策

```cpp
// 是否需要 ibverbs 通道
bool needLevel0Rdma = isUsedRdma              // 多机 RDMA
    || (!isA2MC2MultiServer && isA2MC2NeedIbverbs_);  // 单机 MC2 ibverbs

if (needLevel0Rdma) {
    commInfoLevel0.forceRdma = true;
    // 启动 commLevel0Rdma 线程
}
```

---

## 5. 数据流向

```
HcclCommunicator::InitHcclAlg()
  └── hcclImpl 初始化
        └── CreateCommByAlg()
              ├── commLevel0 线程（P2P transport）
              │     └── CommFactory::CreateCommPlane(forceRdma=false)
              │           └── CommBase 初始化 → TransportManager
              │                 └── CreateDestSockets(forceRdma=false)
              │                       → P2P transport
              │
              └── commLevel0Rdma 线程（ibverbs transport，若 needLevel0Rdma）
                    └── CommFactory::CreateCommPlane(forceRdma=true)
                          └── GetIsUsedRdma() → isUsedRdma=true（因 forceRdma）
                                └── TransportManager
                                      └── CreateDestSockets(forceRdma=true)
                                            └── UpdateIsInterRdma → isInterRdma=true
                                                  └── netDevCtxMap_[devIpAddr_[0]]
                                                        → HcclNetDevGetNicType(netDevCtx)
```

---

## 6. 关键业务逻辑

### `GetIsUsedRdma` 修改点
原始逻辑只考虑跨机/PCIe，不处理 `forceRdma`。已添加 `|| commParaInfo.forceRdma` 使单机 MC2 场景正确创建 ibverbs transport：

```cpp
isUsedRdma = (isInterSuperPod) ||
             (isInterServer && !isUsedInterHccsMode_) ||
             (isConnectedWithPcie && isUsedRdmaLevel0_) ||
             commParaInfo.forceRdma;  // ← 新增
```

### CommLevel0Rdma 线程等待
`commLevel0Rdma` 线程与 `commLevel0` 线程并行创建，但都通过 `WaitCommThread()` 同步等待完成。若任一线程失败，`CreateCommByAlg` 返回错误码。

### Transport 类型选择（`UpdateIsInterRdma`）
```cpp
// 单机场景下
isInterRdma = (isUsedRdmaLevel0_ && linkType == PXI_TYPE) || forceRdma;
```
`forceRdma=true` 时，即使是单机也走 ibverbs 路径（PXI_TYPE）。

---

## 7. 开发注意事项

- `comm_factory.cc` 中 `GetIsUsedRdma` 的修改是 `|| commParaInfo.forceRdma` 这一行，不可删除，否则单机 ibverbs transport 退化为 P2P。
- `hccl_socket_manager.cc:345`（`HcclNetDevGetNicType(netDevCtx, ...)`）在 `netDevCtx=nullptr` 时会 crash，根本原因是 `netDevCtxMap_` 未初始化。
- `isA2MC2NeedIbverbs_` 必须在 `CreateCommByAlg` 调用前被设置为 `true`，否则 `commLevel0Rdma` 不会创建。
- 修改 `CommParaInfo` 结构体时注意该结构体有多个传递点（`CreateCommPlane` → `CommBase` 构造 → `TransportManager`），需全链路检查。

---

## 8. 未来可扩展点

- **Transport 工厂抽象**：当前 `GetIsUsedRdma` 是硬编码规则，可重构为策略模式，不同拓扑注册不同策略。
- **通信域缓存**：`commLevel0Rdma` 每次 MC2 操作重建，可增加 tag-based 缓存，同 tag 复用已建立的 transport。
- **多级 ibverbs**：当前只有 level0 Rdma，未来多机多级场景可扩展为 `commLevel1Rdma`（`CommInfo` 中已有 `commLevel1Rdma` 字段）。
