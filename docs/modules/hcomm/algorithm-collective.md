# 模块：hcomm/algorithm-collective（集合通信执行器）

## 1. 模块作用

将集合通信算子（AllGather、AllReduce、AllToAll、Broadcast、Reduce、ReduceScatter、Scatter、Send/Recv 等）的执行逻辑具体化。每个算子有独立的 Executor 类，负责：根据拓扑选择具体算法模板、管理多 stream/多 ring 并发、协调 profiling 和 deterministic 模式。`AlgConfigurator` 负责全局算法配置，`TopoMatcher` 提供拓扑信息。

---

## 2. 目录结构

```
src/algorithm/impl/
├── coll_executor/                         # 集合通信执行器（核心）
│   ├── coll_comm_executor.cc/.h           # 多 ring AllReduce 执行器（135KB）
│   ├── coll_executor_base.cc/.h           # 执行器基类
│   ├── coll_native_executor_base.cc/.h    # 原生执行器基类（43KB）
│   ├── alg_profiling.cc                   # 算法 profiling
│   ├── registry/                          # 执行器注册表
│   │   └── executor_registry.cc/.h
│   ├── coll_all_gather/                   # AllGather 执行器
│   │   ├── coll_all_gather_executor.cc/.h
│   │   ├── coll_aligned_all_gather_double_ring_executor.cc/.h
│   │   ├── coll_all_gather_halving_doubling_executor.cc/.h
│   │   ├── coll_all_gather_mesh_executor.cc/.h
│   │   └── 310P/                          # 310P 特化
│   ├── coll_all_gather_v/                 # AllGatherV 执行器（30+ 文件）
│   ├── coll_all_reduce/                   # AllReduce 执行器
│   ├── coll_all_to_all/                   # AllToAll 执行器（40+ 文件）
│   │   ├── coll_all_to_all_mesh_aiv_executor.cc/.h
│   │   ├── coll_all_to_all_v_fullmesh_executor.cc
│   │   ├── coll_all_to_all_v_2level_pipeline_excecutor.cc/.h
│   │   └── 310P/
│   ├── coll_broadcast/                    # Broadcast 执行器
│   ├── coll_reduce/                       # Reduce 执行器
│   │   ├── coll_reduce_executor.cc/.h
│   │   ├── coll_reduce_ring_plus_hd_executor.cc/.h
│   │   ├── coll_reduce_mesh_executor.cc/.h
│   │   └── coll_reduce_ring_for_910_93_executor.cc/.h
│   ├── coll_reduce_scatter/               # ReduceScatter 执行器
│   ├── coll_reduce_scatter_v/             # ReduceScatterV 执行器（30+ 文件）
│   ├── coll_scatter/                      # Scatter 执行器
│   └── coll_send_receive/                 # Send/Recv 执行器
├── alg_configurator.cc/.h                # 算法配置器
├── alg_env_config.cc                     # 环境变量配置
├── topo_matcher.cc/.h                    # 拓扑匹配器（213 行头文件）
├── hccl_alg.cc/.h                        # 算法层主入口
└── legacy/
    └── hccl_impl.cc/.h                   # 算法执行器（含 CreateCommByAlg）
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `coll_comm_executor.cc` | `CollCommExecutor`：多 ring 并发 AllReduce 实现；`GetSubStreamInfoOnOneRing()`（获取单环 stream 信息）、`MultiRingAllReduce()`（多环并行 AllReduce） |
| `coll_native_executor_base.cc` | `CollNativeExecutorBase`（43KB）：所有原生执行器的基类，提供 stream 管理、notify 管理、profiling 挂载点 |
| `alg_configurator.h` | `AlgConfigurator`：为每种命令类型（AllReduce/AllGather/...）选择 Level0/Level1/Level2 算法；管理确定性模式、AIV 模式、AICPU 展开等开关 |
| `topo_matcher.h` | `TopoMatcher`（213 行）：持有全局拓扑视图，提供通信平面计算、rank 分组、bridge rank 查询；`HcclExternalEnable` 包含 FFTS/确定性/AIV 等特性开关 |
| `coll_all_to_all/coll_all_to_all_v_2level_pipeline_excecutor.cc` | 两级 pipeline AllToAllV：level0（片内）和 level1（跨机）并行执行 |
| `coll_reduce/coll_reduce_ring_plus_hd_executor.cc` | Ring + HD 混合 Reduce：大数据用 ring，小数据用 halving-doubling |

---

## 4. 核心函数 / 类 / 接口

### CollExecutorBase（`coll_executor_base.h`）

```cpp
class CollExecutorBase {
    // 构造
    CollExecutorBase(const HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher>& topoMatcher);

    // 配置（构造后必须调用 SetAlgType）
    void SetAlgType(AlgType);
    void SetCCLInBuffer(u64 cclbufferSize);
    void SetIsSupportSDMAReduce(bool);
    void SetAlgOpContext(AlgOpContext);
    void SetAivClearEnable(bool);
    void SetRmaInfo(void*);
    void SetNumBlocks(const u32&);
    void SetOpCounter(const OpCounterInfo&);

    // 纯虚接口（子类必须实现）
    virtual HcclResult CalcResRequest(const OpParam& param,
        AlgResourceRequest& resourceRequest) = 0;
    virtual HcclResult Orchestrate(OpParam& param,
        AlgResourceResponse& algRes) = 0;

    // 静态辅助
    static HcclResult RunTemplate(const std::unique_ptr<AlgTemplateBase>& tempAlg,
        const SubCommInfo& commInfo);
};
```

### CollNativeExecutorBase（`coll_native_executor_base.h`）

```cpp
// 常量
constexpr u64 HCCL_INPLACE_MEMCOPY_SIZE   = 131072; // 128 KB
constexpr u64 HCCL_POST_SYNC_MEMCOPY_SIZE = 131072; // 128 KB

// 内存上下文（每次 KernelRun 传入）
struct ExecMem {
    u64       count     = 0;
    DeviceMem inputMem;    // 单算子: InCCLMem;  图模式: InUserMem
    DeviceMem outputMem;   // 单算子: OutCCLMem; 图模式: OutUserMem
    DeviceMem scratchMem;  // scratch/temp buffer
    void*     inputPtr  = nullptr;
    void*     outputPtr = nullptr;
};

class CollNativeExecutorBase : public CollExecutorBase {
    // 资源计算（子类按需重写）
    virtual HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcLevel0CommInfo(TransportMemType, TransportMemType,
        std::vector<LevelNSubCommTransport>&);
    virtual HcclResult CalcLevel1CommInfo(TransportMemType, TransportMemType,
        std::vector<LevelNSubCommTransport>&);
    virtual HcclResult CalcLevel2CommInfo(TransportMemType, TransportMemType,
        std::vector<LevelNSubCommTransport>&);
    virtual HcclResult CalcStreamNum(u32& streamNum);
    virtual HcclResult CalcScratchMemSize(u64& scratchMemSize);
    virtual HcclResult CalcNotifyNum(u32 streamNum, u32& notifyNum);

    // 执行路径（子类实现）
    virtual HcclResult KernelRun(const OpParam& param, ExecMem& execMem);
    // 零拷贝三段式（可选重写）
    virtual HcclResult KernelRunIntraServerPre(const OpParam&, ExecMem&);
    virtual HcclResult KernelRunInterServer(const OpParam&, ExecMem&);
    virtual HcclResult KernelRunIntraServerPost(const OpParam&, ExecMem&);

    // 子流同步辅助
    HcclResult ActiveSlaveStreams(const Stream& stream);
    HcclResult NotifySubStreamStart(Stream&, std::vector<Stream>&, ...);
    HcclResult WaitSubStreamFinish(Stream&, std::vector<Stream>&, ...);
};
```

### TopoMatcher（`topo_matcher.h`）—— 参见 hcomm/topo.md 中的完整接口

```cpp
// HcclExternalEnable：运行时特性开关（完整字段）
using HcclExternalEnable = struct HcclExternalEnableDef {
    u32  enableFfts;        // FFTS 开关（默认 1）
    u32  deterministic;     // 确定性计算（默认 0）
    u32  intraRoceSwitch;   // 片内 RoCE（默认 0）
    u32  dumpDebug;         // 调试 dump（默认 0）
    u32  interHccsDisable;  // 禁用跨机 HCCS（默认 0）
    bool aivMode;           // AIV 模式（默认 false）
    bool aicpuUnfold;       // AICPU 展开（默认 false）
    bool isOnlyAiv;         // 仅 AIV 模式（默认 false）
    s32  execTimeOut;       // 执行超时
    std::map<HcclCMDType, std::vector<HcclAlgoType>> algoConfig; // 每算子算法配置
};

class TopoMatcher {
    HcclResult CalcCommPlaneInfo(const std::string& tag,
        const CommParaInfo& commParaInfo,
        std::vector<SingleSubCommTransport>& commTransport,
        TransportMemType inputMemType, TransportMemType outputMemType);

    // 特性开关（读/写）
    u32  GetExternalInputHcclEnableFfts();
    bool GetAivModeConfig() const;
    bool GetAicpuUnfoldConfig() const;
    HcclResult SetDeterministicConfig(u8 deterministic);
    HcclResult SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>&);
};
```

### CollAlgExecRegistry（`coll_alg_exec_registry.h`）—— 执行器注册表

```cpp
// 工厂函数类型
using CollExecCreator = std::function<
    CollExecutorBase*(const HcclDispatcher, std::unique_ptr<TopoMatcher>&)>;

// 默认模板工厂（静态方法）
template<typename P>
static CollExecutorBase* DefaultExecCreator(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher) {
    static_assert(std::is_base_of<CollExecutorBase, P>::value, ...);
    return new (std::nothrow) P(dispatcher, topoMatcher);
}

// 单例注册表
class CollAlgExecRegistry {
    static CollAlgExecRegistry& Instance();

    // 注册（重复 tag 返回 HCCL_E_INTERNAL）
    HcclResult Register(const std::string& tag, const CollExecCreator& creator);

    // 查找并实例化（tag 不存在返回 nullptr）
    std::unique_ptr<CollExecutorBase> GetAlgExec(const std::string& tag,
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
};

// 注册宏（在各 executor .cc 文件中使用）
// REGISTER_EXEC(tag, name, CollExecutorSubclass)
// 展开为文件作用域 static 变量，在程序启动时自动调用 Register()
#define REGISTER_EXEC(tag, name, collExecBase) \
    static HcclResult g_func_##name##_<N> = \
        CollAlgExecRegistry::Instance().Register(tag, DefaultExecCreator<collExecBase>)

// 使用示例：
// REGISTER_EXEC("AllReduce_Ring_A800", MyAllReduceExecutor, MyAllReduceExecutor);
```

### CollCommExecutor（`coll_comm_executor.h`）—— 多 Ring 执行器

```cpp
class CollCommExecutor : public CollNativeExecutorBase {
    // 多 ring AllReduce（带切片参数）
    HcclResult MultiRingAllReduce(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem,
        u64 count, HcclDataType dataType, HcclReduceOp reductionOp,
        const std::vector<std::vector<Slice>>& multRingsSliceZero,
        Stream stream, s32 profStage, u64 baseOffset = 0);

    // 多 ring ReduceScatter（支持并发版本）
    HcclResult MultiRingReduceScatter(...);
    HcclResult MultiRingReduceScatterConcurrent(...);
    HcclResult Level1ReduceScatterConcurrent(...);

    // 多 ring AllGather（支持并发版本）
    HcclResult MultiRingAllGather(...);
    HcclResult MultiRingAllGatherConcurrent(...);
    HcclResult Level1AllGatherConcurrent(...);

    // Mesh 多流 ReduceScatter
    HcclResult MultiStreamReduceScatterMesh(...);
    HcclResult MultiStreamReduceScatterMeshAtomic(...);

    // AnyPath 路径选择（基于 NIC 拓扑动态选 ring 顺序）
    std::vector<std::vector<u32>> GetRingsOrderForAnyPath(
        u32 ranksSize, TopoType topoType, std::vector<u32>& nicList);
    std::vector<std::vector<Slice>> AnyPathPrepareMultiRingSlice(
        const std::vector<Slice>& dataSegsSlice, const std::string& tag, ...);

    // Ring 顺序计算
    std::vector<std::vector<u32>> GetRingsOrderByTopoType(
        u32 ranksSize, TopoType topoType, std::vector<u32>& nicList);
    std::vector<std::vector<Slice>> PrepareMultiRingSlice(
        const std::vector<Slice>& dataSegsSlice, const std::string& tag, ...);

    // 拓扑查询辅助
    bool Is910BSingleMesh();
    bool NeedCreateSingleMeshPlane(bool isInlineReduce);
    u64  GetReduceAttr(DeviceMem&, DeviceMem&, HcclDataType, HcclReduceOp);
};
```

---

## 5. 数据流向

```
HCCL 公共 API（如 HcclAllReduce）
  │
  ▼
hcclImpl::RunCollective()
  ├── AlgConfigurator::SelectAlgType()    → 确定 AlgType
  ├── TopoMatcher::CalcCommPlaneInfo()    → 计算通信平面
  └── ExecutorRegistry::GetExecutor()    → 获取对应 Executor
        │
        ▼
  Executor::RunAsync()（如 CollCommExecutor）
    ├── AllocScratchMem()                 → 分配 scratch 内存
    ├── AttachProfiling()                 → 挂载 profiling
    ├── MultiRingAllReduce()              → 调度多 ring 并行
    │     └── AlgTemplateBase::RunAsync() → 具体算法执行
    └── 等待所有 stream 完成
```

---

## 6. 关键业务逻辑

### 算法分级配置
`AlgConfigurator` 为 Level0（片内）、Level1（跨机）、Level2（超节点间）分别配置算法：
- Level0：通常用 Ring 或 Halving-Doubling（片内带宽高，latency 低）
- Level1：通常用 Ring 或 Mesh（跨机带宽受限，需要高吞吐算法）
- Level2：通常用 Ring 或 Star（超节点间连接稀疏）

### 多 Ring 并发（MultiRingAllReduce）
910B 有多条 HCCS 链路，`CollCommExecutor` 可同时利用多个 ring（通常 2 个）并行执行 AllReduce，理论带宽翻倍。每个 ring 使用独立的 stream（sub-stream），通过 event/notify 同步。

### 310P 特化
`coll_*/310P/` 目录下是针对 310P3 推理芯片的特化执行器，310P3 不支持某些 HCCS 链路，算法选择和 stream 数量与 910B 不同。

### 两级 Pipeline AllToAllV
`coll_all_to_all_v_2level_pipeline_excecutor.cc` 将 level0（片内）和 level1（跨机）的 AllToAllV 流水化：在执行 level1 通信的同时，已完成 level0 的数据可以开始 level1 的 reduce 处理。

---

## 7. 开发注意事项

- 新增算子时，需在 `executor_registry.cc` 中注册新的 Executor，否则 `GetExecutor()` 返回空。
- `CollNativeExecutorBase` 的 scratch 内存生命周期由执行器管理，`RunAsync` 返回前必须释放，否则内存泄漏。
- 310P 特化执行器（`310P/` 目录）与 910B 执行器共享基类但不共享算法选择逻辑，修改基类时需同时测试两种芯片。
- `alg_env_config.cc` 中的环境变量配置覆盖默认算法选择，开发时注意环境变量的副作用。
- `MultiRingAllReduce` 的 ring 数量由拓扑决定，不可手动指定，需通过 `TopoMatcher` 查询。

---

## 8. 未来可扩展点

- **算子 Fusion**：将相邻的 AllReduce + AllGather 融合为单个执行器，减少通信轮次。
- **自适应算法**：在运行时根据实测带宽动态切换 Ring/HD 算法，而非静态配置。
- **异步执行器**：当前执行器同步等待 stream 完成，可改为 future/callback 模式支持异步流水。
- **更多 pipeline 模式**：将两级 pipeline 扩展到三级（L0/L1/L2），适应超节点场景。
