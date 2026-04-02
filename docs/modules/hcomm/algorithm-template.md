# 模块：hcomm/algorithm-template（算法模板层）

## 1. 模块作用

提供所有集合通信算法的可复用模板实现，分为两类：
- **`alg_template/`**：传统 AIC（AI Core）算法模板，基于 ring/halving-doubling/mesh/bruck 等经典拓扑，通过 `AlgTemplateBase` 基类统一接口
- **`alg_aiv_template/`**：AIV（AI Vector）算法模板，直接在 AICore 的 vector 单元上执行通信逻辑，延迟更低，适合小数据量场景

两类模板均按算子类型（AllGather/AllReduce/AllToAll/Broadcast/Reduce/ReduceScatter/Scatter/SendRecv）组织子目录。

---

## 2. 目录结构

```
src/algorithm/base/
├── alg_template/                          # AIC 算法模板（360KB+）
│   ├── alg_template_base.cc/.h            # 模板基类（44KB）
│   ├── alg_template_base_pub.h            # 公共基类接口（37KB）
│   ├── alg_template_multi_deter_pipeline.cc/.h  # 多确定性 pipeline 模板（24KB）
│   ├── alg_template_register.cc/.h        # 模板注册机制
│   ├── asymmetric_hierarchical_concatenate_alg_template_base.cc  # 非对称层次拼接（23KB）
│   ├── asymmetric_hierarchical_concatenate_base.cc               # 基础实现（66KB）
│   ├── nonuniform_bruck_base.cc            # 非均匀 Bruck 算法
│   ├── nonuniform_hierarchical_ring_base.cc # 非均匀层次 Ring
│   ├── recursive_halvingdoubling_base.cc   # 递归 Halving-Doubling
│   ├── component/
│   │   ├── reducer.cc/.h                  # Reduce 组件（14KB）
│   │   └── sender.cc/.h                   # Send 组件
│   ├── temp_all_gather/                   # AllGather 模板（30+ 文件，416KB）
│   ├── temp_all_reduce/                   # AllReduce 模板
│   ├── temp_alltoall/                     # AllToAll 模板
│   ├── temp_alltoallv/                    # AllToAllV 模板
│   ├── temp_broadcast/                    # Broadcast 模板
│   ├── temp_gather/                       # Gather 模板
│   ├── temp_reduce/                       # Reduce 模板
│   ├── temp_reduce_scatter/               # ReduceScatter 模板
│   ├── temp_scatter/                      # Scatter 模板
│   └── temp_send_recv/                    # SendRecv 模板
│
└── alg_aiv_template/                      # AIV 算法模板（812KB）
    ├── aiv_communication_base.h            # AIV 通信基类（45KB）
    ├── aiv_communication.h                 # AIV 通信接口（14KB）
    ├── aiv_crossnode_91093_base.h          # 91093 跨节点 AIV 基类（33KB）
    ├── aiv_npu_direct_base.h               # NPU Direct 基类
    ├── aiv_all_gather_910b_bigdata.h       # 910B AllGather 大数据量
    ├── aiv_all_gather_910b_smalldata.h     # 910B AllGather 小数据量
    ├── aiv_all_reduce_910b_bigdata.h       # 910B AllReduce 大数据量
    ├── aiv_all_reduce_910b_rdma_smalldata.h # 910B AllReduce RDMA 小数据量
    ├── aiv_all_reduce_deter_910b_*.h       # 910B 确定性 AllReduce（多文件）
    ├── aiv_all_to_all_910b_direct_fullmesh.h # 910B AllToAll 全连接
    ├── aiv_reduce_scatter_910b_*.h         # 910B ReduceScatter（多文件）
    ├── aiv_sync_910b.h / aiv_sync_910b_rdma.h  # 910B 同步原语
    ├── aiv_*_superkernel.h                 # Superkernel 实现（AllGather/AllReduce/AllToAll/ReduceScatter）
    ├── aiv_sk_*_crossnode.h                # 跨节点 Superkernel
    └── aiv_interface/
        └── sync_interface.h               # AIV 同步接口
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `alg_template_base_pub.h` | 所有 AIC 模板的公共基类接口（37KB）：定义 `RunAsync()`、`GetOutputSize()`、`CalcScratchMemSize()` 等纯虚函数 |
| `alg_template_multi_deter_pipeline.cc` | 多确定性 pipeline 模板：保证多次运行结果完全一致（确定性训练需求） |
| `asymmetric_hierarchical_concatenate_base.cc` | 非对称层次拼接（66KB）：处理 rank 数不是 2 的幂次的场景 |
| `recursive_halvingdoubling_base.cc` | 递归 Halving-Doubling：AllReduce 的经典算法，O(log N) 步完成 |
| `temp_all_gather/` | AllGather 模板集合：包含 double-ring、halving-doubling、mesh、NHR（Non-Hierarchical Ring）等多种实现 |
| `aiv_communication_base.h` | AIV 通信基类（45KB）：定义 AIV 模式下的通信原语，直接操作 AICore vector 单元 |
| `aiv_*_superkernel.h` | Superkernel：将多个通信步骤合并为单个 kernel launch，减少 launch 开销 |

---

## 4. 核心函数 / 类 / 接口

### AlgTemplateBase（`alg_template_base_pub.h`）

```cpp
class AlgTemplateBase {
public:
    // 异步执行（主接口）
    virtual HcclResult RunAsync(const std::string& tag,
        DeviceMem& inputMem, DeviceMem& outputMem,
        u64 count, HcclDataType dataType, HcclReduceOp op,
        u32 root, HcclRtStream stream) = 0;

    // 计算 scratch 内存需求
    virtual HcclResult CalcScratchMemSize(u64 count, HcclDataType dataType,
        u64& scratchSize) = 0;

    // 获取输出大小
    virtual u64 GetOutputSize(u64 inputSize) const = 0;
};
```

### AllGather 模板示例（`temp_all_gather/`）

```cpp
// Double Ring AllGather（910B 大数据量最优）
class AllGatherDoubleRing : public AlgTemplateBase {
    HcclResult RunAsync(...) override;
    // 两个 ring 并行传输，带宽利用率接近 100%
};

// Halving-Doubling AllGather（小数据量，步数少）
class AllGatherHalvingDoubling : public AlgTemplateBase {
    HcclResult RunAsync(...) override;
    // O(log N) 步完成，适合 rank 数多但数据量小的场景
};
```

### AIV 通信基类（`aiv_communication_base.h`）

```cpp
// AIV 模式：通信逻辑在 AICore vector 单元上执行
class AivCommunicationBase {
    // 发送数据到 peer（通过共享内存或 RDMA）
    __aicore__ void SendToPeer(int dstRank, LocalTensor<T>& data);

    // 从 peer 接收数据
    __aicore__ void RecvFromPeer(int srcRank, LocalTensor<T>& data);

    // 同步（等待所有 peer 完成）
    __aicore__ void Barrier();
};
```

### Superkernel（`aiv_ar_superkernel.h` 等）

```cpp
// 将多步通信合并为单个 kernel，减少 launch 开销
// 适用于小数据量、高频调用场景
extern "C" __global__ __aicore__ void AllReduceSuperKernel(
    GM_ADDR input, GM_ADDR output, GM_ADDR commArgs,
    uint32_t count, uint32_t dataType, uint32_t reduceOp) {
    AivAllReduceSuperKernel op;
    op.Init(input, output, (CommArgs*)commArgs, count, dataType, reduceOp);
    op.Process();
}
```

---

## 5. 数据流向

```
算法选择（AlgConfigurator / TopoMatcher）
  └── 确定算法类型（AlgType）
        └── 选择对应模板类（如 AllGatherDoubleRing）

模板执行
  AlgTemplateBase::RunAsync()
    ├── 计算 step 数和每步数据量
    ├── 循环执行各 step：
    │     ├── 发送本 rank 数据到 peer（通过 Transport）
    │     └── 接收 peer 数据（写入 outputMem）
    └── 同步（等待所有 step 完成）

AIV 模式（aiv_template）
  AivCommunicationBase::SendToPeer/RecvFromPeer
    └── 直接操作 AICore vector 单元（无需 CPU 介入）
          └── 通过 CommArgs.peerMems[] 访问 peer 内存
```

---

## 6. 关键业务逻辑

### 算法选择策略
- **数据量大（>= 阈值）**：Double Ring（AllGather/ReduceScatter）、Ring AllReduce
- **数据量小（< 阈值）**：Halving-Doubling、AIV Superkernel
- **非 2 的幂次 rank 数**：`asymmetric_hierarchical_concatenate_base.cc` 处理
- **确定性训练**：`alg_template_multi_deter_pipeline.cc`，保证多次运行结果一致

### AIV vs AIC 模式
- **AIC 模式**：通信由 CPU 侧 transport 驱动，AICore 只做计算
- **AIV 模式**：通信和计算都在 AICore 上执行，延迟更低，但受 AICore 内存限制

### Superkernel 优化
将多个小 kernel launch 合并为一个，减少 launch 开销（每次 launch 约 10-20μs）。适用于 AllReduce 步数多但每步数据量小的场景。

### 确定性算法
`alg_template_multi_deter_pipeline.cc` 通过固定计算顺序和使用确定性 reduce 操作（避免浮点加法的结合律问题），保证多次运行的数值结果完全一致。

---

## 7. 开发注意事项

- 新增算法模板时，必须继承 `AlgTemplateBase` 并实现所有纯虚函数，然后通过 `alg_template_register.cc` 注册。
- AIV 模板（`aiv_*.h`）是 header-only 实现，修改后需重新编译所有引用它的 `.cc` 文件。
- `aiv_communication_base.h`（45KB）是 AIV 模板的核心，修改时需全面测试所有 AIV 算子。
- 确定性算法（`deter_*`）有额外的内存和计算开销，不应在非确定性场景下使用。
- Superkernel 的 `count` 参数有上限（受 AICore L1 buffer 大小限制），超过上限需分片处理。

---

## 8. 未来可扩展点

- **自适应算法切换**：根据运行时数据量动态切换 AIC/AIV 模式，而非编译期固定。
- **新拓扑支持**：增加 Butterfly/Hypercube 等拓扑的模板实现，适应更大规模集群。
- **混合精度 Reduce**：在 `reducer.cc` 中增加 FP8/INT4 的 reduce 操作支持。
- **跨节点 AIV**：`aiv_crossnode_91093_base.h` 已有框架，可扩展到更多芯片型号的跨节点 AIV 通信。
