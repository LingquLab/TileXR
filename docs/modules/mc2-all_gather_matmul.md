# 模块：mc2-all_gather_matmul（融合 AllGather + MatMul 算子）

## 1. 模块作用

实现 AllGather 通信与矩阵乘法（MatMul）的融合算子，将通信与计算 overlap，核心场景是大模型分布式训练中的 Tensor Parallel 前向计算。提供完整的 aclnn API、图融合 pass、以及单元 / 系统测试套件。

---

## 2. 目录结构

```
src/mc2/all_gather_matmul/
├── README.md
├── CMakeLists.txt
├── docs/                          # 补充文档
├── op_host/                       # Host 侧
│   ├── config/
│   │   ├── ascend910_93/
│   │   └── ascend910b/
│   └── op_tiling/
│       ├── all_gather_matmul_tiling.cpp    # 核心 tiling（613 行）
│       ├── all_gather_formulaic_tiling.cpp # 公式化 tiling 辅助
│       └── all_gather_formulaic_tiling.h
├── op_api/                        # aclnn C API
│   └── aclnn_all_gather_matmul.cpp
├── op_graph/                      # 图融合 pass
│   └── fusion_pass/
│       └── all_gather_matmul_fusion_pass.cpp
├── op_kernel/                     # AICore kernel
│   └── all_gather_matmul.cpp
└── tests/
    ├── ut/                        # 单元测试
    └── st/                        # 系统测试
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `op_tiling/all_gather_matmul_tiling.cpp` | 主 tiling 逻辑：参数校验、M 分片、comm 算法选择、workspace 计算 |
| `op_tiling/all_gather_formulaic_tiling.cpp` | 公式化 tiling 辅助函数，计算 block/tile 分配方案 |
| `op_api/aclnn_all_gather_matmul.cpp` | `aclnnAllGatherMatmul` API，串联 tiling + kernel launch |
| `op_graph/fusion_pass/...` | 图编译阶段 fusion pass：识别 AllGather + MatMul 模式，合并为单算子 |
| `op_kernel/all_gather_matmul.cpp` | AICore kernel 入口，调用 `mc2/common` 中的 `new_mc2_mm` 原语 |
| `tests/ut/` | 单元测试：tiling 参数验证、边界条件 |
| `tests/st/` | 系统测试：端到端正确性 + 性能 |

---

## 4. 核心函数 / 类 / 接口

### Tiling（`all_gather_matmul_tiling.cpp`）

```cpp
// 主 tiling 函数（注册到算子框架）
ge::graphStatus AllGatherMatmulTilingFunc(gert::TilingContext* context);

// 参数校验
static ge::graphStatus AllGatherParamsCheck(const gert::TilingContext* context);

// M 维度分片：将 AllGather 后的 M 分配给各 block 处理
static void MCSpliteM(TilingContext* ctx, AllGatherMatmulTilingData* tilingData);

// Comm 算法选择（ring / double-ring / mesh 等）
static void SetCommAlg(AllGatherMatmulTilingData* tilingData, uint32_t rankSize);

// MatMul tiling 计算
static ge::graphStatus CalcMatmulTiling(gert::TilingContext* context,
    AllGatherMatmulTilingData* tilingData);

// Workspace 计算
static ge::graphStatus MC2SetWorkspace(gert::TilingContext* context,
    AllGatherMatmulTilingData* tilingData);
```

### Host API

```cpp
// op_api/aclnn_all_gather_matmul.cpp
aclnnStatus aclnnAllGatherMatmul(
    const aclTensor* a,        // 本 rank 的输入矩阵分片
    const aclTensor* weight,   // MatMul 权重
    aclTensor* output,         // 输出
    int64_t rankSize,
    aclrtStream stream,
    void* workspace,
    uint64_t workspaceSize
);
```

---

## 5. 数据流向

```
图编译阶段（可选）
  AllGather 算子 + MatMul 算子
       └── fusion_pass 识别 → 替换为 AllGatherMatmul 单算子

运行时
  aclnnAllGatherMatmul()
    │
    ▼
  AllGatherMatmulTilingFunc()
    ├── AllGatherParamsCheck()     参数合法性
    ├── MCSpliteM()                M 维度分片方案
    ├── SetCommAlg()               确定通信算法
    ├── CalcMatmulTiling()         MatMul cube 分块
    └── MC2SetWorkspace()          workspace 大小
    │
    ▼
  Launch AICore kernel
    ├── 各 block 并行：通信拉取 peer 的 a 分片
    ├── 通信与 MatMul 计算 overlap（double buffer）
    └── 结果写入 output
```

---

## 6. 关键业务逻辑

### M 维度分片（MCSpliteM）
AllGather 后的总 M = `localM × rankSize`，tiling 将其按 block 数分片，每个 AICore block 处理一个 M 分片的 MatMul，实现计算并行化。

### 通信与计算 Overlap
Kernel 内部使用 double buffer 技术：在 block i 执行 MatMul 的同时，提前发起 block i+1 数据的通信，隐藏通信延迟。

### Comm 算法选择（SetCommAlg）
根据 `rankSize` 和拓扑类型（HCCS/PCIe）选择不同通信算法（ring allgather / 2D mesh 等），写入 `tilingData.commAlg` 供 kernel 读取。

### 支持范围
- 数据类型：FLOAT16（主要）
- rank_size：最大 32
- 需 workspace 内存（动态计算大小）

---

## 7. 开发注意事项

- `AllGatherMatmulTilingFunc` 通过 `REGISTER_TILING_FUNC` 宏注册到框架，函数签名不可改变。
- Tiling 结构体 `AllGatherMatmulTilingData` 在 tiling 侧和 kernel 侧共享（`mc2/common/inc/tiling/mc2_tiling_struct.h`），字段布局需二进制对齐。
- `tests/ut` 中的 tiling 单测会直接调用 tiling 函数，新增 tiling 参数时需同步更新测试用例。
- Fusion pass 只在图编译模式下生效，单算子模式不经过 pass。

---

## 8. 未来可扩展点

- **BF16/INT8 支持**：修改 `AllGatherParamsCheck` 和 MatMul tiling 的数据类型分支。
- **ReduceScatter + MatMul**：对称地实现 `MatMulReduceScatter` 算子（`TileXRType` 已预留枚举值）。
- **动态 rankSize**：当前 rankSize 在 tiling 时确定，可扩展到运行时动态场景。
- **更多 comm 算法**：`SetCommAlg` 中增加 halving-doubling 等算法，适应不同网络拓扑。
