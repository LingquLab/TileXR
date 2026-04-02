# 模块：mc2-common（MC2 公共基础设施）

## 1. 模块作用

为所有 MC2 融合算子（`all_gather_add`、`all_gather_matmul` 等）提供共享的 tiling 头文件、公式化分块工具，以及核心 MatMul 原语 `new_mc2_mm`。是 MC2 算子开发的基础层，避免重复实现通用逻辑。

---

## 2. 目录结构

```
src/mc2/common/
├── CMakeLists.txt
├── inc/
│   ├── kernel/                    # Kernel 侧公共头文件
│   └── tiling/                    # Host 侧 tiling 公共头文件（13 个文件）
│       ├── mc2_tiling_struct.h        # 核心 tiling 结构体定义
│       ├── mc2_tiling_utils.h         # tiling 辅助函数（10KB）
│       ├── matmul_formulaic_tiling.h  # MatMul 公式化分块（10KB）
│       ├── hccl_formulaic_tiling.h    # HCCL 通信公式化分块（5KB）
│       ├── matmul_performance.h       # MatMul 性能模型
│       ├── hccl_performance.h         # HCCL 性能模型
│       ├── formulaic_tiling_datatype.h# 公式化 tiling 数据类型
│       ├── mc2_calc_num_blocks.h      # Block 数计算
│       ├── mc2_opversion_manager.h    # 算子版本管理
│       ├── mc2_tiling_common_var.h    # 公共变量定义
│       ├── one_calc_two_comm_tiling.h # 1计算2通信 tiling 模板
│       ├── moe_tiling_base.h          # MoE 场景 tiling 基类
│       └── tiling_key.h               # Tiling key 定义
├── new_mc2_mm/                    # MatMul 原语
│   ├── kernel/                    # AICore kernel 实现
│   └── tiling/                    # Host tiling 实现
└── src/                           # 公共 .cpp 实现
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `inc/tiling/mc2_tiling_struct.h` | 所有 MC2 算子的 tiling 结构体基类，host/kernel 共享布局 |
| `inc/tiling/mc2_tiling_utils.h` | tiling 工具函数：数据类型字节数、对齐、block 大小计算等 |
| `inc/tiling/matmul_formulaic_tiling.h` | MatMul 公式化分块：M/N/K 维度的 tile 计算，支持 L1/L0 缓存约束 |
| `inc/tiling/hccl_formulaic_tiling.h` | HCCL 通信 tiling：计算每步传输的数据量、时延估计 |
| `inc/tiling/matmul_performance.h` | MatMul 性能模型，用于通信/计算 overlap 的时间估算 |
| `inc/tiling/one_calc_two_comm_tiling.h` | 1个计算 + 2个通信的 pipeline 模板，MC2 core 模式 |
| `new_mc2_mm/` | `new_mc2_mm` 矩阵乘原语，MC2 kernel 内统一调用的 MatMul 实现 |

---

## 4. 核心函数 / 类 / 接口

### `mc2_tiling_struct.h` — Tiling 结构体基类

```cpp
// 所有 MC2 算子的 tiling 结构体基础字段
struct Mc2TilingBase {
    uint32_t commAlg;         // 通信算法选择
    uint32_t rankSize;
    uint32_t localRankSize;
    uint32_t mDimTotal;       // AllGather 后总 M
    uint32_t mDimLocal;       // 本 rank 的 M 分片
    uint32_t nDim;            // MatMul N 维度
    uint32_t kDim;            // MatMul K 维度
    // ... 更多字段（见实际文件）
};
```

### `mc2_tiling_utils.h` — 工具函数

```cpp
uint32_t GetDtypeBytes(ge::DataType dtype);   // 数据类型字节数
uint32_t AlignUp(uint32_t val, uint32_t align); // 向上对齐
uint32_t CalcBlockNum(uint32_t totalM, uint32_t blockM); // block 数量
```

### `matmul_formulaic_tiling.h` — MatMul 公式化分块

```cpp
// 根据 AICore 缓存层级约束计算 MatMul tile 大小
void CalcMatmulTileSize(uint32_t M, uint32_t N, uint32_t K,
    ge::DataType dtype, MatmulTileParams* params);
```

### `hccl_formulaic_tiling.h` — 通信公式化分块

```cpp
// 估算 AllGather 各步通信量，用于与 MatMul 计算对齐
void CalcHcclTilingParams(uint32_t rankSize, uint32_t dataSize,
    HcclTilingParams* params);
```

### `new_mc2_mm/` — 统一 MatMul 原语

```cpp
// AICore kernel 侧调用，封装 cube 单元 GEMM 操作
// 支持不同精度、layout 配置
void NewMc2Mm(LocalTensor<T>& a, LocalTensor<T>& b, LocalTensor<T>& c,
    const MatmulParams& params);
```

---

## 5. 数据流向

```
具体算子 tiling（如 all_gather_matmul_tiling.cpp）
  ├── 引入 mc2_tiling_utils.h → 调用工具函数
  ├── 引入 matmul_formulaic_tiling.h → 计算 MatMul tile 方案
  ├── 引入 hccl_formulaic_tiling.h → 计算通信分块方案
  ├── 引入 one_calc_two_comm_tiling.h → 建立 pipeline 参数
  └── 填充 Mc2TilingBase 派生结构体 → 传给 kernel

具体算子 kernel（如 all_gather_matmul.cpp）
  └── 调用 NewMc2Mm() ← new_mc2_mm/kernel/
```

---

## 6. 关键业务逻辑

### 1计算2通信模板（`one_calc_two_comm_tiling.h`）
MC2 算子的核心 pipeline 模式：每个计算 step 对应两个通信 step（发送 + 接收），三者流水交叠。该头文件提供通用的步数计算和 step 到数据偏移的映射。

### 性能模型驱动分块
`matmul_performance.h` 和 `hccl_performance.h` 提供硬件性能参数（计算峰值、带宽），tiling 利用这些参数做 overlap 最优化，选择让通信恰好被计算遮盖的分块方案。

### 算子版本管理（`mc2_opversion_manager.h`）
不同 CANN 版本或芯片代际支持不同的 tiling key，版本管理器提供查询接口，避免 tiling 函数内部硬编码版本判断。

---

## 7. 开发注意事项

- `mc2_tiling_struct.h` 中的结构体同时被 host tiling 侧和 AICore kernel 侧使用，修改字段时两侧必须同步重编译。
- `new_mc2_mm` 是 cube 单元操作的封装，修改时需确认不同精度（FP16/BF16/INT8）下的 layout 要求。
- 新增算子时，应直接引用 `common/inc/tiling/` 中的头文件，不要复制代码。
- `mc2_opversion_manager.h` 中的版本列表需在升级 CANN 时同步维护。

---

## 8. 未来可扩展点

- **MoE 支持**：`moe_tiling_base.h` 已预留 MoE 场景的 tiling 基类，可在此基础上实现 `MoeAllGatherMatmul` 等算子。
- **更多原语**：`new_mc2_mm/` 中可增加 `new_mc2_reduce_scatter` 等通信原语，供未来算子复用。
- **自动调优**：当前性能模型是静态参数，可接入 AutoTiling 框架做运行时参数搜索。
