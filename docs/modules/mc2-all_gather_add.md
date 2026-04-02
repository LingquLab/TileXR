# 模块：mc2-all_gather_add（融合 AllGather + Add 算子）

## 1. 模块作用

实现 AllGather 通信与 element-wise Add 计算的融合算子，在单机双卡场景下将通信与计算 pipeline 化，减少端到端延迟。固定支持 `a(240,256)` 输入，输出 `b(480,256)`，`rank_size=2`，数据类型 FLOAT16。

---

## 2. 目录结构

```
src/mc2/all_gather_add/
├── README.md                  # 产品文档（中文）
├── CMakeLists.txt
├── op_host/                   # Host 侧实现
│   ├── all_gather_add_def.cpp         # 算子定义
│   ├── CMakeLists.txt
│   ├── config/                        # 芯片配置
│   │   ├── ascend910_93/
│   │   └── ascend910b/
│   ├── op_api/                        # aclnn API 层
│   │   └── aclnn_all_gather_add.cpp
│   └── op_tiling/                     # Tiling 计算
│       └── all_gather_add_tiling.cpp
├── op_kernel/                 # Kernel 侧实现（AICore）
│   ├── all_gather_add.cpp             # 主 kernel 入口
│   ├── all_gather_add.h               # Kernel 类声明
│   └── all_gather_add_tiling.h        # Tiling 结构体
└── examples/                  # 示例代码
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `op_host/all_gather_add_def.cpp` | 算子注册（`REGISTER_OP`），定义输入输出 tensor 签名 |
| `op_host/op_tiling/all_gather_add_tiling.cpp` | Tiling 计算逻辑，决定 block 数、每 block 处理的数据量 |
| `op_host/op_api/aclnn_all_gather_add.cpp` | `aclnnAllGatherAdd` C API 实现，调用 tiling + launch kernel |
| `op_kernel/all_gather_add.cpp` | AICore kernel 入口，实例化 `AllGatherAdd` 类并调用 `Process()` |
| `op_kernel/all_gather_add.h` | Kernel 类实现：DMA 搬运、通信同步、Add 计算 |

---

## 4. 核心函数 / 类 / 接口

### Host 侧 API

```cpp
// aclnn_all_gather_add.cpp
aclnnStatus aclnnAllGatherAdd(
    const aclTensor* a,        // 输入 tensor (240, 256)
    aclTensor* b,              // 输出 tensor (480, 256)
    aclrtStream stream,
    void* workspace,
    uint64_t workspaceSize
);
```

### Kernel 侧类

```cpp
// all_gather_add.h
class AllGatherAdd {
    __aicore__ void Init(GM_ADDR a, GM_ADDR b, CommArgs* commArgs);
    __aicore__ void Process();
private:
    // DMA 搬运、通信、计算流水线
};

// all_gather_add.cpp（入口）
extern "C" __global__ __aicore__ void all_gather_add(GM_ADDR a, GM_ADDR b, GM_ADDR commArgs) {
    AllGatherAdd op;
    op.Init(a, b, (CommArgs*)commArgs);
    op.Process();
}
```

---

## 5. 数据流向

```
用户调用 aclnnAllGatherAdd
  │
  ▼
Tiling 计算（all_gather_add_tiling.cpp）
  ├── 确定 block 数量
  ├── 计算 workspace 大小
  └── 生成 tiling 参数
  │
  ▼
Launch kernel（all_gather_add）
  │
  ▼
AICore kernel 执行
  ├── Rank 0: 读取本地 a[0:240, :] → 写入 peerMems[1][0:240, :]
  ├── Rank 1: 读取本地 a[0:240, :] → 写入 peerMems[0][240:480, :]
  ├── SyncCollectives 同步（等待对方写完）
  ├── 读取 peerMems[peer][...] + 本地 a → Add 计算
  └── 写入 b[0:480, :]
```

---

## 6. 关键业务逻辑

### 固定形状约束
- 输入 `a`: `(240, 256)`, FLOAT16
- 输出 `b`: `(480, 256)`, FLOAT16
- `rank_size = 2`（单机双卡）

这些约束硬编码在 tiling 和 kernel 中，不支持动态形状。

### 通信 + 计算融合
Kernel 内部将 AllGather 通信（写 peer 内存）与 Add 计算（读 peer 内存 + 本地数据）pipeline 化，通过 `SyncCollectives` 的 flag 机制保证数据依赖正确性。

---

## 7. 开发注意事项

- 修改形状支持需同时改 tiling、kernel、README 文档。
- 当前仅支持 FLOAT16，扩展到 FLOAT32/INT8 需修改 DMA 和计算指令。
- `rank_size > 2` 的场景需重新设计通信拓扑（当前是对等双向）。
- 编译时需指定 `--soc=ascend910b`，不同芯片的 config 目录下有不同的参数。

---

## 8. 未来可扩展点

- **动态形状**：将 `(240, 256)` 改为 tiling 参数，支持任意 M/N。
- **多 rank**：扩展到 `rank_size > 2`，需实现 ring/tree 拓扑。
- **多数据类型**：增加 FLOAT32/INT8/BF16 支持。
- **性能优化**：当前是简单的 block 并行，可引入 double buffer 提升带宽利用率。
