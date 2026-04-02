# 模块：op-simulator（算子模拟器）

## 1. 模块作用

在无物理 Ascend 硬件的环境下对 AICore kernel 进行功能验证和性能模拟。提供标准测试模板，可快速接入新算子进行冒烟测试，无需完整的 CANN 设备环境。

---

## 2. 目录结构

```
op-simulator/
├── README.md
├── CMakeLists.txt
├── compile_and_run.sh         # 一键编译并运行
├── run_test_ca.sh             # 运行 ca（compute accuracy）测试
├── scripts/
│   └── run.sh                 # 底层运行脚本
├── src/
│   ├── CMakeLists.txt
│   └── base_test.cpp          # 基础 DMA 测试 kernel
└── test_template.cpp          # 新算子测试模板
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `src/base_test.cpp` | 最基础的 kernel：GM→UB DMA 搬运 + UB→GM 回写，验证 DMA 通路 |
| `test_template.cpp` | 新算子测试的标准模板，提供 init/run/verify 三段式框架 |
| `compile_and_run.sh` | 调用 CMake 编译后直接运行，适合快速迭代 |
| `run_test_ca.sh` | 运行精度对比测试，对比 kernel 输出与参考值 |

---

## 4. 核心函数 / 类 / 接口

### `base_test.cpp` — 基础 DMA kernel

```cpp
// DemoTest：验证 GM ↔ UB DMA 通路
extern "C" __global__ __aicore__ void DemoTest(GM_ADDR src, GM_ADDR dst) {
    __ubuf__ uint8_t local[BUFFER_SIZE];
    copy_gm_to_ubuf_align_b8(local, (GM_ADDR)src, ...);
    set_flag(PIPE_MTE2, PIPE_S, 0);
    wait_flag(PIPE_MTE2, PIPE_S, 0);
    copy_ubuf_to_gm_align_b8((GM_ADDR)dst, local, ...);
    set_flag(PIPE_S, PIPE_MTE3, 0);
    wait_flag(PIPE_S, PIPE_MTE3, 0);
}
```

### `test_template.cpp` — 测试模板结构

```
1. 定义输入/输出 tensor（shape、dtype）
2. 分配 host 内存 + 设备内存
3. 初始化输入数据
4. 调用 aclrtMemcpy H2D
5. Launch kernel
6. aclrtMemcpy D2H
7. 与参考结果对比（误差阈值）
```

---

## 5. 数据流向

```
compile_and_run.sh
  └── CMake 编译（链接模拟器运行时）
        └── 生成测试可执行文件
              └── run_test_ca.sh 执行
                    ├── 初始化模拟器设备上下文
                    ├── 分配模拟内存
                    ├── Launch kernel（模拟执行）
                    └── 验证输出正确性
```

---

## 6. 关键业务逻辑

### DMA + flag 同步模式
`base_test.cpp` 展示了 AICore 中标准的 DMA 同步写法：
- `set_flag(PIPE_MTE2, PIPE_S, 0)` + `wait_flag` 确保 GM→UB 搬运完成后再处理
- `set_flag(PIPE_S, PIPE_MTE3, 0)` 确保处理完成后再回写

这是所有 AICore kernel 都遵循的 pipeline 同步模式。

### 精度对比（run_test_ca.sh）
测试脚本将 kernel 输出与 Python/numpy 生成的参考值做逐元素对比，支持设置相对误差阈值（`rtol`/`atol`）。

---

## 7. 开发注意事项

- 模拟器不支持所有硬件指令，特别是 cube 单元（矩阵乘）的精确模拟有限制，主要用于 vector/DMA 路径验证。
- `test_template.cpp` 中的 tensor shape 和 dtype 需与对应 tiling 逻辑一致，否则内存越界。
- 新算子接入时，以 `test_template.cpp` 为基础，不要从零编写，以保持测试结构一致性。

---

## 8. 未来可扩展点

- **自动化测试集成**：将 `run_test_ca.sh` 接入 CI 流水线，每次提交自动运行模拟器测试。
- **性能 profiling**：在模拟器中增加 cycle 计数，评估 kernel 的理论性能上限。
- **多 kernel 测试**：扩展模板支持同时测试多个 kernel 的协作（如通信 + 计算）。
