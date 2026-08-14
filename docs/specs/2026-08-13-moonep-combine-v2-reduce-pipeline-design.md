# MoonEP Combine V2 Reduce 多缓冲流水设计

状态：Implemented，待构建与硬件验证

日期：2026-08-14

## 1. 目标与范围

优化 MoonEP Combine V2 最终阶段 `ReduceHidden()`。发送、CQ、Self-copy 和
inbound 完成确认结束后，清空旧流水并执行 `TPipe::Reset()`，将当前 AIV 的全部
216 KiB UB 重新分配给 Reduce。

首版采用以下方案：

- 最多 8 个 BF16 MTE2 输入 slot；
- 1 个全流水共享的 FP32 Cast 工作区；
- 1 个当前输出行独占的 FP32 accumulator；
- 2 个独立的 BF16 MTE3 输出 slot；
- TopK 按原 route 顺序执行 FP32 串行 Add；
- 单个 BF16 hidden 行最大 14 KiB，即 7168 个元素；
- `H > 7168` 时沿 hidden 维拆成多行，复用同一组 UB；
- 输入 slot 数根据 H 和可用 UB 计算，上限为 8。

这里的“8 流水”仅表示最多 8 个 BF16 route 行处于 MTE2 已提交、正在搬入或等待
Vector 消费的状态。它不是 8 套完整计算缓冲，也不是 8 路 Vector 并行。

本设计不改变 Host API、Kernel ABI、GM 数据布局、workspace 公式以及
`tools/moonep/test_npu_e2e.py` 的调用契约。

## 2. 背景与参考实现

已测 Ascend950PR、8P、BF16、`BS=8192`、`topK=16`、`H=3584`：

| 指标 | 耗时 |
| --- | ---: |
| Combine V2 Kernel | 约 7038 us |
| `ReduceHidden()` | 约 4735 us |
| inbound wait | 约 63 us |

Reduce 约占 Kernel 时间的 67.3%。当前代码按最多 4096 hidden 元素切 tile，输入
queue 深度为 1，每批搬入 4 个 route 后逐 route 执行 Cast 和 Add，输出 queue
深度也为 1。`H=7168` 会被拆成 `[4096, 3072]` 两个 tile。

参考实现：

```text
D:/3_codex/ops-transformer-v9.1.0-.r401/mc2/
moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_a5_mte.h
```

可借鉴的部分是：

- Reset 后根据剩余 UB 计算 `bufferNum`，最大限制为 8；
- 使用多 slot `TQue<VECIN>` 预取完整 BF16 token 行；
- 每次消费一个输入行，Cast 为 FP32 后累加到常驻 FP32 sum buffer；
- TopK 完成后一次 Cast 并由 MTE3 写回；
- 用 queue 或窄粒度硬事件约束 MTE2、Vector、MTE3 的所有权。

参考代码中的 `Sum` 用于 mask 统计，不用于 TopK hidden 聚合。TopK 聚合实际采用
串行 `Cast -> Muls -> Add`。MoonEP 没有该路径的 expert scale 乘法，因此忽略
反量化和 `Muls`，只参考搬运、Cast、Add 和输出流水。

## 3. 数据分行

BF16 单元素为 2 Byte，单行上限为：

```text
maxRowBytes    = 14 * 1024 = 14336 B
maxRowElements = 14336 / 2 = 7168
```

对任意合法 H：

```text
rowCapacityElements = min(H, 7168)
rowCount             = ceil_div(H, 7168)
rowOffset(i)         = i * 7168
rowElements(i)       = min(7168, H - rowOffset(i))
```

示例：

| H | hidden 分行 |
| ---: | --- |
| 3584 | `[3584]` |
| 4096 | `[4096]` |
| 7168 | `[7168]` |
| 7169 | `[7168, 1]` |
| 8192 | `[7168, 1024]` |
| 14336 | `[7168, 7168]` |

“分行”只描述 Reduce 的 UB 分片。GM 输入仍为
`[token * topK + route, H]`，输出仍为 `[token, H]`。

每个 `(token, hiddenRow)` 独立完成全部 TopK 累加并写回。Accumulator 不跨 token
或 hidden 行共享。整次 Reduce 只按最大行容量分配一次 UB；尾行复用已有 slot，
不再次 Reset。

## 4. UB 布局与流水数计算

### 4.1 对齐量

设本次 Reduce 的最大逻辑行长度为 `rowCapacityElements`：

```text
inputSlotBytes = AlignUp(rowCapacityElements * sizeof(bfloat16_t), 32)
floatRowBytes  = AlignUp(rowCapacityElements * sizeof(float), 32)
outputSlotBytes = inputSlotBytes
```

固定 UB 空间为：

```text
fixedBytes = floatRowBytes                // FP32 Cast workspace
           + floatRowBytes                // FP32 accumulator
           + 2 * outputSlotBytes          // BF16 MTE3 ping/pong
```

可用于输入流水的空间：

```text
availableInputBytes = fullUbBytes - fixedBytes
maxInputSlotsByUb   = floor(availableInputBytes / inputSlotBytes)
```

单核 Reduce 工作项数量：

```text
tokenBegin = floor(BS * core / activeCoreCount)
tokenEnd   = floor(BS * (core + 1) / activeCoreCount)
tokenCount = tokenEnd - tokenBegin

totalRouteWorkItems = tokenCount * rowCount * topK
```

最终输入 slot 数：

```text
inputBufferNum = min(8, maxInputSlotsByUb, totalRouteWorkItems)
```

当 `totalRouteWorkItems > 0` 时必须满足 `inputBufferNum >= 1`。若固定空间或一个输入
slot 已超过 UB，配置非法，不能像参考实现一样强制将 0 改成 1 后继续运行。

### 4.2 典型 UB 预算

`fullUbBytes = 216 KiB`：

| H/最大行 | 单输入 slot | 两块 FP32 | 两个输出 slot | 最大输入 slot | 实际采用 | 总占用 | 余量 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3584 | 7 KiB | 28 KiB | 14 KiB | 24 | 8 | 98 KiB | 118 KiB |
| 4096 | 8 KiB | 32 KiB | 16 KiB | 21 | 8 | 112 KiB | 104 KiB |
| 7168 | 14 KiB | 56 KiB | 28 KiB | 9 | 8 | 196 KiB | 20 KiB |

在 BF16 单行不超过 14 KiB 的当前契约下，只要工作项数量不少于 8，所有 H 都可
配置 8 个输入 slot。动态公式仍必须保留，以处理小工作量、dtype/UB 约束变化以及
未来平台差异。

### 4.3 缓冲职责

| 缓冲 | 数量 | 生产者 | 消费者 | 生命周期 |
| --- | ---: | --- | --- | --- |
| `inputQueue` slot | `1..8` | MTE2 | Vector | 单个 route 行 |
| `castWorkspace` | 1 | Vector Cast | Vector Add | 当前 route |
| `accumulator` | 1 | Vector | Vector/output Cast | 当前 token/hiddenRow |
| `outputQueue` slot | 2 | Vector Cast | MTE3 | 一个完成的 token/hiddenRow |

FP32 工作区不按输入 slot 数复制。Vector 每次只消费一个 route，前后 route 通过
Vector 指令顺序和必要的 `PIPE_V` 依赖共享该工作区。

## 5. Reduce 阶段边界

`WaitInboundDone()` 证明远端和本地 scratch 数据已经到齐，但单独这一条件不足以
直接 Reset。阶段切换必须为：

```text
完成全部 send step
-> 完成对应 CQ/grant
-> 完成 Self-copy
-> WaitInboundDone
-> 完成失败状态发布与跨核收敛
-> drain 旧 MTE/Vector 事件
-> 确认旧 TQue tensor 全部释放
-> TPipe::Reset()
-> 恢复 Reduce 依赖的控制寄存器状态
-> 初始化 Reduce 专用 UB
```

Phase 边界允许一次 `PIPE_ALL` drain。Reduce 内层循环禁止用 `PIPE_ALL` 作为常规
同步，否则会破坏 MTE2/Vector/MTE3 重叠。

`TPipe::Reset()` 后，发送阶段的所有 LocalTensor、queue slot 和 UB 地址均失效，
不得保存并继续引用。参考实现说明 Reset 可能复位控制寄存器；实现时需核对 CANN
9.1/A5 上 Reduce 使用的舍入、饱和等状态并显式恢复需要的非默认配置。

## 6. 工作项线性化

输入 queue 必须保持 issue 和 consume 的 FIFO 顺序。使用单调序号即可推导 GM
地址，不需要为每个 slot 再分配 UB descriptor。

```text
groupOrdinal = workOrdinal / topK
route        = workOrdinal % topK
tokenLocal   = groupOrdinal / rowCount
hiddenRow    = groupOrdinal % rowCount
token        = tokenBegin + tokenLocal
hiddenOffset = hiddenRow * 7168
rowElements  = min(7168, H - hiddenOffset)
```

因此工作顺序固定为：

```text
token 0, row 0, route 0..topK-1
token 0, row 1, route 0..topK-1
...
token 1, row 0, route 0..topK-1
...
```

MTE2 可以跨 hidden 行和 token 提前预取，但 Vector 必须按该顺序消费。这样只有一块
accumulator，并保持旧实现 route 0 到 route `topK-1` 的 FP32 累加顺序。

## 7. 多缓冲流水

### 7.1 Prologue

```text
prefetchCount = min(inputBufferNum, totalRouteWorkItems)

for issueOrdinal in [0, prefetchCount):
    Alloc input slot
    根据 issueOrdinal 计算 GM route 行和逻辑 rowElements
    MTE2 DataCopyPad GM -> input slot
    EnQue input slot
```

所有输入 slot 按最大 `inputSlotBytes` 分配，但 MTE2 只读取当前工作项的逻辑
`rowElements`。非 32 Byte 尾行使用目标 CANN 9.1 重载规定的 padding 方式。

### 7.2 Steady state

对 `consumeOrdinal = 0..totalRouteWorkItems-1`：

1. `DeQue` 当前输入 slot，建立 MTE2 -> Vector 依赖；
2. 根据 ordinal 得到当前 token、hiddenRow、route 和逻辑长度；
3. route 0 开始前将 accumulator 的逻辑区域清零；
4. Cast 当前 BF16 route 到共享 FP32 workspace；
5. 执行 `accumulator += castWorkspace`；
6. `FreeTensor` 当前输入 slot；
7. 若仍有未提交工作项，立即用释放的 slot 提交下一条 MTE2；
8. 若当前 route 为 `topK-1`，将 accumulator Cast 到 BF16 output slot 并提交 MTE3。

稳态目标：

```text
MTE2   : 搬入 route i+1 ... i+7
Vector : Cast + Add route i
MTE3   : 写回上一个已完成 token/hiddenRow
```

`inputBufferNum=8` 只是增加 MTE2 ahead distance。MTE2 引擎仍串行执行自己的 copy，
Vector 引擎也仍串行执行 Cast/Add。

### 7.3 Output 与 Epilogue

完成一个 `(token, hiddenRow)` 后：

```text
Alloc output slot
Cast accumulator FP32 -> BF16
EnQue/DeQue 建立 Vector -> MTE3 依赖
DataCopyPad output slot -> GM[token, hiddenOffset]
Free output slot
```

两个输出 slot 独立于输入 slot和 FP32 工作区，使上一输出的 MTE3 可以和下一组的
MTE2、Duplicate、Cast、Add 重叠。输出只写逻辑 `rowElements`。

最后一个输入消费完成后，等待所有 output slot 的 MTE3 写回完成，再退出 Reduce。
只在该最终 drain 使用全流水同步。

### 7.4 依赖和所有权

| 边界 | 机制 |
| --- | --- |
| MTE2 写 input -> Vector 读 input | input queue `EnQue/DeQue` 或等价 `MTE2_V` |
| Vector 读完 input -> MTE2 复用 slot | `FreeTensor/AllocTensor` 队列所有权 |
| Vector 写 output -> MTE3 读 output | output queue `EnQue/DeQue` 或等价 `V_MTE3` |
| MTE3 读完 output -> Vector 复用 slot | output queue slot 所有权或 `MTE3_V` |
| Cast workspace -> Add | 目标 API 要求的 `PIPE_V` 依赖 |
| Add accumulator -> 下一 Add/最终 Cast | Vector 指令顺序及必要的 `PIPE_V` |

首版使用 `TQue` 管理 MTE 与 Vector 的 slot 生命周期，避免手写 8 组 event ID。
CANN 9.1 的 `TQue` 实现中，模板参数 `depth` 限制同时入队但尚未出队的 tensor 数，
而 `InitBuffer` 的 `num` 控制物理 UB slot 数。输入 prologue 会先连续入队最多 8 个
tensor，因此输入队列必须声明为 `TQue<VECIN, 8>`，再以运行时
`inputBufferNum`（1 到 8）初始化物理 slot。不能使用 `TQue<VECIN, 1>` 连续入队
8 次；第二次 `EnQue` 已超过 queue depth。Ascend 950PR 的 CANN 9.1 头文件同时将
同类 event ID 上限定义为 8，与该输入深度匹配。

输出队列每次 `EnQue` 后立即 `DeQue` 并提交 MTE3，因此编译期 depth 保持 1，物理
slot 数为 2。`FreeTensor` 记录 `MTE3_V` 完成事件；后续 `AllocTensor` 轮转回该 slot
时等待事件，从而避免 MTE3 尚未读完就覆盖输出 UB。

## 8. 累加和数值语义

首版保持旧顺序：

```text
Duplicate(accumulator, 0)
for route = 0..topK-1:
    Cast(route, castWorkspace)
    Add(accumulator, accumulator, castWorkspace)
```

它不使用高阶 RA ReduceSum，不改变 FP32 加法结合顺序。RA ReduceSum 可能使用树形
相加、需要额外 FP32 source/partial 空间，并会降低可用于 MTE2 的流水深度，因此只
保留为后续独立对照方案。

“route 0 直接 Cast 到 accumulator”可以少一次 Duplicate 和一次 Add，但可能改变
有符号零、NaN 等位级行为，不进入首版。若后续启用，需独立进行 exact BF16 和特殊
值验证。

`topK=0` 若 Host 契约非法，则继续在 Host 拒绝；若契约允许，则 Kernel 必须显式
输出零且不能进入空工作项流水。非固定 TopK 通过 ordinal 公式自然支持，不依赖 4
或 8 的倍数。

## 9. H=7168 示例

单核处理 4 个输出 token，每个 token 有 16 个 route：

```text
totalRouteWorkItems = 4 * 1 * 16 = 64
inputBufferNum      = min(8, 9, 64) = 8
```

UB：

```text
8 * 14 KiB BF16 input = 112 KiB
1 * 28 KiB FP32 cast  =  28 KiB
1 * 28 KiB FP32 sum   =  28 KiB
2 * 14 KiB BF16 output=  28 KiB
total                 = 196 KiB
remain                =  20 KiB
```

与当前 `[4096, 3072]` 两 tile 实现比较：

| 操作 | 当前实现 | 新整行流水 |
| --- | ---: | ---: |
| MTE2 DataCopy | 128 | 64 |
| BF16 -> FP32 Cast | 128 | 64 |
| FP32 Add | 128 | 64 |
| accumulator Duplicate | 8 | 4 |
| FP32 -> BF16 Cast | 8 | 4 |
| MTE3 DataCopy | 8 | 4 |

数据量不变：

```text
MTE2 = 4 * 16 * 14 KiB = 896 KiB
MTE3 = 4 * 14 KiB      =  56 KiB
```

新方案每个 route 使用独立 input slot，queue 的 Alloc/EnQue/DeQue/Free 次数会高于
当前“一 slot 装 4 routes”的组织。预期收益来自整行处理减少 Vector/DataCopy 指令
调用，以及深度 8 的 MTE2 ahead window；不能仅凭调用数承诺最终加速。

## 10. 性能预期与风险

理论稳态由串行：

```text
Told ~= Tmte2 + Tvector + Tmte3
```

趋向：

```text
Tnew ~= max(Tmte2, Tvector, Tmte3) + prologue/epilogue
```

但实际收益受以下因素限制：

- MTE2 总字节量和 15/16 次有效 FP32 加法量没有降低；
- MTE2 单引擎内部仍串行，8 slot 不等于 8 倍带宽；
- queue 管理开销可能抵消部分整行收益；
- 深度超过覆盖 MTE2 latency 所需值后，4/6/8 slot 的差异可能很小；
- 14 KiB 整行 Vector 指令效率需要在 Ascend950 实测；
- `H>7168` 的行边界会增加 accumulator 初始化和输出次数；
- 不正确的 Reset drain、尾行 padding 或 output slot 复用会导致竞态或越界。

因此 8 是容量上限和首选测试配置，不预设它必然比 4 或 6 最快。

## 11. 验证计划

### 11.1 静态和构建

- 保持 C++14、CANN 9.1.0 和现有 Kernel ABI；
- 对 `fixedBytes + inputBufferNum * inputSlotBytes <= fullUbBytes` 做宽整数检查；
- 验证 H、BS、TopK 乘法无溢出；
- 验证动态 `InitBuffer` 数量和目标 TQue 重载；
- 检查 Reset 前所有旧 tensor 已释放、事件已 drain；
- 检查尾行 GM/UB 范围和两方向 DataCopyPad 参数单位；
- `DataCopyExtParams.blockLen` 为 `uint32_t`，字节数包含 `sizeof` 时需显式窄化，
  否则 CANN 9.1 AICore 编译器会拒绝列表初始化；
- 检查所有退出路径均 drain 已提交的 MTE3。

### 11.2 正确性矩阵

| 维度 | 用例 |
| --- | --- |
| H 小值和对齐 | `1, 15, 16, 17, 3584` |
| 旧 tile 边界 | `4095, 4096, 4097` |
| 单行上限 | `7167, 7168, 7169` |
| 多行 | `8192, 14336, 14337` |
| TopK | Host 合法的 `1, 2, 3, 4, 5, 15, 16, 17` |
| 工作量 | 少于 8、等于 8、多于 8 个 route work item |
| core 分工 | 小 BS、非均匀合法 BS、`BS=8192` |

重点检查 `H=7169`：第一行写 7168 个元素，第二行只写 1 个元素，不覆盖下一 token。
结果与当前实现和 CPU reference 比较；现有契约要求 exact BF16 时继续按 exact 验收。

### 11.3 性能矩阵

在 Ascend950PR、8P、BF16、`topK=16` 上至少比较：

```text
H = 3584, 7168, 8192
BS = 128, 8192
inputBufferNum cap = 1, 2, 4, 6, 8
output slot = 1, 2
```

记录 Kernel 总耗时、`ReduceHidden()`、inbound wait、MTE2/Vector/MTE3 active 和
stall、最慢 rank，以及各流水深度的收益。默认启用条件为正确性不退化、无死锁和
越界，且主场景 Reduce 与 Kernel 总耗时均有稳定下降。

## 12. 非目标和后续候选

首版不包含：

- RA ReduceSum 或树形 TopK 归约；
- 8 套 FP32 workspace/accumulator；
- 多 token Vector 并行；
- route 0 直接初始化 accumulator；
- 反量化、expert scale 或 RMSNorm；
- GM 布局、Host API 或测试契约修改；
- 代码实现和 NPU 测试。

后续只有在首版数据证明瓶颈仍位于 Vector Add 时，才评估 RA ReduceSum、批量 route
Cast、首 route 直接初始化或不同 hidden 分片策略。
