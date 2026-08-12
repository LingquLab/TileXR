# TileXR MoonEP 接入 MindSpeed 调试经验

本文记录 2026-08-10 至 2026-08-12 在单机 8 卡、4K/8P/EP8 模型中调试
TileXR MoonEP 接入 MindSpeed 的问题、证据链和可复用方法。后续调试应把本文作为
排查入口，但不能把历史根因直接套到新故障上；必须先用当前源码、二进制和运行日志
重新确认第一个失败边界。

## 最终结论

这次适配不是一个单点故障，而是模型规模和完整前反向流程依次暴露了五类契约问题：

1. 通信 runtime 和 RA 的所有权不唯一。
2. MindSpeed 依赖的 Buffer、Tensor view 和 zero-copy 契约超出了 MoonEP 公共 API。
3. 逻辑 worker 数、物理 QP 数和 AIV block 数被错误地绑定到同一固定上限。
4. CQE、SQ tail、ring index 和 cycle bit 的语义没有明确区分。
5. 同一个 plan 跨 Planner、Dispatch、Prefetch、Combine、反向 Dispatch 和 ReduceGrad
   复用，但 status 的输入、输出和清理责任没有形成完整状态机。

最后阻塞 grouped-URMA Dispatch + Combine V2 完整模型的直接原因，是前向 Combine
成功后在复用 plan 中留下 `status=3000`。反向 URMA Dispatch 要求输入状态为 `0`，
并通过 `CAS(0, error)` 发布首错误。旧状态同时导致正常发送路径被跳过，并阻止真实
错误码写入，最终表现为反向 Dispatch 等待 flag 超时。

Python 侧的 `plan.status.zero_()` 不是可靠修复：NPU task queue 和 Kernel launch 可能
位于不同 stream，无法保证 reset 先于 consumer Kernel。最终方案是通过
`TILEXR_MOONEP_FLAG_RESET_STATUS`，由 Host 在 Dispatch 的同一 stream 上排队
`aclrtMemsetAsync`，然后紧邻 Kernel launch。

## 已确认问题

| 现象 | 根因 | 修复原则 | 发现方式 |
|---|---|---|---|
| `RaInit=328002`，UDMA 注册继发 `-7` | SHMEM/HCCL 和 TileXR 重复拥有 RA；rank 间 owned/attached 状态可能不一致 | 通信 runtime 只能有一个 owner；允许显式 attach，并跨 rank 校验所有权 | TileXR-only、SHMEM -> TileXR、TileXR -> SHMEM 三组初始化顺序 A/B |
| 替换 `moonep.Buffer` 后模型仍失败 | MindSpeed 依赖 `_ctx`、token/route buffers、packed Prefetch/Reduce、caller buffer 等私有契约 | 用适配层显式桥接，不能只替换公共类名 | 从 traceback 沿 caller/adapter/native 三层逐项核对对象契约 |
| 权重被 `storage_offset != 0` 拒绝 | 权重是连续 arena 的合法 subview，随后还会复制到独立注册 backing | 区分 source view 和 registered backing；只在立即复制边界允许 offset | 打印 storage、data pointer、offset 和注册对象身份 |
| backward 要求 SHMEM view | zero-copy alias 缺少 owner、generation 和 plan 元数据 | 地址别名必须携带所有权和生命周期标签 | 验证 alias storage 相同，并校验 owner/plan/generation |
| Combine V1 `DecodeRoute=3002` | 4 字节 `DataCopyPad` 标量读取返回陈旧 UB 数据 | GM 小标量使用当前 CANN 版本实机验证过的读取模式 | 逐分支状态码加设备侧原值捕获；立即在 Combine 后同步状态 |
| PrefetchWeight 在 32 QP 返回 `-3` | 把物理 `qpNum` 当作 worker 数，只接受 1/2/4/8 | `workerCount=min(qpNum, maxWorkers)`，Kernel 仍接收完整 `qpNum` | 查询真实 QP 数并对照 Host 参数校验分支 |
| ReduceGrad workspace query 返回 `-1` | Host/Kernel QP 上限固定为 8，实机为 32 | Host、Kernel 和测试统一支持目标硬件 QP 上限 | 1/2/4/8/32 QP 参数矩阵 |
| ReduceGrad 第二轮 CQ 失败，`entryIdx=0x4000` | `entryIdx` 携带 SQ cycle，原实现错误地要求它小于 ring depth | 先按 depth 归一化，再依据绝对 SQ tail 计算完成 BB 数 | 两轮同 QP 精确复现；dump raw CQE、SQ head/tail、outstanding |
| Combine 后复用 plan 的反向 Dispatch 超时 | 旧 `status=3000` 违反 URMA 输入协议，异步 Python reset 又存在跨 stream 竞态 | consumer Host 在同一 stream reset；先检查旧首错误，不能掩盖真实失败 | 生产规模 oracle 对比不清理、异步清理、同步清理和 Host same-stream reset |
| 4K grouped Dispatch 超时或 SQ 满 | 全量 route 无法一次装入 UB，WQE 也不能一次塞入 SQ | route tiling、WQE 分批发布、每批 CQ 回收，以 `head-tail` 计算 outstanding | case 15 生产规模单算子和 H=7168 grouped oracle |

“重复 MR 注册泄漏”“完全没有 poll CQ”“peer 调度不对称”都曾是合理假设，但被后续
A/B、原始队列状态和成功对照推翻，不应继续作为既定根因传播。

## 关键测试如何定位根因

### 1. 初始化顺序 A/B

分别运行 TileXR-only、SHMEM -> TileXR 和 TileXR -> SHMEM。结果显示失败随初始化顺序
变化，把第一个失败边界从模型逻辑缩小到 RA 所有权，而不是 UDMA 数据面。

### 2. 两轮 ReduceGrad 精确复现

第一轮全部通过，第二轮仅部分 rank 失败。raw CQE 显示 `entryIdx=16384`、ring-local
tail 为 0、outstanding 为 1。16384 恰好是一个 SQ cycle，证明错误来自 cycle 归一化，
而不是超时长度、WQE 地址或 MR 注册。加入 modulo-depth 修复后，两轮 16/16 rank-round
通过；扩大 WQE 数量后仍通过。

### 3. 生产规模 grouped-URMA oracle

oracle 使用以下真实模型特征，而不是缩小到会绕开问题的 toy shape：

- 单机 8 卡、8 rank；
- `S=4096`、`K=8`、`H=7168`、EP8、`NvS=32768`；
- grouped-URMA，group width 16；
- model-skew 路由和 route weights；
- PrefetchWeight、Combine V2；
- 连续创建 5 个额外 plan 后复用最后一个 plan；
- 所有 rank 对 Dispatch 和 Combine 输出做 exact comparison。

第一次 Dispatch 和 Combine 正确，而 Combine 后复用 plan 的 Dispatch 失败。A/B 显示：

- 保留 `status=3000`：失败；
- Python `zero_()` 不同步：不稳定；
- Python `zero_()` 后显式同步：通过；
- Host 在 Dispatch stream 上 reset：稳定通过。

该测试把根因从宽泛的“反向 flag 超时”缩小为 plan status 的跨阶段和跨 stream 生命周期
错误。它同时验证了 4K route tiling、WQE 分批和 CQ 回收，因此是完整模型前最重要的
root-cause oracle。

### 4. 完整模型与性能复测

修复后按以下顺序扩大验证范围：

1. case 15：4K/EP8 grouped-URMA Dispatch 单算子复测。
2. grouped oracle：H=7168、model-skew、Prefetch、Combine V2、5 个额外 plan 和 plan reuse，
   8 rank exact comparison。
3. 完整正确性模型 `tilexr_urma_correctness_4k_8p_ep8_0812_015220`：8/8 迭代通过。
4. 关闭 DFX、trace、dump 和 profiler 后运行
   `tilexr_urma_perf_v2_4k_8p_ep8_0812_095543`：8/8 迭代、退出码 0、无 skip/NaN。

完整模型通过只能证明该组合路径闭环，不能替代单算子对边界语义的证明；单算子通过也
不能证明 plan reuse、前反向和多轮资源复用。

## 推荐调试流程

### 1. 固定运行身份

每次运行先保存以下信息：

- commit、工作树 diff 和源码哈希；
- 实际加载的 `.so`、AICore binary 路径和哈希；
- CANN、驱动、固件、SoC 和 conda 环境；
- launcher、环境变量、rank/device 映射；
- 运行前后的 NPU PID、stdout、stderr、plog 和输出目录。

构建成功不等于运行时加载了新库。没有二进制 provenance 的通过或失败结果都不能作为
最终证据。

### 2. 找第一个失败边界

按 Planner -> Dispatch -> Prefetch -> Combine -> reused Dispatch -> ReduceGrad 顺序，在
每个异步 stage 后只增加一个必要同步点。记录该点的 Host 返回码、`plan.status`、DFX
首错误和相关队列状态。后续超时往往只是更早的异步错误在同步点被暴露。

### 3. 一次验证一个可证伪假设

写清楚：

> X 导致症状 Y；如果成立，只改变 Z 后结果应从 A 变成 B。

优先改变初始化顺序、transport、是否 Prefetch、是否 Combine、是否复用 plan、是否同步
reset 等单一变量。一次同时修改 Kernel、Host、timeout 和路由，会使任何成功都无法归因。

### 4. 从协议状态入手，而不是先加 timeout

超时时至少采集：

- plan status 的阶段来源和预期值；
- flag 矩阵在 producer 前后、consumer 前后和超时后的差异；
- SQ/CQ head、tail、depth、owner、`entryIdx`、outstanding；
- 第一个失败 peer、QP、phase 和原始错误码。

延长 timeout 只有在状态持续前进但速度不足时才有意义。状态完全不变、已进入错误分支或
队列字段非法时，延长 timeout 只会降低调试效率。

### 5. 修复协议拥有者

- 跨 stream reset：由 consumer Host 在目标 stream 上排队。
- 跨 rank runtime：明确 owned/attached，并做一致性校验。
- 错误发布：使用 first-error/CAS，禁止后续 core 覆盖首错误。
- ring 字段：在类型、命名和测试中区分 absolute counter、ring index 和 cycle bit。
- Tensor：区分 source view、copy destination 和 registered backing。

### 6. 逐级扩大验证

推荐门禁顺序：

1. Host/unit/mock：参数、状态转换、边界算术。
2. 单 rank 或最小多 rank：验证 ABI 和第一条真实数据路径。
3. 两轮以上真实 UDMA：验证 QP、MR、SQ/CQ 和 magic 复用。
4. case 15：生产 route 数和 grouped-URMA 流控。
5. grouped oracle：model-skew、Prefetch、Combine、额外 plan、plan reuse、exact comparison。
6. 完整 4K/8P/EP8 前反向模型。
7. 关闭所有 DFX 后的独占性能运行。

## 必须长期保留的回归维度

- forward 和 backward 都执行；
- 同一个 plan 在 Combine 后被 Dispatch 复用；
- 至少两轮 MR、QP、SQ/CQ 和 magic 复用；
- route 数覆盖 UB tile 和 SQ depth 边界；
- QP 数覆盖 1/2/4/8/32；
- SQ 位置覆盖 `depth-1`、`depth`、`depth+1`、`2*depth`；
- 同时覆盖 1-BB 和多 BB WQE；
- balanced、model-skew、sparse 和 unique routing；
- `S=4096`、`K=8`、`H=7168`、EP8 的生产规模；
- 正确性运行开启失败时 DFX，性能运行关闭 trace、dump、DFX 和 profiler。

## 避免重复踩坑

- 不要把 error code 的字面含义直接当根因；查调用边界和相邻日志。
- 不要在第一次成功后停止，队列 cycle 和资源复用问题通常第二轮才出现。
- 不要用 Python 异步 tensor 操作实现 Host/Kernel 协议同步。
- 不要把 worker、QP、AIV block 和 rank 数视为同一种并行度。
- 不要用 toy shape 证明生产规模的 UB/SQ 容量安全。
- 不要依赖已消费 SQE 的内容推导 CQ completion。
- 不要在调试运行和性能运行之间复用未审计的环境变量。
- 不要因某次补丁通过完整模型就跳过最小 reproducer；最小 reproducer 才能证明因果。
- 不要删除被推翻的假设记录。保留否定证据可以防止后续重复猜测。

## 当前实现状态说明

本文记录的是已验证经验，不代表所有修复都已经进入 `main`。截至 2026-08-12：

- 已进入 `main`：MoonEP 上游 API 兼容、Combine V2 路由与 reduction、阶段性能报告。
- 当前调试分支中：same-stream status reset、32-QP Prefetch/ReduceGrad、grouped Dispatch
  route tiling/WQE/CQ 流控和相关测试。
- 任务补丁或 MindSpeed 工作树中：MindSpeed adapter、external communication owner、RA
  attach/ownership guard、历史 Combine V1 标量修复和通用 CQE cycle 修复。

开始新的调试任务时，应先检查这些修改是否已经进入当前目标分支和实际加载的二进制，
不能根据本文的历史状态假设代码已经包含修复。
