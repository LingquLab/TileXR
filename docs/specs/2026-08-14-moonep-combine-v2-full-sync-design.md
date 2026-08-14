# MoonEP Combine V2 发送前全同步设计

状态：设计已固化并实现；仅完成静态审查，尚未编译或运行测试

日期：2026-08-14

## 1. 目标

为 MoonEP Combine V2 增加一个默认开启的发送前全同步模式，用于分析和缓解性能测试中
明显的快慢卡差异。

开启后，hidden Combine Kernel 在正常数据发送前执行以下协议：

1. 每个 active 发送核通过本核的 6 口 QP，向本核 schedule 负责的全部远端 peer
   各发送一个 32 Byte 同步信号；self 不发送、不写本地同步槽。
2. 每核的全部同步 WQE 作为一个 batch 构造并发布；每核只更新一次 SQ head、只敲一次
   doorbell，不请求 CQE。
3. 每个发送核等待本核负责的全部远端 source 信号到达；self 不等待、不校验。
4. 完成一次仅包含 active 发送核的核间同步。
5. 所有 active 核进入现有正常发送阶段。

该模式只作用于 `TileXRMoonEpCombineStageV2()` 的 hidden launch。可选的 route-weight
launch 以及直接调用的非 Reduce Combine V2 launch 不执行全同步。

## 2. 已确认决策

| 项目 | 决策 |
| --- | --- |
| 开关 | Host 环境变量 `TILEXR_MOONEP_COMBINE_V2_FULL_SYNC` |
| 默认行为 | 未设置或空串时开启；显式 `0/false/off` 时关闭 |
| QP | 每核只使用其 6 口 QP，即 `MoonEpCombineV2Qp(core, MOONEP_COMBINE_V2_SIX_PORT) == core` |
| 提交 | 每核一个 WQE batch、一次 SQ head 更新、一次 doorbell |
| CQ | 同步 WQE 不请求 CQE，不单独 poll CQ |
| self peer | 不发 WQE、不写 receive/source 信号槽、不等待、不校验 |
| 生效 launch | 仅 `reduceHidden == true` 的 hidden launch |
| WQE 构造 | 新增独立 SIMT VF 函数，WQE 全部在 UB 构造 |
| 轮次隔离 | receive、source 和核间 barrier 槽均按两个 epoch 乒乓 |

## 3. 非目标

- 不改变 Combine V2 的数据路由、payload WQE、grant/done 协议和 Reduce 语义。
- 不改变 `TileXRMoonEpCombineV2()`、`TileXRMoonEpCombineStageV2()` 或
  `tools/moonep/test_npu_e2e.py` 的公共调用契约。
- 不给 route-weight launch 增加同步。
- 不为同步 WQE 增加 CQ、notify 或单独的完成队列回收流程。
- 不用 topology 推导额外 peer；peer 集合严格复用 Combine V2 runtime schedule。
- 不改变 route-weight 和直接非 Reduce launch 的默认数据面；默认同步只作用于 hidden launch。

## 4. 开关和 Host 传递

### 4.1 环境变量

Host 在每次 Combine V2 launch 准备阶段读取：

```text
TILEXR_MOONEP_COMBINE_V2_FULL_SYNC
```

解析规则：

- 未设置、空串：开启；
- `0`、`false`、`FALSE`、`off`、`OFF`：关闭；
- `1`、`true`、`TRUE`、`on`、`ON`：开启；
- 其他值：打印明确错误并返回 `TILEXR_MOONEP_ERROR_INVALID_ARGUMENT`，避免性能运行
  因拼写错误静默落回关闭状态。

最终 Kernel 参数为：

```text
fullSync = envEnabled && reduceHidden
```

因此 `TileXRMoonEpCombineStageV2()` 中：

- hidden 参数设置 `reduceHidden=true`，执行全同步；
- route-weight 参数复用时设置 `reduceHidden=false`，不执行全同步。

所有 rank 必须使用相同的开关值。当前 Host API 没有跨 rank 环境一致性检查；若部分 rank
开启而部分 rank 关闭，开启方将按 full-sync 超时退出，但已进入正常阶段的关闭方仍可能
阻塞在既有 grant/done 协议。启动脚本必须统一导出该变量。

### 4.2 ABI 边界

- 公共 C API 和 MoonEP plan/tensor ABI 不变。
- `CombineV2Params` 或 `CombineV2LaunchContext` 增加内部布尔状态。
- Host 与嵌入式 AICore binary 同步增加一个 64-bit Kernel 参数 `fullSync`。
- `CombineV2KernelArgs` 大小断言和 Kernel 声明同步更新。

## 5. Peer 与 core 映射

### 5.1 发送集合

对 rank `r`、active core `c`：

```text
for step in [0, MoonEpCombineV2StepCount(rankSize)):
    peer = MoonEpCombineV2Peer(r, step, c, rankSize, scheduleMode)
    if peer is valid and peer != r:
        append peer
```

当前 single-ring schedule 下，每个 `(source rank, destination rank)` 恰好由一个发送核
覆盖。每核最多处理 8 个 step，因此同步 batch 最多包含 8 个 WQE。self 所在的
`(core, step)` 被压缩掉，不在 SQ 中留下空 WQE。

全 rank 汇总后，每个 rank 向其余 `rankSize - 1` 个 rank 各发送一个信号。

### 5.2 接收集合

每个 core 的等待集合与本核发送集合相同：

```text
for step in [0, MoonEpCombineV2StepCount(rankSize)):
    peer = MoonEpCombineV2Peer(rank, step, core, rankSize, scheduleMode)
    if peer is valid and peer != rank:
        wait receive[epoch][peer]
```

因此每个发送核只等待其负责的远端 peer，self 不进入等待集合；所有 active 核的等待集合
合并后恰好覆盖全部远端 source，且无重复。

增加一个可在 Host unit test 使用的 schedule helper，计算某个 source 向当前 destination
发送同步信号时所使用的 core。接收信号同时校验该 `sourceCore`，用于发现 schedule 或
槽地址接错，而不仅仅是等待任意 magic。

## 6. Workspace 布局

### 6.1 信号格式

```cpp
struct alignas(32) MoonEpCombineV2FullSyncSignal {
    uint64_t magic;
    uint32_t marker;
    uint32_t sourceRank;
    uint32_t sourceCore;
    uint32_t rankSize;
    uint64_t guard;
};
```

约束：

- `sizeof(MoonEpCombineV2FullSyncSignal) == 32`；
- `marker` 使用独立的 Combine V2 full-sync 常量；
- 接收方校验 `magic`、`marker`、`sourceRank`、`sourceCore`、`rankSize` 和
  由 magic/marker 派生的 `guard`；
- 每个物理槽占 64 Byte，WQE 仅传前 32 Byte，避免相邻 source 共享 cache line。

### 6.2 新增区域

在现有 `failure` 区之后、`output` 区之前增加：

```text
fullSyncReceive[2][128]  // 每槽 64 B，远端写入
fullSyncSource[2][16]    // 每槽 64 B，本核 WQE 的本地 SGE
fullSyncBarrier[2][16]   // 每槽 64 B，本 rank active-core barrier
```

固定字节数：

```text
receiveBytes = 2 * 128 * 64 = 16384 B
sourceBytes  = 2 * 16  * 64 =  2048 B
barrierBytes = 2 * 16  * 64 =  2048 B
totalAdded                         20480 B
```

`CombineV2Layout` 增加对应 offset/bytes 字段，并继续执行 checked add、64 Byte 区域对齐
和最终 2 MiB 注册区对齐。`outputOffset` 及其后布局相应后移；scratch epoch、done、grant、
control-source 和 failure 的既有索引语义不变。

### 6.3 乒乓和生命周期

```text
epoch = MoonEpCombineV2Epoch(magic) = magic & 1
receiveIndex = epoch * 128 + sourceRank
sourceIndex  = epoch * 16  + sourceCore
barrierIndex = epoch * 16  + core
```

所有等待均匹配完整 `magic`，不依赖槽清零。两个 epoch 防止相邻轮次复用同一 cache line；
完整 magic 防止同一 epoch 在两轮后复用时误认陈旧信号。

有远端 WQE 的 core 才在本轮发布前写 source 槽并 clean；self-only core 不写 source 槽。
source 槽在本轮 Kernel 内不再修改。隔轮复用安全性的
协议依据是：轮次 `N+1` 的信号只有在各 rank 完成轮次 `N` 的全同步后才能发送；两个
epoch 因而为仍在读取轮次 `N` source 的设备提供至少一个完整轮次的隔离。

## 7. WQE 构造与发布

### 7.1 新 SIMT 函数

新增独立函数，例如：

```cpp
MoonEpCombineV2BuildFullSyncWqesVf(...)
```

标记为 `__simt_vf__ __aicore__`，沿用现有 128-thread builder launch 约束，只有
`task < peerCount` 的前 8 个以内 task 构造 WQE。Scalar AICore 先把最多 8 个远端 peer
的以下字段整理到 UB context：

- remote EID、token、target hint、TP/segment 信息；
- 远端 `fullSyncReceive[epoch][rank]` 地址；
- 本地 `fullSyncSource[epoch][core]` 地址；
- batch 起始 absolute SQ head。

SIMT task 一一对应压缩后的远端 peer，并在 UB 中完整清零和构造 64 Byte WRITE WQE：

```text
opcode       = WRITE
flag         = 0
nf           = 0
inlineMsgLen = 0
sgeNum       = 1
sge.len      = 32
sge.va       = local source signal
remoteAddr   = target receive slot
sqeBbIdx     = (absoluteHead + task) % SQ_BB_COUNT
owner        = derived from absoluteHead + task
```

禁止在 SQ 中用 scalar/direct-GM store 构造或补丁 WQE。SIMT 返回后，WQE 仍只存在于 UB。

### 7.2 UB 复用

同步发生在 route selection 之前，可复用现有发送阶段 UB：

- `wqeIssueBuf_` 的 6 口区域前 8 个 WQE slot 保存同步 WQE；
- `dstSlotBuf_` 的前部临时保存同步 descriptor/context；
- 正常发送前重新执行 `PrefillOperatorWqes()`，覆盖同步 WQE；
- 随后的 `LoadSelectionChunk()` 覆盖临时 descriptor。

因此本设计不增加常驻 UB 预算，也不改变当前 216 KiB send-buffer 上限。实现必须调整
`InitBuffers()`/`Process()` 顺序，保证 operator prefill 发生在全同步完成之后。

### 7.3 一次发布的精确定义

每核使用 `lane_[MOONEP_COMBINE_V2_SIX_PORT]`，即本核 6 口 shared QP：

1. 检查 `head - tail + batchCount` 不超过 SQ 可用范围；
2. `S_MTE3` 依赖后，把整个 batch 从 UB 搬到 SQ；
3. 若 SQ 尾部空间不足，允许 `CopyIssueToSq()` 生成两段 MTE3 ring copy；这仍是一个
   逻辑 batch；
4. 等待对应 `MTE3_S`，证明全部 WQE 已写入 SQ；
5. `head += batchCount`；
6. 用 `st_dev` 更新一次 `headAddr`；
7. 用 `st_dev` 向 `dbAddr` 写一次新 head，完成唯一一次 doorbell。

`batchCount == 0` 的 self-only 核不更新 head、不敲 doorbell。

### 7.4 CQ 与账本

同步 WQE `flag=0`，因此：

- 不增加 `completionCount`；
- 不写 `wqeCntAddr`；
- 不推进 `cqTarget`；
- 不调用 `PollCqOnce()`；
- 仅推进本地 lane state 和设备 `headAddr`。

正常发送阶段继承更新后的 absolute head。该 QP 后续已有的 ordered-completion WQE
生成 CQE 时，其完成 SQ tail 覆盖此前的同步 WQE，从而一并回收 SQ 空间。不能在全同步
与正常发送之间重新从 GM 读取旧 head 或重置 lane state。

对于有同步 WQE 的 core，schedule 后续必然还包含同一 6 口 QP 的远端正常发送和
ordered-completion 控制 WQE。self-only core 没有同步 WQE，因此不存在无人回收的同步
SQE。

## 8. 等待和核间同步

### 8.1 远端信号等待

每核轮询其发送 schedule 对应 peer 的远端槽：

1. 对槽执行 cache clean-and-invalidate；
2. 读取完整 32 Byte 信号；
3. 校验 magic、marker、source rank、预期 source core 和 rank size；
4. 匹配后标记该 source ready；
5. self 不进入等待集合，不访问 buffer。

每个发送核必须先完成本核全部 WQE 的一次发布，再进入接收轮询。因为每个 rank 的所有
active core 最终合并等待全部远端 source，所以某个 rank 完成接收阶段时，可以证明
所有其他 rank 都已经进入本轮同步并发布了自己的到达信号。

### 8.2 active-core 软件 barrier

不直接使用 `SyncAll<true>()`。现有 API 允许 `aivCoreNum >= 16`，而 runtime 的
`activeCoreCount` 在 2-8P 时只有 rank 数、在 16P 以上固定为 16；Kernel 入口还会让
多余 block 提前返回。硬件全 block barrier 会把 launch block 数和发送核数错误绑定，
存在死锁风险。

采用 workspace 软件 barrier：

1. 每个 active core 在完成自己的全部远端 source 等待后，向
   `fullSyncBarrier[epoch][core]` 写入带 magic 的 32 Byte记录并 clean；
2. 每个 active core invalidate 并等待 `[0, activeCoreCount)` 的全部记录匹配本轮 magic；
3. barrier 完成后才允许调用 `PrefillOperatorWqes()` 并进入 step0 正常发送。

该 barrier 只包含实际发送核，不包含 `aivCoreNum` 中提前返回的 block。每核必须写自己的
barrier 到达记录供其他核观察，但本核直接将自身视为 ready，不回读校验自己的记录；上文
的 self 排除规则专指跨 rank peer 信号。

## 9. 执行顺序

Kernel 主路径调整为：

```text
Init / InitLaneStates
-> InitBuffers（只分配，不预填正常 WQE）
-> 既有配置、poison 和 destination 检查
-> if fullSync:
       BuildFullSyncWqesVf
       if WQE count > 0: PrepareFullSyncSignal
       PublishFullSyncBatch（6 口 QP，一次 DB，无 CQ）
       WaitFullSyncSources（跳过 self）
       FullSyncActiveCoreBarrier
-> PrefillOperatorWqes
-> 现有 step send / CQ / grant
-> WaitInboundDone
-> ReduceHidden
```

关闭开关时除 `fullSync` 分支判断外，执行顺序和数据面行为保持不变。

## 10. 失败处理

全同步是可选性能模式，但启用后属于 Kernel 正确性协议，不能随当前
`kEnableSafetyChecks=false` 一起移除以下检查：

- SQ 容量和 WQ context 合法性；
- remote registration 对 32 Byte 目标范围的覆盖；
- signal 字段完整匹配；
- remote source wait 超时；
- active-core barrier 超时。

新增失败状态：

```text
MOONEP_COMBINE_V2_FULL_SYNC_TIMEOUT
MOONEP_COMBINE_V2_FULL_SYNC_BARRIER_TIMEOUT
```

超时从进入 full-sync 阶段时单独取起始 cycle，避免 Host/prepare 时间消耗同步预算。超时
记录至少包含 source/peer、core、expected magic 和 observed magic；失败核不得进入正常
发送。其他 rank 若因该失败收不到信号，也会在有界时间内退出，而不是永久挂起。

实现中 lane/WQ/SQ/CQ context 及空闲 head/tail 检查采用
`kEnableSafetyChecks || fullSync` 条件；因此即使 trusted benchmark 关闭通用安全检查，
默认开启的 full-sync 仍保留其发布前置条件。初始化失败的 core 不进入同步发布，其他
参与方由 full-sync timeout 有界退出。

由于同步 WQE明确不产生 CQ，远端写错误无法通过 CQ status 直接报告；信号等待超时是该
阶段的数据面失败观测边界。这是不开 CQ 的已接受代价。

## 11. Profiling 与性能判定

当 Combine V2 profiling 打开时，增加以下 time point：

```text
FULL_SYNC_BEGIN
FULL_SYNC_SUBMIT_END
FULL_SYNC_RECEIVE_END
FULL_SYNC_CORE_BARRIER_END
```

新增时间点后有效数量从 22 增至 26，后续 8 个诊断槽要求容量至少为 34。因此
time-point capacity 从 32 增至 40，profile version 从 3 增至 4，record 从 384 Byte
增至 448 Byte；Host layout、Kernel writer、hardware probe 和 trace parser 必须共同使用
结构常量。关闭 full-sync 时连续记录这四个时间点，使阶段表现为近零时长，同时保留正值、
单调的累计时间戳。

性能结论至少同时报告：

- full-sync 关闭的原始 baseline；
- full-sync 开启后的同步阶段耗时；
- Kernel 总耗时 P50/P90/max；
- rank 间 `max - min` 和 max/P50；
- 正常 send、inbound wait 和 Reduce 分段耗时。

全同步降低快慢卡差异不等于提高吞吐；必须分别陈述抖动改善和总耗时开销。

## 12. 验证策略

### 12.1 Host/unit

- 环境变量开、关、未设置和非法值解析；
- `reduceHidden=true/false` 的 hidden-only 传递；
- 公共 ABI 静态断言保持不变，内部 Kernel args 大小同步变化；
- workspace 新区域 offset、大小、64 Byte 隔离、checked overflow 和 2 MiB 总对齐；
- 2/8/16/32/64/128P 下每核 remote peer 数不超过 8；
- 每 rank 总发送和总等待数均为 `rankSize - 1`；
- self 始终不进入发送或等待集合；
- source-to-sender-core helper 与 `MoonEpCombineV2Peer()` 互逆；
- epoch0、epoch1、epoch0 三轮索引与 exact-magic 判定。

### 12.2 Source guard / compile

- 新 WQE builder 是独立 `__simt_vf__`；
- WQE 在 UB 完整构造，通过 MTE3 发布；
- MTE3 完成后才执行 `st_dev`；
- 只有 6 口 QP；
- 同步路径没有 completion flag、`wqeCntAddr` 更新或 CQ poll；
- 每核同步路径只有一个 DB store；
- CANN 9.1、Ascend950/910A5 target compile 通过。

### 12.3 硬件

设备相关验证前先在相同设备子集和拓扑运行官方 HCCL Test；AIV peer-memory 基线使用
`all_reduce_test -a aiv_only`，并按每 rank correctness 判断结果。基线不健康时停止
TileXR 归因。

在 A5/Ascend950 上按以下顺序验证：

1. 2P 和 8P 小 shape，开关关闭/开启均 exact comparison；
2. 连续至少三轮 hidden Combine，覆盖 epoch0 -> epoch1 -> epoch0；
3. 检查 self 对应槽不被本 rank 写入且不参与等待；
4. 检查同步前后 CQ tail 不因 full-sync 增加，正常发送 CQ 完成后 SQ tail 覆盖同步 head；
5. 16P 和目标多机规模，验证每核单 batch/单 DB 以及全 source 覆盖；
6. 生产 shape `BS=8192, topK=16, H=3584` 正确性；
7. 关闭 profiling/DFX 后分别运行 full-sync off/on 性能测试，比较 rank 离散度和总耗时。

远端验证脚本通过 mutagen `one-way-safe` 同步到服务器，再由 SSH 执行。NPU 占用检查遵循
项目 `AGENTS.md`：其他 TileXR 任务可共享；Python 或通信测试进程占用时每 15 秒重试，
120 秒仍不可用则停止本轮验证。

## 13. 风险与约束

- 同步 WQE 无 CQ，错误只能由远端信号超时暴露；这是本设计最主要的诊断限制。
- 同步 WQE占用现有 6 口 SQ，必须让后续正常 CQ completion 覆盖其 absolute head，任何
  lane state 重置都会造成 SQ 账本泄漏。
- 所有 rank 的环境变量必须一致；混用配置可能让开启方在 full-sync 超时，而关闭方进入
  正常协议后继续等待 grant/done，不能把该误配置视为可靠的全局错误收敛机制。
- 开关只同步 hidden launch；不能据此推断 route-weight launch 的 rank 到达一致性。
- 该协议证明所有 rank 已到达 hidden send 前边界，但不保证各 rank 在完全相同 cycle
  开始发送。它是分布式 barrier 语义，不是时钟级 simultaneous launch。
- Host/unit/simulator 不能证明无 CQ UDMA 写和 doorbell 时序；最终结论必须限定在实际
  验证的 A5/Ascend950 硬件、rank 数和拓扑。

## 14. 完成标准

实现只有同时满足以下条件才可标记完成：

- 开关默认开启且公共 API/调用契约不变；
- hidden-only、6 口 QP、跳过 self、32 Byte、双 epoch 均有 unit evidence；
- WQE 全 UB 构造、MTE3 发布、一次 DB、无 CQ 的 source/compile guard 通过；
- 至少三轮硬件正确性覆盖 epoch 复用；
- 正常发送完成后 SQ/CQ 账本无泄漏；
- full-sync off/on 的目标规模性能数据完整，并分别报告同步成本与快慢卡差异。

## 15. 当前实现与验证边界

当前代码已实现 Host 开关解析和 hidden-only 传递、双 epoch workspace、独立 SIMT WQE
构造、6 口 QP 单 batch/单 head/单 DB 发布、无 CQ 账本、远端 source 等待、active-core
软件 barrier，以及 profile/trace 解析。静态契约覆盖默认开关、self 排除、peer/source
覆盖、ping-pong 索引和 WQE 发布约束。

本次按任务要求未执行构建、单元测试、Python trace 测试、模拟器或 A5/Ascend950 硬件
验证。因此第 12 节中的 compile、测试、SQ/CQ 回收、三轮 epoch 复用和性能数据仍是后续
验收项，不能据当前静态审查宣称运行时正确性或性能改善。
