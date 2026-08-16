# MoonEP Combine V2 发送前全同步设计

状态：设计已固化并实现；2026-08-16 改为 Kernel 内部编译期开关且默认关闭，仅完成静态审查，尚未编译或运行测试

日期：2026-08-14

## 1. 目标

为 MoonEP Combine V2 提供一个默认关闭的发送前全同步模式，用于分析性能测试中明显的
快慢卡差异。该模式只能通过 Kernel 内部编译期常量控制。

开启后，每次 Combine V2 Kernel 在正常数据发送前执行以下协议：

1. 每个 active 发送核通过本核的 6 口 QP，向本核 schedule 负责的全部远端 peer
   各发送一个 32 Byte 同步信号；self 不发送、不写本地同步槽。
2. 每核的全部同步 WQE 作为一个 batch 构造并发布；每核只更新一次 SQ head、只敲一次
   doorbell，仅 batch 最后一个 WQE 请求 ordered completion。
3. 每个发送核先等待本地 CQ 确认该 batch 完成，再等待本核负责的全部远端 source 信号
   到达；self 不等待、不校验。
4. 完成一次仅包含 active 发送核的核间同步。
5. 所有 active 核进入现有正常发送阶段。

该模式由当前 AICore binary 的 Kernel 内部编译期开关决定，与 launch 是否执行 Reduce
无关。同一个 binary 中的全部 Combine V2 launch 使用同一配置。

## 2. 已确认决策

| 项目 | 决策 |
| --- | --- |
| 开关 | Kernel 内部编译期常量 `kEnableFullSync` |
| 默认行为 | `false`，不执行发送前全同步 |
| QP | 每核只使用其 6 口 QP，即 `MoonEpCombineV2Qp(core, MOONEP_COMBINE_V2_SIX_PORT) == core` |
| 提交 | 每核一个 WQE batch、一次 SQ head 更新、一次 doorbell |
| CQ | 仅 batch 最后一个 WQE 请求 ordered completion；发送核在等待远端信号前 poll CQ |
| self peer | 不发 WQE、不写 receive/source 信号槽、不等待、不校验 |
| 生效 launch | 常量为 `true` 时作用于所有 Combine V2 launch，与 `reduceHidden` 独立 |
| WQE 构造 | 新增独立 SIMT VF 函数，WQE 全部在 UB 构造 |
| 轮次隔离 | receive、source 和核间 barrier 槽均按两个 epoch 乒乓 |

## 3. 非目标

- 不改变 Combine V2 的数据路由、payload WQE、grant/done 协议和 Reduce 语义。
- 不改变 `TileXRMoonEpCombineV2()`、`TileXRMoonEpCombineStageV2()` 或
  `tools/moonep/test_npu_e2e.py` 的公共调用契约。
- 不为每个同步 WQE 分别请求 CQE；每个非空 batch 只产生一次 ordered completion。
- 不用 topology 推导额外 peer；peer 集合严格复用 Combine V2 runtime schedule。
- 不改变 Reduce 的启用条件；full-sync 和 Reduce 是两个独立阶段。

## 4. Kernel 内部开关与 ABI

### 4.1 编译期配置

Kernel 实现内部定义：

```cpp
constexpr bool kEnableFullSync = false;
```

Host 不读取环境变量，也不向 Kernel 传递 full-sync 参数。需要开启时，将该常量改为
`true` 并重新构建、嵌入和部署 AICore binary。`reduceHidden` 继续只控制 Reduce 阶段，
不参与 full-sync 配置。

当 StageV2 同时处理 hidden 和 route weight 且常量为 `true` 时，两次独立 Kernel launch
各执行一次 full-sync。若性能测试只需要一次 no-reduce 数据面，应直接运行单次 no-reduce
Combine V2，而不是用含两个 launch 的 StageV2 总耗时替代。

### 4.2 ABI 边界

- 公共 C API 和 MoonEP plan/tensor ABI 不变。
- `CombineV2LaunchContext` 不保存 full-sync 布尔状态。
- Host Kernel args 和 AICore Kernel 入口均不包含 full-sync 参数。
- 删除该 64-bit 参数后，`CombineV2KernelArgs` 固定为 21 个 64-bit 槽。

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
flag         = terminal task ? ORDERED_COMPLETION : 0
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
7. completion 账本递增一次，并用 `st_dev` 更新一次 `wqeCntAddr`；
8. 用 `st_dev` 向 `dbAddr` 写一次新 head，完成唯一一次 doorbell；
9. poll CQ，直到 terminal ordered completion 覆盖整个 batch，并校验 SQ 已回收为空。

`batchCount == 0` 的 self-only 核不更新 head、不敲 doorbell。

### 7.4 CQ 与账本

同步 batch 只在最后一个 WQE 设置 ordered-completion flag，因此每个非空 batch：

- `completionCount` 和 `wqeCntAddr` 只增加一次；
- `cqTarget` 推进一个 completion；
- doorbell 后调用 `PollCqOnce()`，直到 CQ 到达目标；
- CQ 成功后要求 `tail == submittedHead` 且 `head == tail`，再进入远端信号等待。

这样 full-sync 自己完成 SQ/CQ 回收，不依赖后续正常发送 WQE。`batchCount == 0` 的
self-only 核不发布 WQE，也不等待 CQ。

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
-> if kEnableFullSync:
       BuildFullSyncWqesVf
       if WQE count > 0: PrepareFullSyncSignal
       PublishFullSyncBatch（6 口 QP，一次 DB，一个 terminal completion）
       WaitFullSyncCq
       WaitFullSyncSources（跳过 self）
       FullSyncActiveCoreBarrier
-> PrefillOperatorWqes
-> 现有 step send / CQ / grant
-> WaitInboundDone
-> ReduceHidden
```

默认关闭时除 `kEnableFullSync` 分支判断外，执行顺序和数据面行为保持不变。

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

full-sync 协议代码保留 lane/WQ/SQ/CQ context、空闲 head/tail 和超时检查。常量为
`false` 时整个同步分支不执行；改为 `true` 后，初始化失败的 core 不进入同步发布，其他
参与方由 full-sync timeout 有界退出。

terminal ordered completion 提供本地提交完成和 CQ status 观测；远端仍需依赖完整信号
匹配确认目标槽已可见。CQ 错误、CQ 超时和远端信号超时分别保留各自失败边界。

## 11. Profiling 与性能判定

当 Combine V2 profiling 打开时，增加以下 time point：

```text
FULL_SYNC_BEGIN
FULL_SYNC_SUBMIT_END
FULL_SYNC_RECEIVE_END
FULL_SYNC_CORE_BARRIER_END
```

新增时间点后有效数量从 22 增至 26，time-point capacity 从 32 增至 40，record 从
384 Byte 增至 448 Byte。当前 profile ABI 后续又加入 Fullmesh 诊断字段，版本为 5；Host
layout、Kernel writer、hardware probe 和 trace parser 必须共同使用结构常量。关闭
full-sync 时连续记录这四个时间点，使阶段表现为近零时长，同时保留正值、单调的累计
时间戳。

性能结论至少同时报告：

- full-sync 关闭的原始 baseline；
- full-sync 开启后的同步阶段耗时；
- Kernel 总耗时 P50/P90/max；
- rank 间 `max - min` 和 max/P50；
- 正常 send、inbound wait 和 Reduce 分段耗时。

全同步降低快慢卡差异不等于提高吞吐；必须分别陈述抖动改善和总耗时开销。

## 12. 验证策略

### 12.1 Host/unit

- Kernel 内部常量存在且默认值为 `false`；
- Host context、Kernel args、Kernel 入口和 `Init()` 均不包含 full-sync 布尔参数；
- `CombineV2KernelArgs` 大小与 21 个参数的 Kernel 声明一致；
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
- 仅 terminal WQE 设置 completion flag，每个 batch 只更新一次 `wqeCntAddr` 并 poll 一次 CQ；
- 每核同步路径只有一个 DB store；
- CANN 9.1、Ascend950/910A5 target compile 通过。

### 12.3 硬件

设备相关验证前先在相同设备子集和拓扑运行官方 HCCL Test；AIV peer-memory 基线使用
`all_reduce_test -a aiv_only`，并按每 rank correctness 判断结果。基线不健康时停止
TileXR 归因。

在 A5/Ascend950 上按以下顺序验证：

1. 分别构建常量为 `false/true` 的 binary，在 2P 和 8P 小 shape 做 exact comparison；
2. 连续至少三轮 hidden Combine，覆盖 epoch0 -> epoch1 -> epoch0；
3. 检查 self 对应槽不被本 rank 写入且不参与等待；
4. 检查每个非空同步 batch 恰好产生一次 CQ completion，且进入远端等待前 SQ 已回收为空；
5. 16P 和目标多机规模，验证每核单 batch/单 DB 以及全 source 覆盖；
6. 生产 shape `BS=8192, topK=16, H=3584` 正确性；
7. 关闭 profiling/DFX 后分别运行两种 binary，比较 rank 离散度和总耗时。

远端验证脚本通过 mutagen `one-way-safe` 同步到服务器，再由 SSH 执行。NPU 占用检查遵循
项目 `AGENTS.md`：其他 TileXR 任务可共享；Python 或通信测试进程占用时每 15 秒重试，
120 秒仍不可用则停止本轮验证。

## 13. 风险与约束

- 同步 batch 的 terminal CQ 只能证明本地 ordered completion 成功；远端仍必须通过完整
  magic-tagged 信号确认可见性，不能用 CQ 代替远端等待。
- 同步 WQE占用现有 6 口 SQ；进入正常发送前必须完成 terminal CQ 并验证 SQ 为空，避免
  full-sync 账本泄漏到后续数据面。
- 所有 rank 必须部署同一配置的 AICore binary；混用开启和关闭版本可能让开启方在
  full-sync 超时，而关闭方进入正常协议后继续等待 grant/done。
- StageV2 同时包含 hidden 和 route-weight 时，常量为 `true` 会产生两次全同步，性能
  口径必须明确是单 launch 还是完整 StageV2。
- 该协议证明所有 rank 已到达当前 Combine V2 launch 的 send 前边界，但不保证各 rank
  在完全相同 cycle 开始发送。它是分布式 barrier 语义，不是时钟级 simultaneous launch。
- Host/unit/simulator 不能证明 UDMA ordered completion、远端可见性和 doorbell 时序；最终结论必须限定在实际
  验证的 A5/Ascend950 硬件、rank 数和拓扑。

## 14. 完成标准

实现只有同时满足以下条件才可标记完成：

- Kernel 内部开关默认关闭，Host 和公共 API 不暴露控制参数；
- launch-independent、6 口 QP、跳过 self、32 Byte、双 epoch 均有 unit evidence；
- WQE 全 UB 构造、MTE3 发布、一次 DB、terminal CQ 的 source/compile guard 通过；
- 至少三轮硬件正确性覆盖 epoch 复用；
- 正常发送完成后 SQ/CQ 账本无泄漏；
- full-sync off/on 的目标规模性能数据完整，并分别报告同步成本与快慢卡差异。

## 15. 当前实现与验证边界

当前代码已实现默认关闭的 Kernel 内部编译期开关、双 epoch workspace、独立 SIMT WQE
构造、6 口 QP 发布、远端 source 等待、active-core 软件 barrier，以及 profile/trace
解析。静态契约覆盖外部参数移除、默认关闭、self 排除、peer/source 覆盖、ping-pong
索引和 WQE 发布约束。

本次按任务要求未执行构建、单元测试、Python trace 测试、模拟器或 A5/Ascend950 硬件
验证。因此第 12 节中的 compile、测试、SQ/CQ 回收、三轮 epoch 复用和性能数据仍是后续
验收项，不能据当前静态审查宣称运行时正确性或性能改善。
