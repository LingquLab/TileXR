# MoonEP Combine V2 Server-Pair、64-Grant 与 16 核 SyncAll 迁移方案及执行计划

状态：方案已细化，待确认后按 R0-R8 串行实施和验收

日期：2026-08-16

参考方案：

```text
C:/Users/h00580772/.codex/worktrees/f9aa/tileXR/docs/plans/
2026-08-16-moonep-combine-v2-ring-to-server-pair-mapping.md
```

本文件吸收上述方案，并固化以下修订：

1. Combine V2 的 Runtime launch blockDim 固定为 16；
2. 卡内同步改为 `AscendC::SyncAll<true>()`；
3. 新 Server-Grant 每个等待卡仍要求 64 个 Grant，但每个核只轮询其中 4 个；
4. 16 个核的 Grant 分片互不重叠，合并后恰好覆盖 64 个槽；
5. Grant 分片完成后通过 `SyncAll<true>()` 汇合，失败状态通过显式共享状态聚合；
6. boundary ID 按执行时间单调编号，phase barrier 不再与 round barrier 冲突；
7. R6-R8 的 64-Grant 发布、shadow、admission 和移除 round barrier 被纳入同一完整计划。

## 1. 执行结论

本次迁移不能作为一次 peer 公式替换完成。正确的实施顺序是：

```text
固定 16 核 launch 和无分歧 SyncAll 控制流
-> 在旧 Ring 上验证可重复全局 barrier
-> 32P+ 停用旧 Ring Grant
-> 切换同半区/跨半区 Server-Pair
-> 增加独立 phase barrier
-> 切换最终奇偶 Server-Pair
-> 接入 64-Grant shadow
-> 64-Grant 成为 admission 条件
-> 移除每 round 全局 barrier
```

对应版本为：

| 版本 | 主要变化 | 当前 step admission | 必须保持 |
| --- | --- | --- | --- |
| R0 | 当前 bidirectional Ring + Fullmesh | Legacy Grant | 当前正确性和 CQ/SQ 基线 |
| R1 | 固定 16 核、`SyncAll<true>()` 汇合、可重复 leading/round barrier | Legacy Grant + round barrier | active peer 仍为 Ring |
| R2 | 32P+ 停用 Legacy Grant | round barrier | 2P-16P Legacy 路径不变 |
| R3 | 激活 same/cross Server-Pair | round barrier | 不引入 phase barrier 和新 Grant |
| R4 | 增加独立 phase barrier | round + phase barrier | peer 仍为 same/cross |
| R5 | 只切换 parity Server-Pair | round + phase barrier | pair/core/route 不变 |
| R6 | 发布并 shadow 校验新 64-Grant | round barrier 先完成，Grant 后校验 | Grant 不负责 admission |
| R7 | Grant wait 前移并参与 admission | Grant + 冗余 round barrier | 两种门控结果必须一致 |
| R8 | 删除 round barrier | Grant + phase barrier | leading barrier 和 phase barrier 保留 |

禁止把 R2、R3 和 R6 合并。否则出现挂死、错数或 CQ 残留时，无法区分是 Grant、peer、
receive inverse、Done 还是 barrier 的问题。

## 2. 权威基线与保护边界

计划编写时确认的代码基线：

```text
repository HEAD = 8ad32059eca4d558a297155b165ea2c23200ae68
kernel SHA-256  = b7aad8eb558f4ed444f81f7506fa8fab21ed6b9e6e98a5de91037655c2a27f4d
```

当前工作区存在用户未提交修改。实施时必须保留它们，尤其是
`WaitStepCqs()` 中 shared-QP 的以下终态校验：

```text
cqTail == cqTarget
tail == submittedHead
head == tail
```

每个阶段开始前记录：

```bash
git rev-parse HEAD
git status --short
sha256sum src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h
```

若 HEAD、kernel hash 或重叠文件发生变化，应重新核对调用链和差异，不按本文位置机械
套用，也不回退用户修改。

## 3. 目标、范围和非目标

### 3.1 目标

对 `rankSize in {32, 64, 128}`：

- 用 Server-Pair 替换旧 Ring peer 顺序；
- 保持每核每 round 只处理一个 peer；
- 保持每个 source 的完整 All2All 覆盖；
- 保持 Self、同 server Fullmesh、跨 server shared-QP 三条数据路径；
- 建立最终 parity 模式下的 server-quarter 64-Grant admission；
- 每核仅轮询 `64 / 16 = 4` 个 Grant；
- 用固定 16 核的硬件全核同步完成卡内汇合；
- 通过分阶段硬件证据证明每一步，而不是一次性替换。

### 3.2 兼容范围

- 公共 `Buffer`、`MoonEPCommPlan`、Combine V2 方法名和签名不变；
- `TileXRMoonEpCombineV2()`、`TileXRMoonEpCombineStageV2()` 参数顺序不变；
- `aivCoreNum` 参数继续保留，但合法值从“至少 16”收紧为“必须等于 16”；
- Host 传给 Runtime 的 blockDim 固定使用 `kMoonEpCombineV2CoreCount`；
- Kernel 入口参数数量和顺序不变；
- `tools/moonep/test_npu_e2e.py` 的调用、关键字、返回 tuple 和 zero-copy/async 合同不变；
- 2P-16P 继续使用旧 Ring 和 Legacy Grant；固定启动的非工作核只参与必要的卡内集合点。

### 3.3 非目标

- 不重写数据选择、payload、scratch 地址或 Reduce 主体；
- 不修改 Fullmesh 建链、QP slot、注册和数据 WQE；
- 不修改 shared-QP 的 6 口/2 口 payload 分流；
- 不修改 Done workspace ABI 和 Done lane 数；
- 不新增公共运行时开关；
- 不从 topology 单独推断 memory/UDMA 数据路径；
- 不用放宽 timeout、跳过 CQ/Done 或关闭错误检查掩盖失败；
- 本计划本身不实施代码，也不提交 Git。

## 4. 固定 16 核 launch 合同

### 4.1 Host 合同

Host validation 改为：

```text
params.aivCoreNum == kMoonEpCombineV2CoreCount == 16
device vectorCoreCount >= 16
```

大于 16 和小于 16 都返回参数错误或不支持。公共头文件注释同步改成“必须等于 16”。
Runtime launch 使用：

```cpp
kMoonEpCombineV2CoreCount
```

而不是继续透传任意 `params.aivCoreNum`。Launch unit test 必须捕获并断言
`capturedBlockDim == 16`。

当前主要 Python runtime 已传入 16，但仍要保留 Host 拒绝非法值的测试，避免其他 C API
调用方绕过约束。

### 4.2 Kernel 合同

Kernel 要求：

```text
GetBlockNum() == 16
core = GetBlockIdx() in [0, 16)
```

`tilexr_moonep_combine_v2_kernel.cpp` 不再允许 block 因
`core >= activeCoreCount` 在入口提前返回。否则剩余 block 到达 `SyncAll<true>()` 后会永久
等待。

对 2P-8P，必须区分：

```text
launchCoreCount = 16
activeWorkerCount = rankSize
```

非 active worker 不执行 peer、QP、payload、Done 或 Reduce 工作，但仍按完全相同的顺序
到达所有卡内 `SyncAll<true>()`。16P 及以上的 16 个核全部是 active worker。

### 4.3 `SyncAll<true>()` 的硬约束

所有 16 个 block 必须执行完全相同数量和顺序的 `SyncAll<true>()`：

- 不允许在必经 barrier 前基于 `valid_`、`succeeded`、peer route 或 core 角色提前返回；
- 不允许 `for (... && succeeded)` 使失败核少执行后续集合点；
- route-specific 函数可以失败，但只能记录 local failure，然后进入共同汇合点；
- 所有核读取统一状态后，才允许一起继续或一起退出；
- 任何新增条件分支都必须通过 source guard 检查 barrier 拓扑。

CANN 9.1 仓库内已有 `AscendC::SyncAll<true>()` 用例，但本机没有目标 toolkit 证明该 API
在当前 kernel 编译配置中的最终可用性。CANN 9.1 A5/3510 target compile 是 R1 的硬门槛，
Host 测试不能替代。

## 5. Server-Pair 调度模型

### 5.1 基本维度

对 `R in {32, 64, 128}`：

```text
R = rankSize
serverCount = R / 8 = 4T
T = R / 32
roundCount = R / 16 = 2T

phase = round / T
phaseStep = round % T
round = phase * T + phaseStep
```

| Rank 数 | `T` | `roundCount` | phase0 | phase1 |
| ---: | ---: | ---: | --- | --- |
| 32P | 1 | 2 | round 0 | round 1 |
| 64P | 2 | 4 | round 0-1 | round 2-3 |
| 128P | 4 | 8 | round 0-3 | round 4-7 |

旧主循环的一维 round 数保持不变，因此现有最多 8 round 的 Profile 容量仍可复用。

### 5.2 Target-half 模式

R3 中间模式：

```text
sourceHalf = sourceRank / (R / 2)
phase0 targetHalf = sourceHalf
phase1 targetHalf = 1 - sourceHalf
```

R5 最终 parity 模式：

```text
sourceParity = sourceRank % 2
phase0 targetHalf = 1 - sourceParity
phase1 targetHalf = sourceParity
```

R5 只允许改变 target-half 选择及其 receive inverse；不能同时改变 pair、core、route、
Done 或 barrier。

### 5.3 Server pair

```text
sourceServer = sourceRank / 8
pairIndex = (sourceServer % T + phaseStep) % T
halfBase = targetHalf * 2T

firstServer = halfBase + ((2 * pairIndex + 1) % (2T))
secondServer = halfBase + ((2 * pairIndex + 2) % (2T))
```

Core 到 peer：

```text
core 0..7:
    targetServer = firstServer
    targetLocalRank = core

core 8..15:
    targetServer = secondServer
    targetLocalRank = core - 8

peer = targetServer * 8 + targetLocalRank
```

每个 source rank 在一个 round 中覆盖两个完整 server；`2T` 个 round 与 16 个 core 的
笛卡尔覆盖仍为 `R` 个 destination。

### 5.4 128P 锚点

128P 时 `T=4`。source server0 的 pair 顺序必须为：

```text
前半区: (1,2) -> (3,4) -> (5,6) -> (7,0)
后半区: (9,10) -> (11,12) -> (13,14) -> (15,8)
```

最终 parity 模式：

```text
rank0 偶卡: phase0 后半区，phase1 前半区
rank1 奇卡: phase0 前半区，phase1 后半区
```

## 6. 接收逆映射和统一 helper

给定 `(destinationRank, sourceRank, R)`：

```text
sourceServer = sourceRank / 8
destinationServer = destinationRank / 8
destinationLocal = destinationRank % 8
targetHalf = destinationServer / (2T)
relativeServer = destinationServer - targetHalf * 2T

pairIndex = (((relativeServer + 2T - 1) % (2T)) / 2)
phaseStep = (pairIndex + T - (sourceServer % T)) % T
```

same/cross 模式：

```text
sourceHalf = sourceRank / (R/2)
phase = targetHalf == sourceHalf ? 0 : 1
```

parity 模式：

```text
sourceParity = sourceRank % 2
phase = targetHalf == sourceParity ? 1 : 0
```

发送核和接收 round：

```text
firstServer = targetHalf * 2T + ((2 * pairIndex + 1) % (2T))

senderCore = destinationServer == firstServer
    ? destinationLocal
    : 8 + destinationLocal

receiveRound = phase * T + phaseStep
```

`src/moonep/common/moonep_combine_schedule.h` 提供 Host/AICore 共用的 C++14 helper，
至少包含：

```text
MoonEpCombineV2ServerPairRankSize
MoonEpCombineV2ServerPairPhaseStepCount
MoonEpCombineV2ServerPairRoundCount
MoonEpCombineV2ServerPairPhase
MoonEpCombineV2ServerPairPhaseStep
MoonEpCombineV2ServerPairTargetHalf
MoonEpCombineV2ServerPairIndex
MoonEpCombineV2ServerPairPeer
MoonEpCombineV2ServerPairReceive
MoonEpCombineV2ServerPairSenderCore
```

建议统一返回：

```cpp
struct MoonEpCombineV2ScheduleCoordinate {
    uint32_t phase;
    uint32_t phaseStep;
    uint32_t round;
    uint32_t core;
};
```

发送、Done wait、full-sync、Grant、Profile 和 trace 只能消费这些 helper，禁止复制公式。
非法输入返回 invalid coordinate/peer，不能发生 `% 0`。

## 7. 数据路径和 Done 合同

只替换 `peer`，继续按现有条件路由：

```text
peer == rank
    -> SendSelfStep

peer != rank && sameServer(rank, peer)
    -> SendFullmeshStep

peer != rank && !sameServer(rank, peer)
    -> SendRemoteStep（shared 6-port/2-port）
```

Done token 保持：

```text
DoneToken = MoonEpCombineV2Token(magic, round)
```

Done lane 数保持：

```text
Self                    -> 0
同 server Fullmesh      -> lane 0，共 1 个
跨 server shared-QP     -> lane 0/1，共 2 个
```

`WaitInboundDone()` 只把旧 `ReceiveStep()` 换成 Server-Pair receive coordinate 的
`round`。`DoneIndex(epoch, source, lane)`、source 分工、scratch 和 Reduce 语义不变。

每个 route 的 CQ 必须按实际 submission 等待：

| Route | 数据阶段 CQ |
| --- | --- |
| Self | 无数据 CQ；若后续发布控制 WQE，则等待控制 CQ |
| Fullmesh | 仅等待对应 Fullmesh CQ |
| shared | 等待两个实际提交 lane 的 CQ |

任何 route 在进入 Grant 发布或全局 barrier 前，都必须完成本 step 的 submission 回收，且
保持用户基线新增的 CQ/SQ 终态检查。

## 8. 16 核失败汇合协议

`SyncAll<true>()` 只提供 rendezvous，不会自动把 16 个 `localReady` 做 AND 归约。因此用
原 `fullSyncBarrier` workspace 承载 magic-tagged collective status，不新增 Kernel 参数。

### 8.1 状态布局

保留 `fullSyncBarrierOffset` 参数位置，将其内部语义改成 collective status 区。建议每个
epoch 预留 16 个 64 Byte status slot，按 `stageId % 16` 严格串行复用。现有：

```text
2 epochs * 16 slots * 64 B = 2048 B
```

已经足够。逻辑 stage 数可以超过 16；复用安全性来自所有 16 核完成旧 stage 的 end
`SyncAll` 后，core0 才能初始化同一物理槽的新 stage。每个 slot 至少包含：

```text
magic, marker, stageId, status, firstFailureCore, guard
```

详细错误继续写现有 per-core `MoonEpCombineV2FailureRecord`，collective status 只负责
“本 stage 是否有任一核失败”。

### 8.2 每次 invocation

1. core0 初始化 invocation header，并执行必要的 cache clean；
2. 全 16 核执行初始 `SyncAll<true>()`，确认 workspace 和 magic 已就绪；
3. 每个逻辑 stage 开始时，core0 初始化
   `status[(epoch * 16) + (stageId % 16)]` 的 magic/stageId/success；
4. 全 16 核执行 stage-begin `SyncAll<true>()`，再校验该 slot 的 magic/stageId；
5. active worker 执行本 stage 工作，非 active worker 仅保持控制流一致；
6. 失败核用原子操作把当前 status 从 success 改为 failure；
7. 全 16 核执行 stage-end `SyncAll<true>()`；
8. 每核 invalidate/read 同一个 status，得到一致的继续或退出决策；
9. 需要退出时，所有 16 核经过同一最终集合点后一起返回。

Grant 轮询过程中应周期读取 aggregate failure。一核发现缺失/坏信号并置失败后，其他核
可停止无意义轮询并到达共同 `SyncAll`，不必各自等满 timeout。

### 8.3 必须消除的分歧点

- kernel 入口的 inactive-core 提前返回；
- `Process()` 的 `if (!valid_) return`；
- `for (step ... && succeeded)`；
- full-sync 中失败核直接 return、成功核继续；
- Grant 分片中某核失败后跳过本 step 的 `SyncAll`；
- Reduce 前只有部分核进入或退出。

## 9. Boundary-aware 全局 barrier

### 9.1 执行语义

复用现有 full-sync 的 shared 6-port control QP、UB WQE 构造、MTE3 SQ 发布、`st_dev`
doorbell、terminal ordered completion 和远端 source wait，将一次性的 leading full-sync
泛化为：

```text
RunGlobalBarrier(boundaryId)
```

每次 barrier：

1. 每个 active worker 按 active schedule 枚举远端 peer；
2. 在 UB 完整构造本 boundary 的 signal WQE batch；
3. MTE3 写 SQ 完成后才更新 head/wqeCnt 并 `st_dev` doorbell；
4. 等 terminal CQ 并验证 SQ/CQ drain；
5. 等待本核负责的远端 source signal；
6. 记录 local stage status；
7. 全 16 核执行 `SyncAll<true>()`；
8. 读取 aggregate status 后统一继续或退出。

不再使用旧 `FullSyncActiveCoreBarrier()` 的 GM 轮询实现。

### 9.2 单调 boundary ID

令 `T=R/32`，`round` 为扁平 round：

```text
leading                           = 0
phase0 round phaseStep=s          = 1 + s
phase barrier                     = 1 + T
phase1 round phaseStep=s          = 2 + T + s
```

等价地，对 phase1 的扁平 `round in [T, 2T)`：

```text
phase1 round boundary = 2 + round
```

128P 的实际顺序为：

```text
leading 0
phase0 round 0..3 -> boundary 1..4
phase barrier -> boundary 5
phase1 round 4..7 -> boundary 6..9
```

R1 尚未启用 phase barrier 时保留 boundary `1+T` 的编号空位，使 R4 激活 phase barrier
时其他 boundary ID 不变。

### 9.3 两个 generation slot

```text
generation = executionOrdinal % 2

receiveIndex = (epoch * 2 + generation) * 128 + sourceRank
sourceIndex  = (epoch * 2 + generation) * 16  + sourceCore
```

`boundaryId` 是稳定的协议编号，`executionOrdinal` 是本 invocation 中实际执行的 barrier
序号，两者不能混用。R1-R3 尚未执行 phase barrier，因此：

```text
leading executionOrdinal = 0
round s executionOrdinal = s + 1
```

例如 128P 的 boundary 4 后直接执行 boundary 6，但 execution ordinal 为 4、5，必须使用
不同 generation。R4 加入 phase barrier 后，应把该 barrier 纳入连续 execution ordinal，
并相应顺延 phase1 round 的 ordinal。

signal guard 必须包含完整 `boundaryId`：

```text
guard = HashOrXor(magic, marker, boundaryId, sourceRank, sourceCore, rankSize)
```

接收方同时校验 `magic/marker/boundary/sourceRank/sourceCore/rankSize/guard`。仅比较 generation
会把 boundary `n` 的陈旧信号误认成 `n+2`。

两槽安全依赖严格串行执行顺序：所有 rank 完成 execution ordinal `n+1` 后才可能发布
`n+2`。若实现或测试不能证明该顺序，必须按实际 barrier 数量分配独立槽，不能依赖稳定
boundary 编号或时间猜测。

### 9.4 Workspace

```text
fullSyncReceive = 2 epochs * 2 generations * 128 * 64 B = 32768 B
fullSyncSource  = 2 epochs * 2 generations * 16  * 64 B = 4096 B
collectiveStatus（复用原 barrier 区）                = 2048 B
```

公共 API 和 Kernel 参数顺序不变；Host layout 只扩大 receive/source 区并重新计算后续
offset 和最终 2 MiB 对齐。

## 10. R2 停用 Legacy Ring Grant

旧 Grant 的 successor、每核双 lane wait 和 slot 语义都依赖 Ring，不能只替换 peer 后继续
使用。32P+ 的 R2 必须：

- shared `SubmitPair()` 的 final batch 不再追加 Legacy Grant，只追加 Done；
- Self 和 Fullmesh epilogue 不再调用 `SubmitSelfGrant()`；
- 不调用 `WaitStepGrant()`；
- final batch 的实际 control count 从每 lane 2 降为 1；
- `kControlWqesPerLane` 可以保留容量上界，但实际 `count[lane]` 只能包含已构造 WQE；
- 不能把删除 Grant 后的未初始化 UB slot 发布到 SQ；
- 每 round 数据 CQ drain 后进入 `RunGlobalBarrier(roundBoundary)`；
- generation 按实际 barrier 执行序号交替，不能因预留 phase boundary 编号而连续复用；
- 2P-16P 保留原 Legacy Grant 分支。

R2 是新 peer 映射前的 barrier-only 完整 All2All 基线。

## 11. 最终 64-Grant Server 协议

R6-R8 只在 R5 parity Server-Pair 已通过后启用。

### 11.1 Quarter 和 Grant successor

```text
T = R / 32
sourceServer = sourceRank / 8
quarter = sourceServer / T                    // 0..3
serverInQuarter = sourceServer % T            // 0..T-1

grantTargetServer = quarter * T +
    ((serverInQuarter + T - 1) % T)
```

原因是同一 phase 内，当前 server `i` 在 `phaseStep=s` 使用的 pairIndex 为：

```text
(i + s) % T
```

quarter 内前一个 server `i-1` 在下一 step 使用：

```text
((i - 1) + (s + 1)) % T == (i + s) % T
```

因此当前 server 完成 pair 后，向 `i-1` 发布 Grant，恰好授权下一 phaseStep 使用相同
目标 server pair 的 source server。final transition 仍执行完整 Grant closure；它不触发
额外数据 round。phase 切换仍由独立 phase barrier 保护。

### 11.2 4 source cards x 16 cores

在 parity 模式下，一个 source server 内同奇偶的四张卡构成 Grant publisher group：

```text
sourceParity = sourceRank % 2
sourceCardLane = (sourceRank % 8) / 2       // 0..3
```

每张 publisher 卡的 16 个核都向 `grantTargetServer` 中同 parity 的四张 target 卡发布
一个 signal：

```text
for targetCardLane in [0, 4):
    grantTargetRank = grantTargetServer * 8 +
        2 * targetCardLane + sourceParity
```

因此每张 target 卡每个 transition 收到：

```text
4 source cards * 16 source cores = 64 Grant signals
```

T=1 的 32P 情况下 `grantTargetServer == sourceServer`。self target 使用本地发布，同 server
其他三张卡仍走已验证的 shared control QP；不得用 UDMA loopback 等待 self completion。

### 11.3 64 槽编号与 1/16 分片

```text
grantOrdinal = sourceCardLane * 16 + sourceCore   // [0, 64)
```

target 卡上的本地 core `c` 只检查：

```text
c, 16+c, 32+c, 48+c
```

建议代码形态：

```cpp
for (uint32_t sourceCardLane = 0U;
    sourceCardLane < 4U; ++sourceCardLane) {
    WaitServerGrant(phase, transitionStep, sourceCardLane, core_);
}
ReportCollectiveStatus(stageId, localReady);
AscendC::SyncAll<true>();
localReady = ReadCollectiveStatus(stageId);
```

必须由 unit oracle 证明：

- 每核恰好读取 4 个槽；
- 任意两个核的槽集合不相交；
- 16 核并集恰好为 `[0,64)`；
- 每个期望 signal 的 `sourceRank/sourceCore` 与 ordinal 唯一对应；
- 不存在全核重复扫描 64 槽的旧实现。

### 11.4 Signal 和 stale rejection

新 Grant 使用独立的 64 Byte、magic-tagged signal，字段至少包含：

```text
magic, marker, sourceRank, sourceCore, rankSize, flatRound, guard
```

```text
flatRound = phase * T + transitionStep
```

接收方校验全部字段。`guard` 至少覆盖上述字段，不能只检查 magic 或物理 slot。超时诊断
记录 `phase/transitionStep/sourceCardLane/sourceCore/expected/observed`。

### 11.5 Workspace 复用

固定最大索引：

```text
signalIndex = (epoch * 8 + flatRound) * 64 + grantOrdinal
```

最大 1024 个 signal slot。receive/source 分区各使用 64 Byte slot：

```text
serverGrantReceive = 2 * 8 * 64 * 64 B = 65536 B
serverGrantSource  = 2 * 8 * 64 * 64 B = 65536 B
total                                      131072 B
```

现有 Legacy Grant 区为 262144 B。由于 2P-16P 和 32P+ 新协议在一个 invocation 中互斥，
可以把 `grantOffset` 区设计成二者的 union，保持 `grantBytes=262144` 和 Kernel 参数不变。
Host layout 和 unit test 必须用 `max(legacyBytes, serverGrantBytes)` 计算并加静态边界检查，
不能依赖手算地址碰巧落在旧区内。

### 11.6 发布 WQE 和 CQ

每个 publisher core：

1. 在本地 serverGrantSource slot 构造一次 64 Byte signal 并 clean；
2. 对四个 target card 逐一处理；self target 直接写 receive slot；
3. 其他 target 使用本核 6-port shared control QP 构造 WRITE WQE；
4. 所有 WQE 在 UB 完整构造并通过 MTE3 发布；
5. 仅最后一个远端 WQE请求 ordered completion；
6. MTE3 完成后才用 `st_dev` 更新 SQ head、wqeCnt 和 doorbell；
7. 等 terminal CQ 并验证 `tail == submittedHead && head == tail`；
8. 完成后才进入 shadow/read 或 admission wait。

Grant publication 必须发生在本 step 数据 submission 的 CQ 完成之后。不能把新 Grant 插到
payload Done 前，也不能依赖不同 QP 之间的隐式顺序。

### 11.7 R6、R7、R8 顺序差异

R6 shadow：

```text
data CQ drain
-> publish Server-Grant and wait Grant CQ
-> round global barrier（authoritative）
-> 每核检查 4 个 Grant
-> SyncAll 汇合并校验，无缺失才通过本版本
```

由于 round barrier 先完成，Grant 不决定下一 round 的开始时间，只验证发布、地址、token
和 64 槽覆盖。

R7 admission：

```text
data CQ drain
-> publish Server-Grant and wait Grant CQ
-> 每核等待 4 个 Grant
-> SyncAll 汇合
-> round global barrier（冗余）
-> next round
```

Grant 已经成为 admission；round barrier 继续存在以发现两种门控的不一致。

R8 final：

```text
data CQ drain
-> publish Server-Grant and wait Grant CQ
-> 每核等待 4 个 Grant
-> SyncAll 汇合
-> next round
```

phase0 结束后仍执行 phase barrier；leading barrier 仍在 invocation 开始时执行。删除的只是
每个 round 的全局 barrier。

## 12. Profile、trace 和诊断

Profile 固定 8-round 容量可保留，但数值语义需要升级版本并统一：

```text
STEP_SEND_END  = 数据发送及对应 CQ 完成
STEP_READY_END = Grant/round barrier admission 完成
```

R6-R8 应增加或复用诊断字段表示：

```text
rank, core, round, phase, phaseStep,
sourceServer, sourceParity, targetHalf,
pairIndex, firstServer, secondServer, peer,
route, logicalQp,
grantTargetServer, sourceCardLane,
grantShardCount, grantWaitEnd,
boundaryId, collectiveStatus
```

失败记录的 `step` 继续保存扁平 round，避免扩大公共错误 ABI；日志和 trace 再派生
`phase/phaseStep`。旧 `clos_grant_*` 字段必须重命名或按 profile version 分支解释，避免把
Server-Grant 误标为 Legacy CLOS Grant。

## 13. 逐文件改动清单

| 文件 | 计划改动 |
| --- | --- |
| `src/moonep/common/moonep_combine_schedule.h` | 新 mode、Server-Pair 正反映射、boundary、server-quarter Grant、64 槽分片 helper 和 signal/index 常量；保留 Legacy helper |
| `src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h` | 无分歧 16 核控制流、repeatable global barrier、route CQ 分流、Legacy Grant 隔离、Server-Grant publish/wait、`SyncAll` 汇合 |
| `src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.cpp` | 删除 inactive block 提前返回，确保 16 block 共同进入 collective skeleton |
| `src/moonep/combine_v2/host/combine_v2_layout.h` | 扩展 full-sync generation 字段或内部 alias，明确 Grant union 和 collective status |
| `src/moonep/combine_v2/host/combine_v2_layout.cpp` | checked arithmetic 计算两 generation 和 Grant union，重算 downstream offset |
| `src/moonep/combine_v2/host/combine_v2_host.cpp` | `aivCoreNum == 16`，设备至少 16 vector cores |
| `src/moonep/combine_v2/host/combine_v2_launch.cpp` | launch blockDim 固定为 `kMoonEpCombineV2CoreCount` |
| `src/include/tilexr_moonep_combine_v2.h` | 公共注释由 `>=16` 改为 `==16`，签名不变 |
| `src/moonep/combine_v2/common/combine_v2_profile.h` | profile version、Server-Pair/Grant/barrier 语义和容量检查 |
| `tests/moonep_combine_v2/unit/test_combine_v2_schedule.cpp` | 映射、inverse、boundary、quarter successor、64 Grant 和分片穷举 oracle |
| `tests/moonep_combine_v2/unit/test_combine_v2_layout.cpp` | generation、Grant union、status、offset、overflow、2 MiB 对齐 |
| `tests/moonep_combine_v2/unit/test_combine_v2_host.cpp` | 15/16/17 核输入和设备 core 数边界 |
| `tests/moonep_combine_v2/unit/test_combine_v2_launch.cpp` | 捕获 blockDim 恒为 16，Kernel args ABI 不变 |
| `tests/moonep_combine_v2/unit/test_combine_v2_source_guard.cpp` | WQE/MTE3/doorbell、无提前 return、SyncAll 对称、无 64 槽全扫描、CQ 终态 guard |
| `tests/moonep_combine_v2/demo/tilexr_moonep_combine_v2_hardware_probe.cpp` | 输出 schedule/Grant/barrier 诊断，保持原调用合同 |
| `tools/moonep/combine_v2_trace.py` | 解码 phase/phaseStep、Server-Pair、Server-Grant 和 boundary |
| `tests/moonep/python/test_combine_v2_trace.py` | 新旧 profile version、16 核、Grant/phase 事件和错误输入 |
| `tools/moonep/cab0_2_128p/compile.sh` | 保持 CANN 9.1 target build，输出 build provenance |
| `tools/moonep/cab0_2_128p/sync_runtime.sh` | 分发同一 install 到全部 worker 并校验 artifact hash |
| `tools/moonep/cab0_2_128p/run.sh` | NPU 占用门禁、stage/case 选择、逐 rank correctness 和日志归档 |
| `docs/moonep/COMBINE_V2_MULTIHOST_PERF.md` 或现有最相关文档 | 实施后记录可复用的 Grant 分片、SyncAll 分歧风险和验证边界 |

## 14. 详细实施计划单

每项任务都必须在进入下一项前满足 acceptance。计划不要求每项单独 commit，但要求能从
日志和 diff 明确识别每个版本。

### 任务 0：冻结 R0 基线

**目标：** 建立当前 Ring + Fullmesh + Legacy Grant 的可复现正确性和日志基线。

**前置：** 记录 HEAD、kernel hash、CANN/driver、hostfile、设备映射和当前二进制 hash。

**修改范围：** 不改生产代码；只允许补充不改变行为的测试/脚本日志字段。

**验证：** Host unit；目标编译；32P/64P/128P small、production 和 80 iterations；hidden
和 route-weight correctness；每 rank correctness。

**Acceptance：** R0 正确或有明确已知基线问题。若 R0 本身失败，不进入 R1。

**产物：** `R0` 日志目录、命令记录、每 rank 结果、CQ/SQ/Done/Grant 诊断和 artifact hash。

### 任务 1：固定 16 核 Host/Launch 合同

**目标：** 为后续硬件全核同步建立不可变 blockDim。

**前置：** R0 基线完成。

**修改范围：** public header 注释、Host validation、launch、host/launch/public ABI tests。

**约束：** API 参数和 Kernel args ABI 不变；只收紧合法值；设备仍需报告至少 16 核。

**验证：** 15 和 17 被拒绝；16 成功；captured Runtime blockDim 恒为 16；Python runtime
调用不变；CANN target compile。

**Acceptance：** 所有合法 launch 都是 16 block，任何调用方无法启动其他 blockDim。

**产物：** 固定 launch 合同和回归测试。

### 任务 2：无分歧 `SyncAll` 控制流

**目标：** 所有 16 block 在成功、局部失败和非 active worker 情况下执行一致的集合点。

**前置：** 任务 1。

**修改范围：** kernel wrapper、`Init/Process/PublishFailureAndConverge`、collective status
layout 和 source guards。

**约束：** 暂不改 active Ring peer 和 Legacy Grant；不把 `SyncAll` 当作状态归约；保留
per-core failure record。

**验证：** source guard 枚举所有 barrier；故障注入覆盖一个 worker 失败、inactive worker、
uniform invalid config；目标编译；小 shape 连续 invocation。

**Acceptance：** 不存在 barrier 前分歧 return，所有错误路径有界、统一退出。

**产物：** 可供所有后续阶段复用的 collective skeleton。

### 任务 3：Schedule helper 和 Host oracle

**目标：** 建立 same/cross、parity、receive inverse、sender core、boundary 和 Grant successor
的唯一公式来源。

**前置：** 可与任务 1-2 的实现准备并行，但 active kernel mode 不切换。

**修改范围：** schedule header 和 schedule unit test。

**约束：** C++14 Host/AICore 共用；保留旧 Ring；非法输入有界返回；不使用 Host-only STL
容器进入 device helper。

**验证：** 对 32P/64P/128P 穷举 source、destination、round、core、mode、Grant publisher
和 waiter。

**Acceptance：** 第 15 节全部 oracle 不变量通过，失败时打印确定性最小反例。

**产物：** `Peer/Receive/SenderCore/Boundary/GrantTarget/GrantSource/Ordinal` helper。

### 任务 4：R1 repeatable global barrier

**目标：** 在 active Ring 不变时，把 leading full-sync 泛化为 leading/round barrier，并用
`SyncAll<true>()` 替换本卡 GM polling barrier。

**前置：** 任务 1-3。

**修改范围：** full-sync signal/index、Host layout、kernel full-sync 方法、profile、layout
和 source-guard tests。

**约束：** 两 generation；boundary guard；shared 6-port QP；UB WQE；MTE3；`st_dev`；
R1 不启用 phase barrier。

**验证：** 旧 Ring + Legacy Grant + leading/round barrier 完整硬件阶梯；连续 magic 和
generation 复用；CQ/SQ drain。

**Acceptance：** 所有 boundary 无 stale match、timeout 或 outstanding；数据与 R0 一致。

**产物：** R1 binary 和 boundary 证据。

### 任务 5：R2 停用 32P+ Legacy Grant

**目标：** 建立旧 Ring + barrier-only 的完整 All2All 基线。

**前置：** R1 硬件通过。

**修改范围：** `SubmitPair()` control count、Self/Fullmesh epilogue、Process admission 和
source guards。

**约束：** 2P-16P Legacy 不变；Done ordering 不变；只等待实际 submission 的 CQ。

**验证：** 确认 32P+ Legacy Grant receive/source 无读写；final batch 不发布未初始化 slot；
检查 128P boundary 4 -> 6 仍切换 generation；32/64/128P 硬件阶梯。

**Acceptance：** 强 round barrier 下完整正确且无死锁；失败则不切 peer。

**产物：** R2 barrier-only Ring 基线。

### 任务 6：R3 same/cross Server-Pair

**目标：** 只切换 peer、receive round 和 full-sync sender-core，验证 Server-Pair 数据面。

**前置：** R2 和 schedule oracle 通过。

**修改范围：** kernel Init/Process/WaitInboundDone/full-sync、profile/probe/trace。

**约束：** target-half 用 source-half；round barrier authoritative；不启用 phase barrier；
不接新 Grant。

**验证：** 每目标每 round 恰好 2 source servers x 8 cards；Self、Fullmesh、shared route 都
被 trace 覆盖；Done round 与 inverse 一致。

**Acceptance：** 32/64/128P hidden 和 route-weight 完整正确，CQ/SQ/Done 无残留。

**产物：** R3 可独立运行的 same/cross binary。

### 任务 7：R4 独立 phase barrier

**目标：** 在不改 peer 的情况下验证 boundary `1+T`。

**前置：** R3。

**修改范围：** Process phase0/phase1 边界和 profile/trace。

**约束：** phase0 最后 round barrier 与 phase barrier 都保留；二者故意冗余。

**验证：** phase barrier generation/guard、CQ drain、连续 invocation、完整硬件阶梯。

**Acceptance：** phase barrier 可单独定位且不误认相邻 round signal。

**产物：** R4 phase boundary 证据。

### 任务 8：R5 parity Server-Pair

**目标：** 只切换 target-half 和 receive inverse，得到最终发送顺序。

**前置：** R4。

**修改范围：** active mode 常量/helper 调用、oracle 和 trace 预期。

**约束：** pair、core、route、Done、round/phase barrier 不变。

**验证：** 每目标每 round 4 source servers x 4 parity cards；same-half source servers 恰好
2；两个 phase 合并覆盖全部 source。

**Acceptance：** 32/64/128P 完整正确，trace 与 Host oracle 逐项一致。

**产物：** R5 最终 parity mapping 和稳定 Grant helper 接口。

### 任务 9：R6 64-Grant shadow

**目标：** 验证 server-quarter publisher、64 槽地址、signal、CQ 和每核四槽 wait。

**前置：** R5。

**修改范围：** Grant signal/index、layout union、publish WQE、sharded wait、failure/profile、
schedule/layout/source-guard tests。

**约束：** round barrier 先完成并保持 authoritative；每核不得扫描非本 shard 槽；Grant
WQE 在数据 CQ 后发布；Grant CQ 必须 drain。

**验证：** 故障注入分别缺失 ordinal 0、15、16、63；确认唯一责任 core 报错，全卡通过
status + `SyncAll` 一致失败；32P T=1 self-server case；完整硬件阶梯。

**Acceptance：** 每 transition 每 target 卡正好 64 个有效 signal，无重复、遗漏、stale
match 或全核 64 槽重复扫描。

**产物：** R6 shadow 证据和 Grant coverage dump。

### 任务 10：R7 Grant admission

**目标：** 证明新 Grant 在 round barrier 前能够独立阻止下一 round。

**前置：** R6。

**修改范围：** 仅调整 Process 内 Grant wait 与 round barrier 的顺序和 profile 标签。

**约束：** 不改 signal、地址、publisher 或 peer；round barrier 保留。

**验证：** 延迟/缺失一个 Grant shard，确认下一 round 不发送；正常 case 中 Grant 和 barrier
均通过；完整硬件阶梯和 80 iterations。

**Acceptance：** Grant gate 与 round barrier 结果一致，无下一 round 提前发布。

**产物：** R7 双门控证据。

### 任务 11：R8 删除 round barrier

**目标：** 形成最终 `Grant + phase barrier` 控制流。

**前置：** R7。

**修改范围：** 只删除每 round `RunGlobalBarrier` 调用和对应 active profile event；保留实现
供调试或 compile-time 对照时需有明确开关。

**约束：** leading barrier、phase barrier、Grant、Done、CQ 和 failure convergence 不变。

**验证：** 先 32P small，再完整阶梯；重复 epoch0->epoch1->epoch0；80 iterations；对比
R7 correctness 和性能分段。

**Acceptance：** 无 round barrier 时仍无错数、挂死、stale Grant、SQ/CQ 残留或下一轮
提前发送。

**产物：** R8 最终实现证据。

### 任务 12：诊断、脚本和维护文档收尾

**目标：** 让远端验证和问题定位可重复，并记录可复用经验。

**前置：** 随每阶段持续更新，R8 后统一检查。

**修改范围：** profile、hardware probe、trace parser/tests、`cab0_2_128p` 三个脚本和最
相关 MoonEP 文档。

**约束：** 不改变 `test_npu_e2e.py` 公共调用；不创建散落临时说明；日志包含 commit、
artifact hash、CANN/driver、host/device map 和 case 参数。

**验证：** Python parser tests；脚本 shell syntax；一次从 source flush 到 128P 结果的
完整闭环。

**Acceptance：** 新人只使用本文件和脚本即可重现 build/deploy/run，并能从日志区分
mapping、Grant、barrier、CQ 和 correctness 失败。

**产物：** 可维护脚本、trace 和经验文档。

## 15. Host oracle 必须证明的不变量

对 32P、64P、128P 和 same/cross、parity 两种模式穷举：

1. 每个 source 的 `roundCount * 16 == R` 个 peer 覆盖全部 destination，且无重复；
2. 每个 destination 在全部 source 中被覆盖一次；
3. sender mapping 与 receive inverse 的 round/core 完全一致；
4. 每个 round 的 16 core 恰好覆盖两个完整 server；
5. Self 每个 source 恰好一次；
6. same/cross 每目标 round 恰好 2 source servers、16 source ranks；
7. parity 每目标 round 恰好 4 source servers、每 server 4 同 parity cards；
8. parity 的 same-half source servers 恰好 2；
9. 128P server0 pair 锚点成立；
10. Fullmesh Done lane 为 1，shared Done lane 为 2，Self 为 0；
11. full-sync sender-core 对每个非 Self source/destination 唯一；
12. boundary ID 在真实执行顺序中严格递增；
13. boundary generation 为 `boundary & 1`，guard 能拒绝 `n-2` 陈旧信号；
14. Grant target 与 next-step pair invariant 对每个 quarter/server/phaseStep 成立；
15. 每个 target card 的 publisher 集合恰好是 predecessor server 的 4 张同 parity cards；
16. 每个 publisher card 的 16 核各发布一次，并覆盖四张 target card；
17. 每个 target card 每 transition 恰好 64 个唯一 Grant；
18. core `c` 的 Grant ordinal 恰好为 `{c,16+c,32+c,48+c}`；
19. 16 个 shard 两两不交且并集为 `[0,64)`；
20. Grant receive/source index 在两个 epoch 和八个 flat round 中无重叠、无越界。

Host oracle 只能证明公式和 layout，不能证明 device cache、UDMA ordering、CQ 或
`SyncAll<true>()` 的硬件行为。

## 16. 分层验证策略

### 16.1 本地 Host/unit

在 Git Bash 中执行：

```bash
cmake -S tests/moonep_combine_v2 -B build-combine-v2-host
cmake --build build-combine-v2-host -j"$(nproc)"
ctest --test-dir build-combine-v2-host --output-on-failure
python tests/moonep/python/test_combine_v2_trace.py
```

证据范围：schedule、inverse、Grant shard、layout、Host/launch、public ABI、source guard 和
trace parser。不能据此宣称 target compile 或硬件正确。

### 16.2 CANN 9.1 target compile

必须检查：

- `SyncAll<true>()` 在目标 AICore 编译模型中可用；
- 所有 16 block 的 barrier 代码生成成功；
- Server-Grant signal/WQE builder 通过资源限制；
- WQE 完全在 UB 构造；
- MTE3 完成后才 doorbell；
- kernel binary、Host signature 和 21 个 64-bit 参数一致。

### 16.3 硬件阶梯

每个 R1-R8 版本按以下顺序串行：

```text
32P small
-> 32P production
-> 64P small
-> 64P production
-> 128P small
-> 128P production
-> 连续多轮
-> 80 iterations stress
```

每个规模至少覆盖：

- BF16 hidden，不启用 Reduce；
- BF16 hidden，启用 Reduce；
- FP32 route-weight；
- Self、Fullmesh、shared 三种 route；
- epoch0 -> epoch1 -> epoch0；
- 每 rank correctness；
- CQ/SQ drain、Done round、Grant shard、boundary guard 和 stale rejection。

正确性以每 rank 输出为准，不以 controller exit code 单独判断。

### 16.4 通信环境基线

在任何 device-path 调试前，必须在相同设备子集、相同 topology 和匹配 accelerator mode
运行 installed official HCCL Test。AIV-only peer-memory case 使用：

```text
all_reduce_test -a aiv_only
```

按每 rank correctness 判断 HCCL 健康，不只看退出码。匹配 HCCL baseline 失败时，归类为
环境/平台不健康，保留日志并停止 operator 修改，不为未证明健康的环境编写绕过代码。

## 17. 四条远端闭环命令

以下四条命令是每个可部署阶段的标准闭环，顺序固定。源码同步会话必须是已经配置好的
Mutagen `one-way-safe`，不得改成双向同步。

### 命令 1：把本地代码同步到服务器

```bash
mutagen sync flush tilexr-combine-v2-cab0-2-fast
```

作用：强制刷新现有 Mutagen session，把本地工作区的最新源码单向同步到主服务器目录：

```text
/home/h00580772/tilexr_combine_v2_cab0_2_fast
```

通过条件：flush 成功，主服务器目标源码的 commit/diff 或目标文件 hash 与本地本阶段一致。
它只同步源码，不执行编译，也不把服务器产生的文件反向覆盖本地。

### 命令 2：在主服务器编译

```bash
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/compile.sh'
```

作用：在 `141.61.53.150` 上加载 `/home/pkg/b131/cann-9.1.0`，调用
`tools/moonep/build_combine_v2_perf.sh`，使用 16 个 build job 构建并安装到：

```text
/home/h00580772/tilexr_combine_v2_cab0_2_fast/install
```

主要日志：

```text
/home/h00580772/tilexr_combine_v2_cab0_2_fast/logs/build_latest.log
```

通过条件：Host 库、embedded AICore binary 和 hardware probe 均构建/安装成功；日志中的
source commit、CANN 和 binary hash 属于当前阶段。编译成功不代表硬件正确。

### 命令 3：把编译结果同步到 worker 服务器

```bash
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/sync_runtime.sh'
```

作用：从主服务器把 `install` 运行时产物分发到 `hostfile` 中 16 台服务器、每台 8 卡，
确保 128 个 rank 使用同一版本的 Host 库、kernel binary 和测试程序。

主要日志：

```text
/home/h00580772/tilexr_combine_v2_cab0_2_fast/logs/sync_runtime_latest.log
```

通过条件：16 台服务器分发全部成功，关键 artifact hash 一致。此命令同步的是编译结果，
不是把服务器结果同步回本地。

### 命令 4：在服务器集群执行测试

```bash
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/run.sh'
```

作用：由主服务器启动多机 Combine V2 验证。当前脚本默认配置为：

```text
128P, bs=8192, experts=256, hidden=3584,
warmup=20, iterations=80, no profile, timeout=1200s
```

脚本最后输出实际 `LOG_FILE`。通过条件不是 SSH 返回 0，而是：

- 128 个 rank 都有结果；
- 每条 `COMBINE_V2_RANK_PERF` 均为 `correctness=passed`；
- 汇总 `COMBINE_V2_PERF` 为 `correctness=passed`；
- 无 CQ、Done、Grant、barrier timeout；
- 无 SQ/CQ terminal consistency 错误；
- 日志中的 rank/device map、binary hash 和 stage 与预期一致。

`run.sh` 的默认命令覆盖 128P production stress。32P/64P 和 small case 由同一脚本体系通过
受控 case 参数或对应 hostfile 子集执行，且仍遵循“同步源码 -> 编译 -> 分发产物 -> 测试”
四步顺序。

## 18. NPU 占用与远端停止规则

`run.sh` 或其 preflight helper 必须执行以下策略：

1. 检测到其他 `tilexr_*` 任务时允许共用 NPU，不等待；
2. 检测到 Python 进程或其他通信测试进程时，不启动本轮；
3. 每 15 秒重新检测；
4. 120 秒内仍没有兼容空闲条件，则当前测试任务停止并保留状态；
5. 不 kill 不属于当前 stage/job-id 的远端进程；
6. 等待后续唤起再从未完成的版本继续，不越过当前 acceptance gate。

## 19. 每阶段证据清单

每个 R0-R8 目录至少保留：

```text
stage.txt
git_head.txt
git_status.txt
source_hashes.txt
cann_version.txt
driver_version.txt
hostfile
rank_map.txt
artifact_hashes.txt
hccl_baseline.log
build.log
sync_runtime.log
run_controller.log
rank_*.log
correctness_summary.txt
profile_or_trace（该阶段要求时）
```

阶段结论必须写清：

- 哪些命令实际执行；
- 哪些规模、shape、dtype、reduce 路径通过；
- 哪个硬件/拓扑被验证；
- 哪些边界未测；
- 是否允许进入下一阶段。

## 20. 停止条件

出现以下任一情况，停止当前版本，不进入下一版本：

- Host oracle 出现 peer 重复、遗漏或 inverse 不一致；
- 固定 launch 不是 16 block；
- 任一控制流路径可能少执行一次 `SyncAll<true>()`；
- R1 repeatable barrier 在旧 Ring 下超时或 stale match；
- R2 在强 round barrier 下仍错数或死锁；
- Fullmesh/shared route CQ/SQ 未 drain；
- Done expected round 与实际 token 不一致；
- R3 fan-in 不是 `2 servers * 8 cards`；
- R5 fan-in 不是 `4 servers * 4 parity cards`；
- R6 每 target card 不是 64 个唯一 Grant；
- 某 core 检查的 Grant 数不等于 4，或 shard 有交叉/空洞；
- R7 Grant 未完成时下一 round 已开始；
- R8 删除 round barrier 后出现错数、hang 或 outstanding；
- 需要修改公共 API、scratch/Reduce 语义或 `test_npu_e2e.py` 调用才能继续；
- 匹配设备/拓扑的 HCCL baseline 失败。

## 21. 主要风险和控制措施

| 风险 | 影响 | 控制 |
| --- | --- | --- |
| 部分 block 提前 return | `SyncAll` 永久挂死 | 固定 16 block、collective skeleton、source guard、故障注入 |
| 把 `SyncAll` 当归约 | 某核 Grant 失败但其他核继续 | magic-tagged aggregate status + atomic failure |
| peer 与 inverse 分开实现 | Done 永久等待或读错 round | 单一 schedule helper + exhaustive oracle |
| Legacy Grant 跟随新 peer | successor 错误、死锁 | R2 先完全隔离 32P+ Legacy Grant |
| boundary 复用仅看 generation | 误认 `n-2` stale signal | guard 包含完整 boundaryId |
| Grant publisher server 方向反了 | 64 个槽全部等待错误 source | next-step pair invariant 和 publisher/waiter 双向 oracle |
| 每核仍扫描 64 槽 | 轮询放大、性能目标失败 | ordinal source guard + 每核计数 profile |
| Grant 在数据 CQ 前发布 | 下一 source 过早复用目标资源 | 数据 CQ drain 后独立发布 Grant |
| Grant WQE 未回收 | 下 round SQ/CQ 账本污染 | terminal ordered completion 和终态检查 |
| 多机 binary 不一致 | barrier/Grant 协议互相等待 | sync_runtime hash 校验 |
| 只看进程退出码 | 漏掉每 rank 错误 | 强制解析 rank correctness 和 failure records |

## 22. 完成标准

只有以下全部满足，R8 才能标记完成：

1. 公共 API 和 Python 调用合同不变，`aivCoreNum` 仅接受 16；
2. Runtime 实际 blockDim 恒为 16；
3. 所有 16 block 的 `SyncAll<true>()` 数量和顺序一致；
4. same/cross 和 parity Server-Pair 的 Host oracle 全部通过；
5. 32P+ 不再访问 Legacy Ring Grant；
6. 每 target card 每 step 收到 64 个唯一 Server-Grant；
7. 每核只检查四个 disjoint Grant ordinal；
8. Grant wait 后通过显式状态和 `SyncAll` 统一继续或退出；
9. R8 只保留 leading、Grant 和 phase barrier，不保留 round barrier；
10. 32P、64P、128P small/production、hidden/route-weight、Reduce on/off 正确；
11. 128P 20 warmup + 80 iterations 每 rank correctness 通过；
12. Self、Fullmesh、shared-QP 均有实际 trace 证据；
13. CQ/SQ、Done、Grant、boundary 和 epoch 复用无 timeout、stale match 或 outstanding；
14. 匹配 HCCL baseline 健康；
15. 实施中产生的可复用经验已更新到现有 MoonEP 维护文档，并与实现一起交付。

Host/unit、target compile 和真实硬件是不同证据层。任何未执行层都必须明确标记为未验证，
不能用较低层结果代替。
