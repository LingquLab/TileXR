# MoonEP Combine V2 128P Group All2All 单 Done/Credit 设计

状态：设计已确认，代码实现完成，待 CANN target compile 与 128P 实机验证

日期：2026-08-17

## 1. 背景

`tilexr_moonep_combine_v2_group_kernel.h` 当前由
`tilexr_moonep_combine_v2_kernel.h` 直接复制而来，仅修改了 include guard 和类名。它仍然
包含原 Combine V2 的 Server-Pair schedule、Legacy Grant、Server-Grant、全 rank
FullSync、phase barrier 和最终统一 Done 等待。这些协议不属于新的 128P Group All2All
路径，继续保留会增加重定义、误调用和状态机混用风险。

本设计只定义新的 128P Group 路径及其清理边界。原 `MoonEpCombineV2` 保持现有行为，
非 128P launch 继续使用原实现。

## 2. 目标

1. 128P 使用固定 8 个 16 卡 Group 和 8-Step Group All2All 映射。
2. 每个接收 core 每 step 只等待一个 source rank 的单个 Done。
3. 接收 core 看到 Done 后，只向下一 step 对应的一个 sender rank/core 发布一个 credit。
4. step0 无前置 credit；step7 不发布也不等待 credit。
5. step 热路径不执行跨 core 强同步，也不执行跨 rank FullSync 或 phase barrier。
6. 先把 Group kernel 清理为可继续开发的最小骨架，再实现新协议。

## 3. 非目标

- 不改变原 `MoonEpCombineV2` 的 schedule、Grant、Done、FullSync、Reduce 或兼容规模。
- 不改变公共 Host API、Python API、Kernel 参数顺序或 workspace 对外 ABI。
- 不在第一版引入 Group 级 Done 聚合、Group barrier 或卡内 step admission。
- 不为非 128P 实现 Group schedule fallback。
- 清理阶段不要求 128P Group 数据面功能完整或硬件正确性通过，但应保持源码结构明确，
  并以能够完成 CANN target compile 为目标。

## 4. Kernel 入口与代码隔离

`tilexr_moonep_combine_v2_kernel.cpp` 同时包含原 kernel header 和 Group kernel header，并从
`CommArgs` 读取 `rankSize`：

```text
rankSize == 128 -> TileXRGroup128::MoonEpCombineV2Group
otherwise       -> MoonEpCombineV2
```

当前两个 header 在匿名命名空间内定义了大量同名常量、结构和 helper，不能直接同时包含。
Group header 的实现必须整体进入独立的 `TileXRGroup128` 命名空间。原 header 不做对应
重构，避免扩大非 128P 路径的修改面。

所有 block 在同一 launch 中读取相同的 `rankSize`，必须进入同一分支。入口在解引用前仍
检查 `commArgs`，Group 类本身不额外承担非 128P fallback。

## 5. 固定 128P Group 映射

拓扑和编号以 `README_GroupAll2All_128P.md` 为权威参考：

```text
rankID = cabinetID * 64 + serverID * 8 + cardID
groupID = cabinetID * 4 + cardHalf * 2 + serverParity
groupInnerIdx = (serverID >> 1) * 4 + (cardID & 3)
```

每个 Group 包含同一机柜、相同 server parity 的四台 Server，每台 Server 提供连续四卡
半组。每个 step 中源 Group 和目标 Group 一一对应；每个源 Group 在 8 step 中遍历全部
8 个目标 Group。

发送 core 直接作为目标 Group 内索引：

```text
sendPeer(rank, step, core) = GetSendDstRank(rank, step, core)
```

接收 core 直接作为源 Group 内索引：

```text
recvSource(rank, step, core) = GetRecvSrcRank(rank, step, core)
```

因此每个发送 rank 在一个 step 中由 16 个 core 访问目标 Group 的全部 16 个 rank；每个
接收 rank 的 16 个 core 分别监听源 Group 的 16 个 rank。8 step 合并后恰好覆盖完整
128 x 128 通信矩阵。

映射 helper 放在 Combine V2 的 schedule wrapper 中，供 AICore 和 Host unit test 使用，
不修改原通用 schedule mode 的行为。

## 6. Point-to-Point Done/Credit 所有权

对接收 rank `D`、step `s` 和接收 core `i`：

```text
currentSource = GetRecvSrcRank(D, s, i)
nextSender = GetRecvSrcRank(D, s + 1, i)
nextSenderCore = GetGroupInnerIdx(D)
```

`D/core i` 只等待 `currentSource` 的一个 Done。Done 匹配后，`D/core i` 向
`nextSender/nextSenderCore` 发布一个 credit。

`nextSenderCore` 一般不等于 `i`。接收 core 表示当前源 Group 的 inner index；credit
目标 core 表示下一 sender 向 `D` 发送时使用的目标 Group inner index。不能因为
`rank0/core0` 示例中两者恰好都为 0，就把 credit 固定发给同号 core。

### 6.1 rank0/core0 示例

```text
rank0 属于 L0，groupInnerIdx = 0
step0 source = L1/idx0 = rank8
step1 source = L7/idx0 = rank76
```

`rank0/core0` 收到 `rank8` 的 Done 后，向 `rank76/core0` 发布 credit；`rank76/core0`
获得授权后在 step1 向 rank0 发送。

### 6.2 单 Done

每个 source rank 对每个 destination rank 每 step 只发送一个 Done，receiver 也只轮询
一个 Done 槽。

跨 Server payload 使用两个独立 CLOS QP。单个 QP 尾部的 Done 不能证明另一 QP 的
payload 已完成，因此 CLOS 路径采用以下顺序：

```text
发布两条 lane 的 payload
-> 每条非空 lane 的最后一个 payload WQE 请求 ordered completion
-> 本 core 等待全部非空 payload lane CQ
-> 通过 6-port lane 单独发布一个 Done WQE
-> 等待该 Done WQE CQ
```

为避免额外 data-fence WQE，发送扫描期间分别在 UB 中保留每条 lane 的最后一个 payload
WQE，确认该 peer 不再有 payload 后再设置 ordered completion 并发布。某 lane 没有
payload 时不等待该 lane CQ；两条 lane 都没有 payload 时直接发布 Done。

同 Server Fullmesh 只有一个数据 QP，保留一个 ordered Done。Self 不发送 Done，以本地
copy 完成为等价完成条件。

### 6.3 单 Credit

每个 transition 只发布一个 64 Byte credit signal，目标 sender core 也只等待这个
signal。signal 至少包含完整 magic、transition step、source rank/core、target rank/core
和 guard，避免把陈旧或错路由写入误认为授权。

远端 credit 通过现有 CLOS 6-port lane 在 UB 中构造 WQE，经 MTE3 发布并用 `st_dev`
doorbell。publisher 等待本 core 的 credit CQ 后才复用 source buffer。若 next sender 是
本 rank，则直接写本地 receive slot并执行必要的 cache 可见性操作。

credit receive/source 区域复用现有 grant workspace；Host layout 和 Kernel 参数暂不
改变。新路径不使用 wrapped terminal Grant token。

## 7. Step 状态机

step0 无前置 credit，所有 core 可独立开始发送。

step0 至 step6，每个 core 独立执行：

```text
SendPayload
-> WaitPayloadCq
-> PublishSingleDone
-> WaitDoneCq
-> WaitCurrentInboundDone
-> PublishNextCredit
-> WaitCreditCq
-> WaitOwnNextCredit
-> next step
```

step7 执行：

```text
SendPayload
-> WaitPayloadCq
-> PublishSingleDone
-> WaitDoneCq
-> WaitFinalInboundDone
-> finish
```

step7 不发布 credit，也不等待 credit。每个 core 只依赖自身的 SQ/CQ、一个 inbound Done
和一个 inbound credit，不等待其他 core 的 step 状态。

## 8. 同步与失败边界

Group 路径删除：

- 发送前跨 rank `RunGlobalBarrier()`；
- phase 中点的 global barrier；
- 每 step 的 Server-Grant collective/admission；
- 任何新增的 step 级 `SyncAll<true>()` 或软件 barrier。

保留单卡内以下汇合：

- 初始化和配置验证；
- step7 全部 core 结束后的失败状态汇合；
- Reduce 前后的既有卡内同步。

单 core 失败后不再发布无依据的 Done 或 credit。其他依赖 core/rank 通过现有有界 timeout
退出。Group 路径不通过全 rank barrier传播首错误。

## 9. 第一阶段：Group Kernel 清理

清理目标是删除原协议负担，形成后续实现可直接扩展的编译骨架。该阶段允许 Group
`Process()` 暂时不执行完整 128P 数据面。

### 9.1 保留的能力

- Kernel `Init()`、UB buffer 初始化和参数/registry 基础校验；
- route selection load、mask、index 收集；
- CLOS payload WQE 的 UB 构造、MTE3 SQ 发布和 doorbell；
- Fullmesh payload WQE、QP 初始化和 CQ 处理；
- remote field、registration range 和 QP context 解析；
- SQ/CQ ring copy、CQ poll 和队列回收 helper；
- Self copy 的行复制和 tiled copy；
- ReduceHidden 及其 UB/MTE helper；
- failure record、poison、profile point/metric 和 profile writer；
- 初始化、最终状态和 Reduce 所需的卡内 collective helper；
- `doneBase_/doneOffset_`、`grantBase_/grantOffset_` 和
  `controlSourceBase_`，供后续单 Done/Credit 复用。

### 9.2 删除的旧能力

- `kEnableFullSync`、FullSync build context、SIMT builder、receive/source wait、CQ 和
  global barrier 全部实现；
- `fullSyncReceiveBase_`、`fullSyncSourceBase_`、`fullSyncReceiveOffset_`、
  `fullSyncStartCycles_` 等仅服务跨 rank FullSync 的成员；
- Legacy Grant 的 `WaitStepGrant()`、`PublishLocalGrant()`、`SubmitSelfGrant()` 及 payload
  尾部 Grant 分支；
- Server-Grant signal 初始化、publisher、shard/direct wait 和 admission；
- 旧 `SERVER_PAIR_PARITY` schedule mode 常量、Successor 和 phase barrier 调用；
- 旧 `WaitInboundDone()` 全 source 汇总轮询及 `sourcesPerCore_`；
- 原 `Process()` 中依赖旧 schedule、Grant、FullSync 和最终统一 Done 的 step loop；
- 仅用于上述被删路径的结构、常量、成员变量、profile 临时状态和函数声明。

`fullSyncBarrierBase_` 暂时保留，因为既有卡内 collective status 仍复用这一区域。Kernel
入口参数中的 full-sync offsets 继续保留以维持 ABI；Group `Init()` 对不再使用的参数
显式忽略。

### 9.3 清理阶段验收

- Group header 进入独立命名空间，可与原 header 同时 include；
- kernel 入口能够按 128P/非128P 分支构造不同实现；
- Group header 不再引用 Legacy Grant、Server-Grant、RunGlobalBarrier、phase barrier 或
  旧全量 `WaitInboundDone()`；
- 非 128P 分支仍使用原 `MoonEpCombineV2`；
- source guard 更新为检查隔离和已删除协议，不再要求被删函数存在；
- Host/unit 中与原实现无关的既有测试不被清理阶段修改；
- 完成 CANN 9.1 AICore target compile，若环境暂不可用则明确记录该验证缺口。

### 9.4 清理结果与后续约束

清理阶段已完成以下落地：

- Group header 已进入独立 `TileXRGroup128` 命名空间，Kernel 入口同时 include 原实现与
  Group 实现，并仅在 `rankSize == 128` 时选择 Group；
- 已删除 Legacy Grant、Server-Grant、FullSync、phase barrier、旧全 source Done wait、
  `SubmitPair()`、`SendRemoteStep()` 及旧 schedule `Process()`；
- 保留 CLOS/Fullmesh WQE 构造与发布、CQ、Self、Reduce、失败记录、profiling 和卡内
  collective helper；full-sync receive/source ABI 参数仅显式忽略，barrier workspace 继续
  供卡内 collective status 使用；
- Group `Process()` 在新状态机接入前显式汇报 `INVALID_CONFIG`，不写最终 Reduce 输出，
  防止清理中间态被误认为可用的 128P 数据面；
- source guard 改为约束命名空间隔离、128P 分流、底层保留能力和旧协议符号清零。

后续实现 CLOS lane 初始化时，首个 peer 必须由新的 Group send mapping 提供。旧实现用
`SERVER_PAIR_PARITY` schedule 推导首 peer；清理后 `InitLaneStates(firstPeer)` 已改为显式传参，
不能用本 rank 或旧 schedule 代替，否则会校验错误的 SQ/CQ context。双实现同入口开发时，
应先隔离 header 级常量、结构和 SIMT helper 的命名空间，再同时 include；只改类名不足以
避免重定义。

## 10. 后续实现阶段

清理骨架确认后，按以下依赖顺序实现：

1. 增加 128P Group send/receive/credit mapping helper 和穷举 Host oracle。
2. 重构 CLOS payload terminal WQE，支持每 lane 一个最终 CQ 和单 Done。
3. 实现逐 step 单 source Done wait、单 credit publish/wait 和本地 credit 路径。
4. 接入 Fullmesh、Self、step7 terminal 规则和最终 Reduce 汇合。
5. 更新 source guard、profiling 解释和 128P hardware probe。

## 11. 验证策略

### 11.1 Host 与 source guard

- 128 rank x 8 step x 16 core 穷举 send/receive 互逆；
- 每 step 8 个 Group 的目标为双射；每源 Group 八步覆盖全部目标 Group；
- 每 rank 八步覆盖全部 128 destination/source 且不重复；
- 每个 `(targetRank, transition, targetCore)` 恰有一个 credit writer；
- credit writer/target rank/core 双向反推一致；
- cardHalf0 的 Self Group 位于 step2，cardHalf1 位于 step7；
- step6 credit 授权 step7，step7 无 credit slot访问；
- CLOS 只发布一个可观察 Done，receiver 只读取一个 Done；
- 每个 transition 只发布和等待一个 credit；
- step loop 不包含跨 core barrier。

### 11.2 Compile 与硬件

硬件测试前先在相同设备和拓扑执行官方 HCCL Test，并按各 rank correctness 判断环境
基线。部署使用 mutagen `one-way-safe`，远端测试通过脚本和 SSH 执行。

目标硬件验证只声明 128P Ascend950/A5 范围：

1. CANN 9.1 target compile；
2. 至少三轮覆盖 epoch0 -> epoch1 -> epoch0；
3. balanced、model-skew、sparse 和空 peer 路由；
4. 128/128 rank exact correctness；
5. 无 CQ、Done、credit、registration 或 timeout 错误；
6. profiling 证明每 core 每 step 只有一个 Done 和一个 credit，且无 step barrier；
7. 关闭 profiling/DFX 后报告 `avg_ms`、`max_ms` 和 rank 离散度。

Host、source guard、target compile 或非 128P 测试不能证明 128P UDMA 数据面正确性。

## 12. 风险

- 两个 header 未完成命名空间隔离时，同时 include 会直接重定义。
- 单 Done 若早于任一 payload QP 完成，会让 receiver 提前授权并破坏数据可见性。
- 把接收 core `i` 错当成 credit target core，会在非 groupInnerIdx0 rank 上错路由。
- 删除 step barrier 后，任何隐藏的共享 core 状态都会转化为竞态，新增状态必须按 core
  独占或用唯一 workspace slot隔离。
- Self 和空 payload 不产生普通数据 CQ，状态机必须显式处理，不能等待不存在的 CQ。
- 清理阶段功能不完整，不能把 compile 通过描述为 Group 通信正确。

## 13. 完成标准

只有在设计确认、清理骨架完成、新状态机实现完成，并取得 128P 硬件 exact correctness
证据后，才能把 Group 路径标记为功能完成。清理阶段只标记为“协议骨架完成”。

## 14. 代码实现记录

2026-08-17 已完成第一版代码实现：

- schedule wrapper 增加 128P send/receive/group-inner 映射和穷举 Host oracle；
- credit receive/source 按 `epoch x transition(1..7) x core` 复用原 Grant workspace，
  两个区域合计 28672 Byte，不改变 Host layout 和 Kernel 参数 ABI；
- CLOS 仅等待非空 payload lane 的末端 CQ，随后通过 six-port lane 独立发布一个 Done；
- receiver 每 step 只读取一个 source Done，并只向下一 step 的 sender/core 发布一个 Credit；
- step0 不等待 Credit，step6 发布 step7 Credit，step7 不发布后继 Credit；
- step 热循环不包含 `SyncAll<true>()`，只保留循环外的卡内初始化、最终状态和 Reduce 收敛。

实现中的可复用注意事项：预填充 WQE buffer 会跨批次复用，payload builder 必须在每次重建
WQE 时显式恢复普通 `flag`。否则中间或末端批次设置的 ordered-completion 会残留到下一批，
造成未计入 `wqeCnt` 的额外 CQ。当前 Host/source guard 只能验证映射、槽边界和源码顺序，
不能替代 CANN 9.1 target compile 或 128P UDMA 数据面验证。
