# MoonEP Combine V2 Grant 到 Credit 迁移设计

## 1. 最终决策与范围

本设计将 `MoonEpCombineV2` 的 Legacy Grant、Server Grant、FullSync 和 phase barrier
全部删除，统一改为 receiver-driven step Credit。

适用规模和 transition 数如下：

| rankSize | step 数 | Credit transition 数 |
| --- | ---: | ---: |
| 2P-16P | 1 | 0 |
| 32P | 2 | 1 |
| 64P | 4 | 3 |
| 128P | 8 | 7 |

最终协议约束：

- step0 不等待 Credit，也不执行全 rank FullSync；
- 非最后 step 在接收方确认当前 inbound Done 后发布下一 step Credit；
- Credit 使用 `CommArgs::creditMems[]` 做 UB 到远端 GM 的 MTE3 copy，不进入 UDMA
  SQ，不产生、不等待 CQ；
- 原 phase 边界按普通 step transition 处理，只依赖 Done -> Credit；
- 2P-16P 只有 step0，删除 Grant 后不增加 Credit；
- Kernel 不再执行任何跨 rank barrier；
- 保留必要的卡内 core 汇合，用于初始化、最终失败状态和 Reduce。这些卡内 `SyncAll`
  不属于本设计删除的 FullSync/phase barrier；
- 保持发送 schedule、数据 QP 数量、公共 Combine API 和 Kernel 21 参数位置兼容。

不在本任务中改变 payload/Done completion 策略，也不合并 Group 路径独立设计的本端 Done
CQ 与远端 Done 公平轮询优化。

## 2. 为什么删除 FullSync 和 phase barrier

FullSync 和 phase barrier 都会让已满足局部依赖的 rank/core 等待无关参与者：

- FullSync 在每次调用 step0 前执行一次全 rank 信号发送、CQ 等待和远端信号等待；
- phase barrier 在 32P step0、64P step1、128P step3 后再次执行同类全 rank同步；
- 任一慢 rank 会把进度差传播到全部 rank，阻止不同 receiver slot 独立推进；
- barrier 使用额外控制 WQE、CQ 和轮询，占用 SQ/CQ 与 AICore 周期；
- Credit 已经为每个接收 slot 建立精确的 Done -> next sender 依赖，全局 barrier 的依赖范围
  过大。

“负优化”必须通过相同代码基线的硬件 A/B 数据证明，不能只根据协议推断。最终性能验收见
第 14 节。

## 3. Dispatch Credit 参考模式

Combine V2 Credit 采用当前 Dispatch 的以下模式：

- `PublishDispatchNextCredit()`：在 UB 中准备 Credit，通过
  `args->creditMems[targetRank]` 执行 MTE3 `DataCopy`；
- `WaitDispatchPeerCredit()`：从本 rank 的 `creditMems[rank]` 通过 MTE2 搬入 UB 并
  轮询 token；
- Credit 发布和等待都不访问 UDMA SQ/CQ。

Combine V2 不直接复用 Dispatch token 格式，但复用相同传输原则：

```text
publisher: UB signal -> MTE3 -> target creditMems
waiter:    local creditMems -> MTE2 -> UB -> validate
```

禁止退化为以下实现：

- 使用 `UDMAPutNbiOnQp()` 发送 Credit；
- 为 Credit 构造 UDMA WQE；
- 为 Credit 增加 completion count、CQ target 或 CQ wait；
- 将 workspace UDMA 注册区继续作为 Credit 数据面。

## 4. 调度与唯一所有者

每个 rank/core 在同一 step 同时具有两个角色：

- sender：向当前 step destination 发送 payload 和 Done；
- receiver：等待当前 step source 写入本 rank/core 管理的接收 slot。

在 `moonep_combine_schedule.h` 增加以下纯函数：

```cpp
MoonEpCombineV2EffectivePeer(source, step, core, rankSize, mode);
MoonEpCombineV2ReceiveSource(destination, step, receiverCore, rankSize, mode);
MoonEpCombineV2TransferCore(source, destination, step, rankSize, mode);
```

`ReceiveSource()` 对固定 `(destination, step)` 按 rank 递增枚举满足
`ReceiveStep(destination, source) == step` 的 source，并以枚举序号作为 receiver core。
每个 step 恰有 `activeCoreCount` 个入站 source，因此每个 active receiver core 管理一个
source。不能继续使用旧 `SourceForCore(core + n * activeCoreCount)` 分组：该分组只适用于
整轮结束后的批量 Done 等待，在 32P 等调度中同一 step 可能为一个 core 匹配多个 source。

`ReceiveCore(destination, source, step)` 是 `ReceiveSource()` 的逆函数，sender 在等待
Credit 时用它定位发布该 Credit 的 receiver core。

`TransferCore()` 在指定 step 中反解 source 到 destination 的 sender core。它必须将
invalid peer 按 self 处理，不能直接复用当前会拒绝 self 的 `SenderCore()`。

对 receiver `D/core C`，完成 step `S` 后发布 transition `S+1` Credit：

```text
nextSource = ReceiveSource(D, S + 1, C)
targetCore = TransferCore(nextSource, D, S + 1)

Credit {
    sourceRank = D,
    sourceCore = C,
    targetRank = nextSource,
    targetCore = targetCore,
    transitionStep = S + 1
}
```

对 sender `R/core T`，开始 step `S>0` 前：

```text
destination = EffectivePeer(Peer(R, S, T), R)
receiverCore = destination 上负责接收 R 的 core

等待 Credit {
    sourceRank = destination,
    sourceCore = receiverCore,
    targetRank = R,
    targetCore = T,
    transitionStep = S
}
```

Host 穷举必须证明每个
`(epoch, transitionStep, targetRank, targetCore)` 恰好一个 writer，且 publisher 字段与
waiter 期望完全一致。正确性不能依赖默认关闭的运行时 safety check。

## 5. Credit IPC 内存布局

`CommArgs` 已提供 `creditMems[TILEXR_MAX_RANK_SIZE]`，Communicator 已负责 Credit IPC
内存的分配、清零、命名、跨 rank open 和释放。Combine V2 在现有 Dispatch Credit 区域后
增加独立区域，不能与 Dispatch slot 复用。

Combine Credit signal 保持 64 Byte：

```text
magic
marker = CRDT
transitionStep
sourceRank/sourceCore
targetRank/targetCore
guard
reserved
```

布局建议：

```text
dispatchCreditBytes = existing 2 x CREDIT_IPC_SLOT_BYTES

combinePlaneBytes = 7 transitions x 16 cores x 64B = 7168B
combineCreditBytes = 2 epochs x combinePlaneBytes = 14336B

combineCreditBase = AlignUp(dispatchCreditBytes, 64B)
totalCreditIpcBytes = combineCreditBase + combineCreditBytes
```

本 rank 的 receive slot：

```text
index = (transitionStep - 1) * 16 + targetCore
offset = combineCreditBase + epoch * combinePlaneBytes + index * 64B
```

32P/64P 只访问有效 transition；2P-16P 不访问 Combine Credit 区域。匹配条件必须同时
校验 magic、marker、transition、source/target route 和 guard。

Host 在启用 Combine V2 前必须确认所有参与 rank 的 `creditMems[rank]` 非空，不能在缺少
Credit IPC mapping 时回退到 Grant 或 UDMA Credit。

原 Combine workspace 中不再分配 Grant/Credit receive/source 区域。

## 6. Step0 协议

以 rank `R/core C` 为例：

```text
D0 = R/C 在 step0 的发送目标
S0 = step0 写入 R/C 接收 slot 的 source
S1 = step1 写入同一接收 slot 的 source
T1 = S1 向 R 发送 step1 时使用的 sender core
```

执行顺序：

1. 不等待 Credit，不执行 FullSync。
2. 发送 `Data(step0, R/C -> D0)`。
3. 在 payload 后追加 `Done(step0)`：
   - CLOS 两个 lane 各一个 ordered Done；
   - Fullmesh 一个 ordered Done；
   - 空 payload 仍发送 Done；
   - Self 完成本地 copy，不发送 Done。
4. 等待发送侧 Done CQ。它只证明本 core 的 outbound payload/Done 已完成。
5. 等待 `S0 -> R/C` 的 inbound Done。CLOS 等两个 token，Fullmesh 等一个，Self 不等。
6. inbound Done 完成后，通过 MTE3 向 `creditMems[S1]` 发布 Credit1。
7. Credit MTE3 完成后直接结束 step0，不执行 Credit CQ wait或 phase barrier。

时序为：

```text
Step0:
Data0/Done0 publish
-> outbound Done CQ wait
-> inbound Done0 wait
-> Credit1 MTE3 publish
-> step0 complete
```

32P 的原 step0 phase boundary 被删除，Credit1 是 step0 到 step1 的唯一放行条件。

## 7. Step1 协议

```text
D1 = R/C 在 step1 的发送目标
S1 = step1 写入 R/C 接收 slot 的 source
S2 = step2 写入同一接收 slot 的 source
```

执行顺序：

1. 从本地 `creditMems[R]` 的 `(epoch, transition=1, core=C)` slot MTE2 load 并等待
   来自 `D1` 对应 receiver core 的 Credit1。
2. Credit1 完整 route 和 guard 匹配后，发送 `Data(step1, R/C -> D1)`。
3. 在 payload 后追加 `Done(step1)`；CLOS/Fullmesh/Self 规则与 step0 相同。
4. 等待发送侧 Done CQ。
5. 等待 `S1 -> R/C` 的 inbound Done1。
6. 32P 的 step1 是最后 step，不发布 Credit2。
7. 64P/128P 通过 MTE3 向 `creditMems[S2]` 发布 Credit2。
8. 不执行 Credit CQ wait或 phase barrier。

时序为：

```text
Step1:
Credit1 wait(MTE2 polling)
-> Data1/Done1 publish
-> outbound Done CQ wait
-> inbound Done1 wait
-> Credit2 MTE3 publish(only 64P/128P)
-> step1 complete
```

64P 的原 step1 phase boundary 被删除，Credit2 是 step1 到 step2 的唯一放行条件。

## 8. 原 phase 边界的替代关系

删除 phase barrier 后，原边界转换为普通 Credit transition：

| rankSize | 原 barrier | 替代依赖 |
| --- | --- | --- |
| 32P | step0 后 | inbound Done0 -> Credit1 -> step1 |
| 64P | step1 后 | inbound Done1 -> Credit2 -> step2 |
| 128P | step3 后 | inbound Done3 -> Credit4 -> step4 |

每个 receiver slot 只放行自己的下一 writer。不同 slot/rank 不再互相等待，因此允许快路径
继续推进，同时仍保持接收空间的唯一所有权。

失败传播从全 rank barrier 改为有界点对点失败：

- 失败 core 不发布没有 Done 依据的 Credit；
- 下游 waiter 在 Done 或 Credit timeout 后退出；
- 卡内最终状态汇合阻止错误输出进入 Reduce 结果；
- 不为快速传播错误重新增加跨 rank barrier。

## 9. 无 FullSync 的跨轮安全证明

删除 FullSync 后，step0 安全依赖以下不可放宽的调用合同：

1. 同一 rank 的 Combine 调用在同一 stream 上严格串行，调用 `N+1` 不能早于本 rank
   调用 `N` 完成；
2. 所有 rank 以相同顺序执行相同 magic 序列，不能跳轮或分叉；
3. 每个调用完成前必须等待本轮所有 inbound Done，并完成 Reduce；
4. scratch 使用两个 epoch，调用 `N` 和 `N+1` 不写同一 epoch；
5. Done token 和 Credit signal 都校验完整 magic/epoch/step，不能用清零代替代际区分。

对任意快 rank `A` 和慢 rank `B`，证明调用 `N+2` 可以安全复用调用 `N` 的 epoch：

```text
A 开始 N+2
=> A 已完成 N+1
=> A 已收到 B 在 N+1 的 inbound Done
=> B 已经开始 N+1
=> 由于 B 同 stream 串行，B 已完成 N
=> B 已完成 N 的 inbound Done wait 和 Reduce
=> B 不再读取 N 的 scratch epoch
```

该关系对所有 `B` 成立，因此 `A` 开始 `N+2` 时，所有远端 rank 都已停止使用调用 `N`
的 epoch。step0 可以直接写当前 epoch，无需全 rank FullSync。

如果未来允许并行 stream、rank 跳轮、异步取消或少于两个 epoch，该证明失效，必须先设计
新的跨轮所有权协议，不能恢复隐式无保护的 step0。

## 10. Credit 发布与等待

### 10.1 PublishNextCredit

1. 在 UB relay 中初始化完整 64 Byte Credit signal；
2. `SyncFunc<S_MTE3>()`；
3. 将目标设为 `args->creditMems[targetRank] + receiveOffset`；
4. 使用 MTE3 `DataCopy` 将 64 Byte 写入目标 GM；
5. `SyncFunc<MTE3_S>()` 后返回。

本地和远端 rank 使用同一路径；`creditMems[rank]` 是本 rank mapping。Credit 不构造 WQE，
不敲 UDMA doorbell，不修改任何 SQ/CQ 状态。

### 10.2 WaitStepCredit

1. 根据 sender rank/core/step 计算 destination 和预期 receiver core；
2. 定位 `args->creditMems[rank]` 的本地 receive slot；
3. clean/invalidate 对应 cache line；
4. 使用 MTE2 将完整 signal 搬入 UB；
5. 完成 MTE2 -> scalar 同步后校验 magic、marker、transition、route 和 guard；
6. 未匹配时有界轮询，超时记录 `CREDIT_TIMEOUT`、expected 和 observed signal。

Credit wait 不访问 UDMA CQ。

## 11. 删除清单与 ABI

删除 common schedule 中：

- Legacy Grant token、slot、index、workspace 和 `LegacyGrantEnabled()`；
- Server Grant signal、guard、mapping、workspace 和 `ServerGrantEnabled()`；
- FullSync signal、guard、generation、boundary 和 phase barrier helper；
- SGRT/FSYN marker 及相关 schedule 测试。

删除 `MoonEpCombineV2` Kernel 中：

- `WaitStepGrant()`、`PublishLocalGrant()`、`SubmitSelfGrant()`；
- `PublishServerGrants()`、`WaitServerGrantShard/Direct()`；
- `RunServerGrantAdmission()`；
- `kEnableFullSync`、`RunGlobalBarrier()` 及 FullSync build/publish/CQ/wait helper；
- phase barrier 分支；
- Grant/FullSync 专用成员、profile 和失败路径。

Host workspace：

- 删除 Grant/Credit workspace；Credit 改用 Communicator `creditMems`；
- 删除 FullSync receive/source workspace；
- 原 `fullSyncBarrier` 区域若仍被卡内 collective status 使用，重命名为
  `collectiveStatusOffset/Bytes`，不得继续保留 FullSync 语义；
- 重新计算 output offset 和 total bytes。

Kernel ABI：

- 保持 21 个参数的位置不变；
- 原 `grantOffset`、`fullSyncReceiveOffset`、`fullSyncSourceOffset` 位置标记为 reserved，Host
  传 0；
- 原 `fullSyncBarrierOffset` 位置改为内部 `collectiveStatusOffset`；
- 参数重命名不改变注册签名。

诊断编号不重排，废弃的 Grant/FullSync status 编号保留为空洞；统一使用
`CREDIT_TIMEOUT`、`DONE_TIMEOUT` 和既有 CQ error。

## 12. 逐文件修改范围

- `src/include/comm_args.h`
  - 扩展 Credit IPC 总字节和 Combine Credit layout 常量。
- `src/comm/tilexr_comm.cpp`、`src/comm/tilexr_comm.h`
  - 按新总字节分配/清零/open/close Credit IPC，保持失败清理完整。
- `src/moonep/common/moonep_combine_schedule.h`
  - 增加通用 Credit signal、offset、guard、route inverse；删除 Grant/FullSync/phase helper。
- `src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h`
  - 改写 step loop、逐 step Done wait、Credit MTE publish/wait，删除两个 barrier。
- `src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_group_kernel.h`
  - 同步公共 Credit 类型和 Credit IPC layout，不恢复跨 rank barrier。
- `src/moonep/combine_v2/host/combine_v2_layout.h/.cpp`
  - 删除 Grant 和 FullSync receive/source workspace，保留并重命名卡内 status 区域。
- `src/moonep/combine_v2/host/combine_v2_launch.cpp`
  - 保持 ABI 位置，废弃参数传 0，传递 collective status offset。
- `src/moonep/combine_v2/common/combine_v2_profile.h`
  - 删除 Grant/FullSync/phase profile 语义，增加 Credit MTE wait/publish 点。
- `tests/moonep_combine_v2/`
  - 更新 schedule、layout、launch、source guard 和 hardware probe。

工作区中用户对 `tilexr_moonep_combine_v2_kernel.cpp` 的 128P Group 入口修改必须保留，不得
恢复或覆盖。

## 13. 实施顺序

1. 扩展 `creditMems` layout、Communicator 分配和 Host mapping 校验。
2. 增加 schedule inverse 与 Credit 唯一写者 Host oracle。
3. 实现 Combine Credit MTE3 publish 和 MTE2 wait 的独立 source/Host guard。
4. 实现逐 step inbound Done wait，切换 step0/step1 状态机。
5. 删除 Legacy Grant、Server Grant、FullSync 和 phase barrier。
6. 删除 workspace 区域并保持 Kernel 21 参数 ABI。
7. 完成 Host/target compile 和多轮硬件正确性。
8. 使用临时编译期 A/B 基线证明两个 barrier 是负优化；最终代码不保留运行时开关。

在 Credit IPC 跨 rank mapping、跨轮安全证明和唯一写者测试通过前，不得删除运行路径中的
旧同步后直接宣称正确。

## 14. 验证与负优化证明

### 14.1 Host 与 source guard

- 穷举 rankSize `2,3,4,5,6,7,8,16,32,64,128`；
- 覆盖 Single Ring、Bidirectional 和当前 Server Pair Parity mode；
- 每 rank/step/core 的 send/receive mapping 互逆；
- 每个 Credit receive slot 恰有一个 writer；
- publisher signal 与 waiter expected route 完全一致；
- Self/invalid-as-self sender core 唯一；
- transition 数分别为 0/1/3/7；
- Dispatch 和 Combine Credit IPC 区域不重叠、不越界；
- Credit publish 只使用 `creditMems + MTE3 DataCopy`；
- Credit 路径不存在 UDMA WQE、completion count、doorbell 和 CQ wait；
- legacy Kernel 不再引用 `RunGlobalBarrier`、`kEnableFullSync`、FullSync signal 或 phase
  barrier helper；
- 两 epoch、至少三轮的事件模型验证第 9 节跨轮证明；
- workspace layout 不再包含 Grant 和 FullSync receive/source 区域。

### 14.2 Compile

- focused Host/unit/source-guard tests；
- C++14 Host build；
- CANN 9.1 AICore target compile；
- 检查 MTE3/MTE2 同步方向、64 Byte copy 和 UB relay 容量；
- 检查 `CommArgs` Host/Kernel ABI 与 Kernel launch 21 参数顺序。

### 14.3 硬件正确性

硬件测试前在相同设备和拓扑执行匹配数据路径的官方 HCCL Test。部署使用 mutagen
`one-way-safe`，远端测试通过脚本和 SSH 执行。

- 2P、8P、16P：证明无 Grant、无 barrier 的单 step 多轮正确；
- 32P：验证 Credit1 跨越原 phase 边界；
- 64P：验证 Credit1-3，特别是 Credit2；
- 128P：验证 Credit1-7，特别是 Credit4；
- epoch0 -> epoch1 -> epoch0 和长轮次 wrap；
- 人工注入 rank 延迟，验证不同 rank 跨 step 进度不一致时仍正确；
- balanced、skew、sparse 和空 payload；
- 所有 rank exact correctness，无 Done/Credit/CQ/IPC timeout。

### 14.4 FullSync/phase barrier 负优化 A/B

A/B 必须使用相同 commit、编译参数、设备、拓扑、输入和轮数，仅通过临时编译期选项切换：

```text
A: Credit 协议 + FullSync + phase barrier
B: Credit 协议 + 无 FullSync + 无 phase barrier
```

两组都必须先通过 exact correctness。分别记录：

- kernel `avg_ms`、`max_ms`、P50/P95；
- rank 间离散度；
- step0 开始时间与原 phase 边界前后等待周期；
- barrier 控制 WQE/CQE 数量；
- Done/Credit wait 周期；
- NPU 侧异常、timeout 和 SQ/CQ outstanding。

只有当 B 在重复运行中的收益超过同配置自然波动、P95 无不可接受回退，且 trace 显示收益
来自删除 barrier 等待和控制 WQE/CQE，才能正式结论“FullSync 和 phase barrier 是负优化”。
文档记录原始轮数、统计方法、绝对耗时差和百分比，不只记录单次最好结果。

最终代码删除临时 A/B 开关和 A 路径，不保留已确认的负优化逻辑。

## 15. 风险与完成标准

主要风险：

- `creditMems` 在任一参与 rank 为空会造成不可恢复的 Credit 路由缺口，必须 Host fail-fast；
- MTE3 publish 若缺少正确同步，waiter 可能观察到部分 signal；
- 两 epoch 跨轮证明依赖同 stream 串行和全 rank 相同调用序列；
- 删除 barrier 后失败通过依赖链 timeout 传播，DFX 必须保留 expected/observed route；
- Credit IPC 扩容影响 Communicator 内存生命周期，partial open 失败必须完整清理；
- `kEnableSafetyChecks=false`，route、slot 和 epoch 正确性必须由 Host oracle 证明。

只有同时满足以下条件，才能完成迁移：

1. active tracked source 中不存在 Legacy/Server Grant、FullSync 和 phase barrier 调用；
2. Credit 只通过 `creditMems` 的 MTE3/MTE2 路径传输，不产生 CQ；
3. 2P-128P schedule、唯一写者和跨轮 epoch 测试通过；
4. Credit IPC allocation/open/cleanup 和 Host fail-fast 测试通过；
5. CANN 9.1 target compile 通过；
6. 目标硬件多轮、rank 延迟、SQ/CQ wrap 和 exact correctness 通过；
7. A/B 数据证明 FullSync 和 phase barrier 是负优化；
8. Kernel ABI、必要卡内 collective、Reduce 和用户现有 128P 分流修改未被破坏。
