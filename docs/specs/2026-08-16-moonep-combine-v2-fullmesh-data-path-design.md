# MoonEP Combine V2 Fullmesh 数据通路设计

## 状态

- 日期：2026-08-16
- 状态：实现完成，编译与 CPU/source 验证待最终重跑，Ascend950 硬件待验证
- 范围：通信 runtime、Combine V2 Host/Kernel、诊断与验证
- 本文不包含性能结论；硬件门禁未执行前不得声明 Fullmesh 数据面可用。

本文补充现有 Combine V2 ring、full-grant 和 full-sync 设计。若本文与
`2026-08-14-moonep-combine-v2-full-sync-design.md` 中“full-sync WQE 不产生 CQ”
的约束冲突，以本文第 9 节为准。

## 1. 目标和已确认决策

目标是在不改变现有 32 个共享 CLOS QP 的前提下，为每个 8 卡服务器增加独立的
Fullmesh 直连数据面：

1. 同服务器远端卡的 Combine V2 payload 和 done 走 Fullmesh。
2. Fullmesh done 的 CQE 成功后，才允许发布该 step 的 grant。
3. 远端 grant 始终走现有 CLOS QP；跨服务器数据继续走现有 CLOS 路径。
4. Fullmesh 能力或当前注册内存不完整时，Combine V2 直接返回
   `TILEXR_MOONEP_ERROR_NOT_SUPPORTED`，不得回退到 CLOS 数据路径。

已确认的 QP 资源模型如下：

| 域 | 逻辑槽位 | 每卡实际建立 | 用途 |
|---|---:|---:|---|
| 共享 CLOS 6-port | 16 | 16 | 跨服务器 payload、done、grant、full-sync |
| 共享 CLOS 2-port | 16 | 16 | 跨服务器 payload、done、grant |
| Fullmesh direct | 8 | `localRankSize - 1`，标准 8 卡服务器为 7 | 同服务器 payload 和 done |

Fullmesh 的 8 个槽位按目标 `localRank` 编号。本卡对应自身的槽位存在于逻辑布局中，
但不建立 QP，也不得被 Kernel 访问。现有采集到的 Ascend950 RootInfo 每卡提供 7 个
单端口 Fullmesh EID、一个 6-port CLOS EID 和一个 2-port CLOS EID，与该模型一致。

## 2. 非目标

- 不改变现有 32 共享 QP 的编号、EID 选择、队列或 6:2 流量比例。
- 不让 PrefetchWeight、ReduceGrad 或其他算子消费新增 Fullmesh QP。
- 不改变 Planner V3 的 `dstLocal` 编码、Combine V2 输出布局或 Reduce 语义。
- 不改变跨服务器 payload 的现有 `3:1` 六口/两口拆分。
- 不使用 Fullmesh 发送 grant 或 full-sync 跨服务器信号。
- 不提供 Fullmesh 缺失时的兼容回退。
- 不把 Host、模拟器或 910B 结果作为 Fullmesh UDMA 数据面证据。

## 3. Runtime 资源模型

### 3.1 两个独立 UDMA 子域

现有 `CommArgs::udmaInfoPtr` 继续指向 `qpNum == 32` 的共享 CLOS `UDMAInfo`。
不得把 8 个 Fullmesh 槽位追加到这个 `UDMAInfo`，也不得让
`TileXRUDMAGetQpCount()` 从 32 变成 40。

Runtime 新增版本化的 Fullmesh capability/device view，逻辑内容至少包括：

```text
version
slotCount = 8
connectedCount = localRankSize - 1
localRank
validPeerMask
registrationReady
UDMAInfo(qpNum = 8)
```

`CommArgs` 在尾部增加 Fullmesh device view 指针，并增加独立 capability flag。
旧的 CLOS pointer、registry pointer 和 shared-QP flag 保持原义。Host 提供只读查询接口，
使 Combine V2 能校验 slot 数、有效 peer mask 和当前注册代次；其他算子无需读取该视图。

### 3.2 Direct QP 建立和对接

设本卡 `localRank=a`，同服务器目标卡 `localRank=b`：

```text
local fullmesh slot  = b
remote fullmesh slot = a
logical diagnostic QP = 32 + b
```

本卡创建 `FM[a -> b]`，通过 topology resolver 绑定 `a` 到 `b` 的单端口 EID；
目标卡创建 `FM[b -> a]` 并绑定反向 EID。两端导入时允许 local/remote slot 不同，不能继续
假设远端 QP index 与本地相等。

Device view 中只有 `(sameNodePeer=b, slot=b)` 的 WQ、CQ 和 remote memory entry 有效。
以下条目必须保持无效且不可访问：

- `slot == localRank`；
- 跨服务器 peer；
- 与目标 `localRank` 不匹配的 slot；
- 当前注册代次尚未完整导入的条目。

### 3.3 Capability 发布

Fullmesh capability 只有在所有 rank 完成以下步骤并进行 rank-wide 一致性确认后才能发布：

1. RootInfo 和 topology 文件可解析；
2. 每个本地远端 peer 都能解析唯一的正反向单端口 EID；
3. 所需 direct QP、SQ、CQ、doorbell 和远端 endpoint 全部建立；
4. 有效 peer mask 与 `localRankSize` 一致；
5. 当前普通注册内存在所有 direct QP 上完成 MR 注册和远端导入。

任一步失败时，清理全部临时 Fullmesh 资源并在所有 rank 清除 capability/pointer。
现有 32 CLOS 子域可以继续存在，但 Combine V2 Host 必须拒绝启动。

Fullmesh 初始化开始前必须记录已有 context/token 所有权。初始化失败时先回收本阶段创建的
QP/CQ/channel，再只释放相对该快照新增的 token 和 context；已有 CLOS context/token 不得
回滚。仍被未清队列引用或底层释放失败的句柄必须保留为可重试状态，不能从账本中丢失。

普通 UDMA 注册的 CLOS 和 Fullmesh device view 必须按同一个注册 generation 原子发布。
不得出现新 CLOS registry 搭配旧 Fullmesh mem info 的组合。Profile registration 继续只描述
现有 32 CLOS QP，不扩大当前 32-binding 公共 ABI。

每个 registration generation 内，CLOS 与 Fullmesh 分别拥有 local MR 和 remote import
账本。Fullmesh 注册失败时按 remote import、local MR、device info 的顺序只清理 Fullmesh
候选资源；清理完整后允许发布不带 Fullmesh capability 的 CLOS generation，Combine V2
仍因 capability 缺失而拒绝启动。若 Fullmesh 局部清理失败，则整个候选 generation 失败并
进入现有可重试清理流程。

## 4. 拓扑和 QP 选择

Kernel 使用显式的服务器判断，不能根据拓扑或 QP 可见性猜测 locality：

```text
sameServer(lhs, rhs) =
    lhs / localRankSize == rhs / localRankSize
```

对一个非 Self 的 data peer：

```text
sameServer(rank, peer): Fullmesh slot = peer % localRankSize
otherwise:              existing CLOS lane pair for this core
```

Host 继续要求标准多机配置的 `localRankSize == 8`。2P 至 7P 配置只要求为当前 job 中的
`localRankSize - 1` 个 peer 建立 direct QP；逻辑 slot 数仍为 8，job 外 slot 保持无效。

当前 single-ring 下，同服务器 peer 出现在每个 source 的本地 group；设计不得依赖它固定
出现在最后一个 step。所有判断都必须基于实际 `peer` 和 `successor`，从而保持
bidirectional schedule 的可扩展性。

## 5. Step 状态机

### 5.1 跨服务器远端 step

保持现有路径：

```text
build CLOS payload on 6-port/2-port lanes
-> append CLOS grant to successor on each lane
-> append CLOS done on each lane (ordered completion)
-> publish both SQs
-> wait both CLOS CQ targets
-> wait both inbound grant tokens
```

done CQ 覆盖本 lane 的 payload、grant 和 done。空 payload 仍提交 grant+done 并产生 CQE。

### 5.2 同服务器远端 step

同服务器路径改为：

```text
select all rows for peer
-> build all payload WQEs on Fullmesh slot(peer.localRank)
-> append one Fullmesh done WQE (ordered completion)
-> publish Fullmesh SQ
-> wait and validate Fullmesh CQ
-> publish this step's two grants on existing CLOS lanes, or publish locally
-> wait deferred CLOS grant CQ targets when grant is remote
-> wait both inbound grant tokens
```

必须满足以下 happens-before：

```text
Fullmesh payload writes
  < Fullmesh done remote visibility
  < successful Fullmesh CQ consumption
  < CLOS grant doorbell
  < outbound grant CQ and inbound grant both ready
  < next data step
```

Fullmesh CQ 未就绪、status 非零、substatus 非零或队列账本非法时，不得写本地 grant，
不得构造或发布远端 CLOS grant。

远端 grant 仍为每个逻辑 CLOS lane 一个，共两个 WQE。下一发送者需要分别获得六口和
两口 lane 的 admission，不能用一个 grant 代替。两个 grant WQE 都请求 ordered
completion；最终进入下一 step 前，两个 CQ target 都必须被消费。实现可交错轮询 outbound
grant CQ 和 inbound grant token，但两组条件必须全部满足。

### 5.3 Self step

Self 保持本地 MTE copy，不建立或访问 Fullmesh QP，也不发送 done。Self copy 完成后：

- successor 为本卡：在 GM 中发布两个本地 grant；
- successor 为远端：沿用两个 CLOS grant-only WQE 和对应 CQ；
- 然后等待本 step 的两个 inbound grant。

### 5.4 空 payload

同服务器远端 peer 即使没有任何选中行，也必须发送 Fullmesh done 并产生 CQE；只有这样
才能证明该 step 已完成并安全发布 grant。禁止用 `selectedCount == 0` 跳过 direct QP。

## 6. Done 和 Grant 协议

### 6.1 Done

现有 done workspace 不扩容。Fullmesh done 复用 source 的 lane-0 done slot：

```text
DoneIndex(epoch, sourceRank, 0)
```

接收方按 source locality 决定等待条件：

| source | 等待条件 |
|---|---|
| Self | 0 个 done |
| 同服务器远端 | lane 0 的 1 个 Fullmesh done |
| 跨服务器远端 | lane 0/1 的 2 个 CLOS done |

Fullmesh 路径不写 lane-1 done，接收方也不得等待或解释该槽。done token 仍编码实际 data
step，magic、epoch 和 `ReceiveStep()` 语义不变。

### 6.2 Grant

Grant workspace、index 和 token 不变。每个 data step 仍发布并等待两个 grant，terminal
round 仍使用 wrapped step-zero token。变化只在同服务器远端 step 的发布时间：从
“与 payload 同批、CQ 前已发布”改为“Fullmesh CQ 成功后发布”。

所有远端 grant 使用现有 CLOS 6-port/2-port QP；即使 alternate schedule 得到同服务器
remote successor，也不把 grant 放到 Fullmesh 数据面。

## 7. WQE、Doorbell 和 UB 约束

- 所有 Fullmesh payload、done 和 deferred CLOS grant WQE 必须完整在 UB 中构造。
- 不得用 scalar/direct-GM store 构造或修补 SQ WQE。
- UB 到 SQ 的写入必须使用 MTE3；MTE3 完成后才能用 `st_dev` 更新 head、completion count
  和 doorbell。
- Fullmesh payload 不再做 3:1 lane split。现有连续 issue UB 总容量可复用，但必须证明
  能容纳最多 128 个 payload WQE 加 1 个 done WQE。
- CQ `entryIdx` 必须先按 SQ depth 归一化，再结合 absolute head/tail 回收；不能把 cycle bit
  当作越界。
- 每个 completion-producing submission 都必须更新 `submittedHead` 和独立的 `cqTarget`。

## 8. SQ/CQ 账本

每个 active core 保留：

- 两个现有 CLOS lane state；
- 至多一个当前 step 使用的 Fullmesh state。

同服务器远端 step 有两个 completion phase：

1. Fullmesh payload+done 的一个 CQ target；
2. 两个 CLOS grant-only submission 的各一个 CQ target。

跨服务器 step 仍只有两个 CLOS CQ target，且它们覆盖 grant。Self step 只在 successor
远端时产生两个 CLOS CQ target。

Kernel 返回前，所有本 invocation 访问过的 Fullmesh/CLOS 队列必须满足：

```text
cqTail == cqTarget
sqTail == submittedHead
head - tail == 0
```

CQ target 到达只是完成条件之一。Deferred CLOS grant-only CQ 消费后必须立即核对
`tail == submittedHead` 和 `head == tail`；任何不一致都按 CQ 账本错误处理，不得进入下一
step。

至少连续运行三轮以覆盖 epoch 复用、SQ wrap、CQ owner cycle 和 direct QP 重用。

## 9. 与 Full-Sync 的关系

Full-sync 继续使用每核 6-port CLOS QP，不改为 Fullmesh。但是原设计让 full-sync WQE
不产生 CQ，并依赖后续正常 CLOS CQ 顺带回收。该假设在 2P 至 8P 全 Fullmesh 数据路径中
不再成立：这些 core 可能整次 invocation 都没有后续 CLOS data CQ。

因此 full-sync 必须改为自包含账本：每个发布过同步 WQE 的 6-port QP，其 batch 最后一个
WQE 请求 ordered completion，发送方在进入正常 data step 前消费该 CQ。full-sync 阶段
结束时必须 `head == tail`，不能依赖后续数据路径回收。

完整顺序为：

```text
publish full-sync CLOS batch
-> wait local full-sync CQ
-> wait all expected remote sync signals
-> active-core barrier
-> enter data steps
```

该调整同时消除 full-sync 和后续 CLOS/Fullmesh route 之间的隐式耦合。

## 10. Host 和 Kernel 校验

Combine V2 Host 在分配 magic 和 launch 前必须同时验证：

- 原有 UDMA、shared-QP、registry 和 32 CLOS QP 条件；
- Fullmesh capability flag 和 device view 非空；
- `slotCount == 8`；
- `connectedCount == localRankSize - 1`；
- valid peer mask 精确排除 Self，并覆盖所有本地 job peer；
- Fullmesh registration generation 与 CLOS registry generation 一致；
- 每个本地远端 peer 的 workspace 范围在对应 direct QP 上已注册；
- 所有 rank 的 capability 已由 runtime 做一致性收敛。

任何条件不满足都返回 `TILEXR_MOONEP_ERROR_NOT_SUPPORTED`。不读取环境变量启用旧路径，
不打印警告后继续，也不只在 device safety check 中发现缺失。

Kernel 仍执行必要的防御校验。Fullmesh CQ owner/status/substatus 检查属于 grant 发布协议，
必须始终启用，不能被 benchmark 的 `kEnableSafetyChecks` 编译掉。可选 DFX 只控制额外范围
检查、超时明细和日志，不控制 CQ 成功判定。

## 11. 失败处理和收敛

| 失败点 | 行为 |
|---|---|
| RootInfo/topology/direct QP 建立失败 | rank-wide 清除 Fullmesh capability；Combine Host 拒绝 launch |
| Fullmesh MR 注册或 remote import 不完整 | 清理 Fullmesh 候选资源，不发布 Fullmesh view；Combine Host 拒绝 launch |
| Fullmesh 候选资源清理失败 | 放弃整个候选 registration generation，保留可重试清理账本 |
| Fullmesh 初始化失败 | 先清 direct 队列，再精确回滚本阶段新增 context/token；已有 CLOS 句柄不动 |
| Fullmesh SQ/WQ/CQ context 非法 | device 记录 invalid-config；不发送 grant |
| Fullmesh CQ timeout/error | 记录首错误；不发送 grant；进入现有失败收敛 |
| Deferred CLOS grant CQ timeout/error | 记录首错误；不得进入下一 step |
| Inbound grant timeout | 保留现有 grant-timeout 语义 |

Failure record 继续使用现有 `lane` 和 `qp` 字段：Fullmesh failure 使用新的 transport/lane
标识，`qp` 记录逻辑编号 `32 + peerLocalRank`。实现必须直接保存实际 failure QP，不能再从
`failureLane` 推导 CLOS QP。

Fullmesh CQ 失败后不发布 grant 是安全性的核心。否则后继 source 会在前一 source 的数据
和 done 未确认完成时进入同一目标，破坏 ring admission 协议。

## 12. Profiling 和可观测性

Profile schema/version 应增加以下边界，且 trace decoder 同步更新：

- Fullmesh WQE build/submit end；
- Fullmesh CQ success；
- deferred CLOS grant submit；
- deferred CLOS grant CQ success；
- transport kind、data peer、successor 和逻辑 QP。

硬件 trace 必须能够证明：

```text
same-node payload/done: logical QP 32..39, transport=fullmesh
cross-node payload/done: QP 0..31, transport=clos
remote grant:            QP 0..31, transport=clos
FM_CQ_SUCCESS < CLOS_GRANT_DB
```

除 grant 和 full-sync 外，同服务器 payload 字节数在 CLOS 上必须为零。

## 13. 受影响组件

预计实现边界如下：

- `src/comm/udma/`：Fullmesh direct QP lifecycle、非对称 slot import、注册 view 和能力收敛；
- `src/include/comm_args.h` 及 UDMA device helpers：追加版本化 Fullmesh view，不改现有
  `udmaInfoPtr` 语义；
- `src/moonep/common/moonep_combine_schedule.h`：same-server、Fullmesh slot 和 done
  expectation helper；
- `src/moonep/combine_v2/host/`：强制 capability/registration 校验；
- `src/moonep/combine_v2/kernels/`：单-lane Fullmesh builder、deferred grant 和双阶段 CQ；
- `tests/udma/`、`tests/moonep_combine_v2/`：路由、账本、状态机和 source guard；
- profiling decoder、hardware probe 和多机 runner：输出 transport/QP/时序证据；
- 现有 full-sync spec/测试：改为自包含 CQ 回收。

PrefetchWeight、ReduceGrad 的 active QP map 和 32-binding profile ABI不变，只增加回归测试
证明它们没有看到或消费 Fullmesh 子域。

## 14. 验证策略

### 14.1 Host 和纯逻辑测试

- 2/8/16/32/64/128P 的 locality、peer、receive-step 和 successor 穷举；
- 每卡 valid peer mask 精确包含 `localRankSize - 1` 个 peer；
- `FM[a -> b]` 导入 `FM[b -> a]`，Self 和跨节点条目无效；
- 同服务器远端等待 1 个 done，跨服务器等待 2 个，Self 等待 0 个；
- 每个 step 仍发布/等待两个 grant，terminal closure 不变；
- 空 payload 仍产生 Fullmesh done CQ；
- Fullmesh CQ 失败路径不发布任何 local/remote grant；
- CLOS QP count/query/profile 仍为 32，PrefetchWeight/ReduceGrad map 不变；
- capability 缺失、mask 缺口、generation 不一致都在 Host launch 前失败。

### 14.2 Runtime 和 source/compile 检查

- 7 个 direct QP 的 topology EID 正反向匹配；
- 现有 32 shared QP config/wire descriptor byte-for-byte 语义不变；
- Fullmesh WQE 全 UB 构造、MTE3 发布、`st_dev` doorbell 顺序正确；
- same-node path 不调用 CLOS payload builder；
- grant doorbell 的控制流严格受 Fullmesh CQ success 支配；
- deferred grant CQ 完成后验证 `cqTail/cqTarget` 和 `head/tail/submittedHead` 闭环；
- Fullmesh 注册失败不遗留 local MR、remote import 或 device info；
- Fullmesh 初始化失败只回滚增量 context/token；
- full-sync 自己产生并消费 CQ，不依赖后续 step；
- Kernel signature、C++14 和 CANN 9.1 编译通过。

### 14.3 硬件门禁

在相同设备和拓扑上先运行安装版官方 HCCL Test；单机 Fullmesh/AIV 基线使用
`all_reduce_test -a aiv_only`，按每 rank correctness 判断。基线失败时停止 TileXR 调试。

随后按以下顺序验证：

1. 8P direct-QP micro-probe：覆盖所有 56 个有向本地 peer，校验数据、CQ、逻辑 slot 和
   实际 EID；
2. 8P Combine：全部远端 payload 走 Fullmesh，覆盖 full-sync 开/关和空 payload；
3. 16P：同一 step 中同时存在 Fullmesh 和 CLOS core，证明两个子域并行且 grant 有序；
4. 32/64/128P：覆盖多 step、terminal grant、balanced/model-skew/sparse/unique routing；
5. reduction 开/关、至少三轮 magic/epoch/SQ/CQ 重用；
6. 注入缺失 route、无效 CQ status 和 registration generation mismatch，确认无 fallback、
   无错误 grant；
7. 关闭 DFX 后与 32-CLOS 基线比较延迟和带宽。

服务器部署使用 mutagen `one-way-safe` 同步；测试通过脚本经 SSH 执行，并保存源码、二进制、
RootInfo、CANN/驱动、rank/device 映射、QP/EID trace 和完整日志 provenance。

## 15. 完成标准与验证边界

代码实现和非硬件验证必须满足：

- 现有 32 shared CLOS QP 的 API、编号和其他算子行为不变；
- 逻辑布局保留 8 个 Fullmesh slot，每卡创建 `localRankSize - 1` 个 direct QP；标准
  8 卡服务器目标为 7 个，Self slot 不建立；
- 同服务器 payload/done 只走 Fullmesh，跨服务器 payload/done 只走 CLOS；
- Fullmesh CQ 成功前不存在 grant publication；
- 所有远端 grant 走 CLOS，且 grant-only CQ 无遗留；
- Fullmesh 不可用或未注册时 Combine Host 明确失败，不发生 CLOS fallback；
- full-sync、data 和 deferred grant 各自完成 SQ/CQ 回收；

以下项目属于后续 Ascend950 硬件 release gate，本次仅 review、CPU/source 测试和编译，
不声明已经满足：

- 标准 8 卡服务器实际建立并互连每卡 7 个 Fullmesh direct QP，Self slot 不建立；
- 8P、16P 和至少一个 32P 以上多机规模 exact correctness 通过；
- 连续多轮后所有访问过的队列 `head == tail`，无 stale done/grant；
- 性能结论只基于实际 Ascend950 Fullmesh/CLOS 硬件数据。
