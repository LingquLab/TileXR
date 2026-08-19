# MoonEP Combine V2 Hidden/Weight 单 Launch 双通信域设计

## 1. 状态与最终目标

本文档定义 MoonEP Combine V2 在一次 AICore Kernel launch、一套 step loop 内同时完成
hidden 和 route weight 传输的方案。

最终数据面如下：

| 数据 | 通信域 | 传输语义 | 完成依据 |
| --- | --- | --- | --- |
| hidden payload | Shared-QP UDMA communicator，domain 0 | UDMA WQE/SQ/CQ | 现有 ordered Done 和 CQ |
| route weight | 独立 Memory-only communicator，建议 domain 1 | `peerMems[]` 上的 MTE copy | 新增 `weightDone` token |

核心决策：

- 保留 hidden 的现有 UDMA schedule、WQE、SQ/CQ 和 Reduce 逻辑；
- route weight 不构造 UDMA WQE，不占用 hidden 的 SQ/CQ；
- hidden 和 weight 共用同一份 `dstLocal` 路由和同一个 step schedule；
- 每个 step 下发 hidden 并敲 doorbell 后，在等待对应 CQ 的窗口内发送 weight；
- 接收端必须同时观察到 hidden Done 和 weight Done，之后才能消费本轮输入；
- `routeWeightsNvs == nullptr && routeWeightsSk == nullptr` 时保持纯 hidden 路径；
- 对外 Python `Buffer`、`MoonEPCommPlan`、方法参数、关键字参数和返回 tuple 不变；
- 保留现有 C ABI，新增融合入口，不直接改变已有 `TileXRMoonEpCombineStageV2` 的参数列表。

本文档是实现前的设计基线。实现过程中如需改变通信域所有权、完成协议、公共接口或输出
语义，必须先更新设计。

## 2. 当前行为与问题

当前 `TileXRMoonEpCombineStageV2()` 已接受：

```text
routeWeightsNvs
routeWeightsSk
```

但它没有在 hidden Kernel 内融合 weight，而是：

1. 将 hidden 拷入 UDMA registered workspace；
2. 以 BF16、`h` 为 hidden size 启动第一次 Combine V2 Kernel；
3. 拷出 hidden 结果；
4. 将 route weight 拷入同一个 workspace；
5. 以 FP32、`h = 1` 启动第二次 Combine V2 Kernel；
6. 从 active scratch 拷出 `bs * topK` 个 weight。

因此传入 route weight 时当前路径是两次 Kernel launch。第二次 launch 复用了 hidden 的
UDMA 数据面，带来重复的 Host 准备、Kernel 初始化、step schedule、控制 WQE 和 CQ 等待。

Combine V1 已证明 `routeWeightsNvs[source] -> remote[target] -> routeWeightsSk[target]`
可以通过 Memory peer window 完成。本设计复用其路由和 MTE 搬运原则，但不复用 V1 的
Kernel 或 workspace 布局。

## 3. 目标与非目标

### 3.1 目标

- route weight 存在时，Combine V2 只启动一次 AICore Kernel；
- hidden 和 weight 在同一 step loop 内推进；
- hidden 继续使用当前 UDMA fast path；
- weight 使用独立 Memory domain 的 `CommArgs::peerMems[]`；
- weight 传输与 hidden CQ latency 重叠；
- weight 不改变 hidden 的 WQE 数、CQ target 和 QP 分配；
- 保持 route weight 当前输入、输出 shape、dtype 和下标语义；
- 支持连续多轮调用，以 generation tag 隔离旧记录，不读取上一轮残留数据；
- 对 Memory、UDMA 和 Fullmesh 分别给出可观测的失败状态和 profile 数据。

### 3.2 非目标

- 不改变 `dstLocal` 的编码格式；
- 不改变 hidden 的 Reduce 算法或精度；
- 不将 weight 合并进 hidden row，也不为 weight 注册 UDMA memory；
- 不用 scalar/direct-GM store 代替 Memory 数据面的 MTE copy；
- 不在本设计中重构 Combine V2 的整体 schedule、Grant/Credit 或 FullSync 协议；
- 不以 Host/模拟器测试宣称跨机 Memory 数据面已验证；
- 不在缺少 Memory mapping 时静默退回第二次 weight launch。

## 4. 总体数据流

每个 active worker 在一个 step 中执行：

```text
Select current hidden peer
        |
        v
Build/submit hidden UDMA WQEs
final batch appends hidden Done
ring UDMA doorbell
        |
        +-----------------------+
        | UDMA progresses       |
        | asynchronously        |
        +-----------------------+
        |
        v
SendWeightMemoryStep(peer, step)
routeWeightsNvs[source] -> UB record -> remote weightRecord[target]
        |
        v
MTE3 completion/barrier
publish remote weightDone(magic, step)
        |
        v
Wait hidden CQ and existing admission protocol
```

所有 step 完成后，接收侧执行：

```text
WaitInboundHiddenDone()
            AND
WaitInboundWeightDone()
             |
             v
ReduceHidden()
CopyReceivedWeights(routeWeightsSk)
```

第一版在 hidden doorbell 后完整发送当前 step 的 weight，再进入 CQ polling。此时 UDMA
已经独立推进，能够覆盖一部分 CQ latency，同时实现复杂度较低。只有 profile 证明 CQ
polling 饥饿或 weight 发送时间过长时，才进一步改为：

```text
poll one CQ -> send one weight batch -> poll one CQ -> ...
```

## 5. Host 和公共接口

### 5.1 C ABI

保留以下已有符号及参数列表：

```cpp
TileXRMoonEpCombineV2(...);
TileXRMoonEpCombineStageV2(...);
```

新增融合入口，名称可在实现时按现有版本规范最终确定，建议：

```cpp
int TileXRMoonEpCombineStageV2Fused(
    void *registeredWorkspace,
    uint64_t registeredWorkspaceBytes,
    const int32_t *dstLocal,
    TileXRCommPtr udmaComm,
    TileXRCommPtr weightMemoryComm,
    int64_t bs,
    int64_t h,
    int64_t topK,
    int64_t nvS,
    uint32_t aivCoreNum,
    const void *hiddenNvsh,
    void *hiddenSh,
    const float *routeWeightsNvs,
    float *routeWeightsSk,
    uint32_t dtype,
    aclrtStream stream);
```

约束：

- 两个 weight 指针必须同时为空或同时非空；
- weight 非空时 `weightMemoryComm` 必须非空；
- weight 为空时允许 `weightMemoryComm == nullptr`，Kernel 完全跳过 weight 路径；
- `udmaComm` 和 `weightMemoryComm` 必须具有相同的 global/local rank 映射；
- `weightMemoryComm` 必须与 `udmaComm` 是不同实例和不同 domain；
- Host 继续校验 `bs * topK <= nvS`、BF16 hidden、core 数和 registered workspace；
- Host 不再把 weight 拷进 registered workspace，也不再启动 FP32 `h = 1` 的第二次
  Combine V2 Kernel。

旧入口可以继续保持当前两次 launch 行为，作为 ABI 兼容路径。MoonEP Torch Runtime 在
完成双 communicator 初始化后切换到融合入口。

### 5.2 Python Runtime

Runtime 增加内部成员：

```text
_comm                 Shared-QP UDMA communicator, domain 0
_weight_memory_comm   Memory-only communicator, domain 1
```

初始化和销毁必须遵循：

- 所有 rank 以相同顺序创建两个 communicator；
- domain ID 在所有 rank 上一致；
- 任一 communicator 初始化失败时，销毁已经成功创建的实例；
- Runtime shutdown 时分别销毁两个 communicator；
- Python 用户可见的 `comm_ptr`、plan、方法签名和返回值保持兼容；
- 是否启用融合路径由 Runtime 内部能力检查决定，不要求调用方新增参数。

## 6. Memory-only Communicator

当前 `TileXRCommInitRankWithDomain()` 创建普通非 shared-QP communicator，但仍受进程级
`TILEXR_ENABLE_UDMA` 控制并执行 `InitUDMA()`。直接用它创建第二个 domain 可能重复初始化
UDMA/RA，增加资源占用并产生 ownership 冲突。

因此新增显式 Memory-only 初始化能力，例如：

```cpp
int TileXRCommInitRankMemoryDomain(
    int commDomain, int rankSize, int rank, TileXRCommPtr *comm);
```

该 communicator：

- 初始化 bootstrap、socket exchange、本地 peer memory 和远端 mapping；
- 填充 `CommArgs::peerMems[]`、rank 和 topology 字段；
- 不初始化 UDMA context、UDMA QP、UDMA registry 或 Fullmesh view；
- 默认不初始化本设计不需要的 SDMA 数据面；
- 继续使用独立 domain 隔离 peer memory 名称、socket/bootstrap 和生命周期；
- 失败时不能影响已经工作的主 UDMA communicator。

实现应使用 communicator 实例级 capability/options，而不是临时修改进程级环境变量。

## 7. Weight Memory Window 布局

Combine V2 weight 区域位于每个 Memory communicator peer window 的数据区：

```text
peerBase(peer) = memoryArgs->peerMems[peer] + TileXR::IPC_DATA_OFFSET
```

固化布局如下，所有 offset 相对 `peerBase`：

```text
weightRecordEpochBytes = AlignUp(nvS * 16, 64)
weightRecordBytes = 2 * weightRecordEpochBytes
weightRecordOffset = 0
weightDoneOffset = weightRecordBytes
weightDoneEpochBytes = rankSize * 64
weightDoneBytes = 2 * weightDoneEpochBytes

totalWeightBytes = AlignUp(weightDoneOffset + weightDoneBytes, 64)
```

逻辑结构：

```text
weightRecord[2][nvS]           16-byte generation-tagged record
weightDone[2][rankSize]        64-byte token slot
```

每个 weight record 的 ABI 为：

```cpp
struct alignas(16) MoonEpCombineV2WeightRecord {
    float value;
    uint32_t reserved;
    uint64_t magic;
};
```

Host 必须以 checked arithmetic 计算布局，并验证：

- `totalWeightBytes <= IPC_BUFF_MAX_SIZE`；
- 所有 offset 和乘法无溢出；
- record 区与 done 区不重叠；
- weight 区不进入 `IPC_DATA_OFFSET` 之前的 flag 保留区；
- `peerMems[rank]` 以及本轮需要的所有 `peerMems[peer]` 均非空。

本设计不把 weight 区加入 registered UDMA workspace，因此现有
`TileXRMoonEpCombineGetWorkspaceSizeV2()` 的 hidden workspace 大小不因 weight 增长。

### 7.1 Generation 隔离

Credit 迁移后，step 0 前不存在可用于“接收端清零完成”的跨 rank 初始化屏障，因此不清理
weight receive 区。仅使用一份 generation-tagged record 仍不安全：快 rank 完成 generation N
后可先进入 N+1，并覆盖慢 rank 尚未复制的 N 数据；完整 magic 只能发现覆盖，不能防止合法
数据丢失。

record 和 done 区因此都按 `epoch = magic & 1` 分为两份。发送端把 `value`、`reserved=0` 和
当前完整 `magic` 作为一个 16 Byte record 通过同一次 MTE3 写入；接收端只在
`record.magic == current magic` 时输出 `record.value`，否则输出零。两份 epoch 足够，因为
rank 若要从 N+1 继续进入 N+2，必须先收到所有 source 的 N+1 Done；仍在复制 N 的慢 rank
尚未进入 N+1，不可能发布该 Done。因此任一 rank 最多领先一代，不会在慢 rank 消费 N 时
复用 N 的物理 epoch。该方案保证：

- `dstLocal == -1` 或本轮未覆盖的 target 不会暴露旧 weight；
- 旧 record 即使物理保留，也会因 magic 不匹配而失效；
- 相邻 generation 写入不同物理 epoch，不会覆盖仍在消费的数据；
- step 0 不依赖额外 receiver-ready token 或初始化屏障；
- `weightDone` 仍用于证明本 source 本 step 的所有 record 写入已经完成。

## 8. Route 映射和唯一所有者

weight 与 hidden 共享 `dstLocal[source]`：

```text
encoded = dstLocal[source]

encoded == -1:
    skip

peer   = encoded / nvS
target = encoded - peer * nvS
```

对当前 step 的 `peer`，复用：

```text
LoadSelectionChunk()
SelectPeerIndices(peer)
selectedIndexBuf
dstSlotBuf
```

每个被选择的 source 执行：

```text
routeWeightsNvs[source]
    -> UB transfer buffer
    -> {value, reserved=0, magic}
    -> peerBase(peer) + weightRecordOffset
       + epoch * weightRecordEpochBytes + target * 16
```

必须保持以下唯一所有者约束：

- 一个有效 source 在一次 Combine 调用中只由一个 sender core 发送；
- 同一 `(destination rank, target)` 本轮只有一个有效 writer；
- 发布 `weightDone[destination][source rank]` 的 core 是该 source/destination pair 的唯一
  schedule owner；
- self、Fullmesh 和 CLOS 只改变 hidden path，不改变 weight 的 owner 推导。

Host schedule 单测需要穷举所有支持的 rank size，证明每个有效 pair 的 sender core、step 和
receiver 期望完全一致。

## 9. Kernel ABI 与状态

`CombineV2Params` 和 launch context 增加：

```text
weightMemoryComm
weightMemoryHostArgs
weightMemoryDevArgs
routeWeightsNvs
routeWeightsSk
weight layout offsets/bytes
hasRouteWeight
```

Kernel 参数在现有参数之后追加，保持原有参数的顺序和含义：

```text
memoryCommArgs
routeWeightsNvs
routeWeightsSk
weightRecordOffset
weightDoneOffset
weightWindowBytes
weightOutputElements
hasRouteWeight
```

Host `CombineV2KernelArgs`、静态大小断言、注册 Kernel signature 和 AICore 入口必须同步
修改。现有 hidden-only入口也按新 Kernel ABI 填充尾部字段，weight 关闭时传空指针和
`hasRouteWeight = 0`。

Kernel 新增主要状态：

```text
weightMemoryArgs_
localWeightWindow_
weightRecordOffset_
weightDoneOffset_
weightWindowBytes_
routeWeightsNvsAddr_
routeWeightsSkAddr_
hasRouteWeight_
```

Generic 和 Group Kernel 如同时保留，必须消费同一 Host ABI 和同一 token/layout 定义；不能
只修改入口而让某一 rank-size 分支忽略 weight。

## 10. Step 内执行协议

### 10.1 CLOS remote step

```text
SendRemoteStep(peer, step)
  -> hidden payload WQEs
  -> ordered hidden Done WQE
  -> CopyIssueToSq
  -> st_dev doorbell

SendWeightMemoryStep(peer, step)
  -> rescan selection chunks
  -> route weight GM -> UB -> remote Memory GM
  -> wait all weight MTE3 complete
  -> PublishWeightDone(peer, step)

WaitStepCqs(step)
Wait existing Grant/Credit/admission state
```

### 10.2 Fullmesh step

Fullmesh hidden 已有独立 QP/CQ。顺序调整为：

```text
SendFullmeshStep(peer, step)
SendWeightMemoryStep(peer, step)
PublishWeightDone(peer, step)
WaitFullmeshCq(step, peer)
existing control handling
```

weight 始终通过 Memory communicator，不因为 hidden peer 在同 server 而切换到 UDMA
Fullmesh。

### 10.3 Self step

self hidden 保留本地 scratch copy。self weight 仍使用：

```text
memoryArgs->peerMems[rank] + IPC_DATA_OFFSET
```

执行本地 Memory-window MTE copy并发布本地 `weightDone`，从而保持与 remote 完全一致的
generation、输出和完成协议。self 不构造 UDMA WQE，也不等待 hidden CQ。

### 10.4 空 payload

即使某个 peer 在当前 step 没有被选择的 hidden/weight row：

- hidden 保持当前空 payload 仍发送 ordered Done 的规则；
- weight 不执行 payload MTE3，但仍发布 `weightDone(magic, step)`；
- 接收端因此可以区分“本轮没有 weight”与“发送端未完成”。

## 11. Weight MTE 数据搬运

单个 route weight 输入/输出是 4 Byte，Memory window 中的传输记录是 16 Byte，目标位置
通常不连续。第一版采用 GM -> UB -> GM `DataCopyPad`：

```text
MTE2 load 4-byte value into UB
fill reserved=0 and magic in the same UB record
MTE3 write exactly one 16-byte record to remote target
```

约束：

- 不使用 scalar store 写远端 peer memory；
- 不把 16 Byte record 写扩大成无保护的 32 Byte read-modify-write，避免相邻 target writer
  竞态；
- 发布 token 前必须确认所有 weight payload MTE3 已完成；
- 4 Byte GM->UB、16 Byte UB->GM 和 4 Byte output copy 的 `DataCopyPad` overload、padding、
  UB 初始化和 event 顺序必须按 CANN 9.1 编译及实机行为验证，不能只依赖 Host 编译成功；
- profile 证明逐项 MTE 成为瓶颈后，可合并连续 target run，但不得改变完成协议。

## 12. Weight Done 完成协议

现有 hidden final batch 已经追加 ordered Done WQE。它只能证明 UDMA hidden 路径完成，不能
证明独立 Memory 路径完成，因为两个通信域之间不存在天然顺序。

weight token slot：

```text
index  = sourceRank
offset = weightDoneOffset + epoch * weightDoneEpochBytes + index * 64
value  = MoonEpCombineV2Token(magic, step)
```

发送端：

```text
all weight payload MTE3 submitted
-> MTE3 completion/barrier
-> UB token -> MTE3 -> remote weightDone slot
-> wait token MTE3 complete before reusing UB
```

接收端根据现有 schedule 计算每个 source 的 expected receive step，并同时等待：

```text
hiddenDone[epoch][source][required lane]
weightDone[source]
```

只有两类条件全部满足，才能进入 Reduce/output 阶段。推荐将其实现为统一的
`WaitInboundReady()` 公平轮询，分别保留 hidden Done 与 weight Done 的 pending mask、观察
值、timeout detail 和 profile metric。

weight 关闭时，weight pending mask 初始为空，行为退化为现有 hidden Done 等待。

## 13. 输出处理

hidden 输出保持现有流程：

```text
WaitInboundReady()
-> ReduceHidden()
-> registered workspace output
-> Host D2D copy to hiddenSh
```

weight 不执行 Reduce。接收端在 `weightDone` 全部满足后，逐项读取本地 record 区的：

```text
weightRecordOffset + epoch * weightRecordEpochBytes + [0, bs * topK) * 16
```

若 record magic 等于当前 magic，则复制 value 到 `routeWeightsSk`；否则写零。该长度与当前
Stage V2 从 active scratch 拷出的长度一致。

`CopyReceivedWeights()` 由 active cores 按元素区间分片，完成后参加现有卡内最终汇合。
Kernel 返回前必须保证对 `routeWeightsSk` 的 MTE3 已提交完成，使同一 stream 的后续任务能
按照正常 stream 依赖消费输出。

## 14. 失败、超时和可观测性

Host 在 launch 前拒绝：

- 两个 communicator rank/world/local 映射不一致；
- weight 启用但 Memory communicator 或 `peerMems[]` 不完整；
- weight layout 超出 peer memory data capacity；
- weight 指针只提供一个；
- `bs * topK > nvS` 或 size arithmetic 溢出；
- Memory-only capability 未启用。

Kernel 增加可区分的失败信息：

```text
WEIGHT_MEMORY_INVALID_CONFIG
WEIGHT_DONE_TIMEOUT
```

沿用 operation timeout 上限，但失败 detail 至少记录：

```text
step, peer/source, core, epoch, expected token, observed token
```

Profile 增加：

```text
weight selection/rescan cycles
weight Memory payload cycles
weightDone publish point
weight inbound wait cycles
weight output copy cycles
```

现有 hidden WQE/CQ metric 必须保持可比，以验证 weight 没有进入 UDMA SQ/CQ。

## 15. 兼容与迁移

迁移按以下顺序进行：

1. 增加 Memory-only communicator API，并独立验证 `peerMems[]`；
2. 增加 weight window layout 和纯 Host 单测；
3. 扩展 Combine V2 内部 params、launch ABI 和 Kernel 入口，weight 默认关闭；
4. 实现 weight Memory copy、`weightDone` 和接收等待；
5. 增加融合 C 入口，保留旧入口；
6. Runtime 创建第二 communicator 并切换融合入口；
7. 确认融合路径正确后，旧入口继续作为 ABI 兼容实现，不作为 Runtime 默认路径。

不允许在融合路径初始化失败时无提示地退回两次 launch。调用方必须收到明确 capability 或
初始化错误；如确实需要兼容回退，应由显式配置控制并记录日志。

## 16. 预计修改范围

| 模块 | 主要职责 |
| --- | --- |
| `src/include/tilexr_api.h` | 声明 Memory-only communicator 初始化接口 |
| `src/comm/comm_wrap.cpp` | 暴露新初始化入口和失败清理 |
| `src/comm/tilexr_comm.h/.cpp` | 增加实例级 capability，跳过第二 domain 的 UDMA/SDMA |
| `src/include/tilexr_moonep_combine_v2.h` | 增加融合 Stage C API，保留旧 ABI |
| `src/moonep/combine_v2/host/combine_v2_host.*` | 双 comm 校验、layout 和 launch context |
| `src/moonep/combine_v2/host/combine_v2_launch.*` | 扩展 registered Kernel 参数 ABI |
| `src/moonep/combine_v2/host/combine_v2_layout.*` | 增加 weight Memory window layout builder |
| `src/moonep/combine_v2/host/tilexr_moonep_combine_v2.cpp` | 融合入口及删除默认路径的第二次 weight launch |
| `src/moonep/combine_v2/kernels/*` | step weight send、token、wait、output 和 profile |
| `integrations/moonep_torch/tilexr_moonep/runtime.py` | 第二 communicator 生命周期及融合入口调用 |
| `tests/`、`tools/moonep/` | Host contract、schedule、Kernel 和硬件端到端验证 |

实施时必须保留工作区中与本设计无关的未提交修改。Generic/Group Kernel 的修改应按当前
实际启用分支展开，不能覆盖其他并行开发中的 128-rank 分支选择。

## 17. 验证策略

### 17.1 Host 和纯逻辑测试

- weight layout 的正常、边界、overflow 和 capacity case；
- 两个 communicator rank/local-rank 映射不一致时拒绝 launch；
- weight 指针同时为空、同时非空和单边为空；
- 所有支持 rank size 的 pair -> step -> core 唯一性穷举；
- token index、magic 和 expected receive step 一致；
- Kernel argument struct、signature 和参数顺序静态检查；
- Python API 参数和返回 tuple 与 `tools/moonep/test_npu_e2e.py` 保持兼容。

### 17.2 Kernel 编译与局部验证

- CANN 9.1 target-toolchain 编译 Generic 及实际启用的 Group variant；
- 检查 4 Byte GM -> UB -> remote GM MTE copy 的生成和 tail 行为；
- 验证 weight 关闭时不访问 Memory comm 和 weight 指针；
- 验证空 payload 仍发布 weight Done；
- 连续多轮调用覆盖多个 magic generation，检测残留值。

### 17.3 实机正确性

在匹配设备和拓扑上，先运行对应模式的官方 HCCL Test 作为环境基线，再验证：

- 无 route weight：结果与当前纯 hidden V2 一致；
- 有 route weight：hidden 和 weight 同时与 PyTorch/CPU reference 一致；
- `dstLocal == -1`、非连续 target、空 peer 和 self peer；
- CLOS、Fullmesh 和跨节点 Memory peer mapping；
- 连续压力调用、generation 复用和 timeout/error 注入；
- Runtime trace 证明一次 combine 调用只有一次 Combine V2 Kernel launch；
- UDMA WQE/CQ 计数与纯 hidden 路径一致，weight 没有进入 UDMA 队列。

Host、模拟器或 910B fallback 结果不能证明跨节点 Memory 数据面。跨节点可达性和 MTE
顺序必须在目标 A5/Ascend950 环境上单独给出证据。

### 17.4 性能验收

至少比较相同代码基线上的：

```text
旧路径：hidden launch + weight launch
新路径：single fused launch, hidden UDMA + weight Memory
```

记录：

- 端到端 combine latency；
- Kernel launch 数和 Host enqueue 时间；
- 每 step hidden send、weight send、CQ wait、inbound wait；
- UDMA SQ/CQ 数量；
- Memory weight 带宽及逐项 4 Byte MTE 的成本。

如果 single launch 正确但性能下降，先根据 profile 判断是 selection rescan、4 Byte MTE 还是
CQ polling 饥饿，再决定连续 target 合并或 CQ/weight progress loop，不改变 token 协议。

## 18. 验收条件

设计实现完成需要同时满足：

- route weight 存在时，Runtime trace 中每次 Combine 只有一次 V2 Kernel launch；
- hidden 和 weight 使用同一 `magic`、step schedule；hidden 控制区继续使用 epoch，weight
  record/done 使用双 epoch 物理区，record 仍校验完整 magic；
- weight 只访问 Memory communicator 的 `peerMems[]`，不新增 UDMA WQE/CQ；
- hidden Done 早到时接收端不会提前读取 weight；
- weight Done 早到时接收端仍等待 hidden Done；
- self、空 payload、连续 generation 复用和 `dstLocal == -1` 输出正确；
- 旧公共 C ABI 和 MoonEP Python API 契约保持兼容；
- Memory-only communicator 不重复初始化 UDMA/RA；
- 目标硬件上的正确性和性能证据已记录，验证声明不超出实际硬件范围。

## 19. 已确认实施决策与边界

1. Credit 协议不提供 step 0 清零屏障；weight record/done 采用双 epoch，record 使用完整
   magic 过滤同 epoch 的旧数据，接收区不清零；
2. CANN 9.1 Kernel 使用 4 Byte GM->UB、16 Byte UB->GM record、64 Byte done token 和
   4 Byte output 的 `DataCopyPad`，每个跨流水线依赖显式同步；
3. 8/32/64P 走 Generic Kernel；Group/128P 实现保持同一 ABI 和 weight 协议，但是否启用由
   128P 分支自身的集成状态决定；
4. 实机验证必须先通过匹配拓扑的 HCCL `all_reduce_test -a aiv_only`，再运行 fused-weight
   probe；
5. Host/编译验证不能替代跨节点 Memory mapping、MTE 顺序和连续 generation 的实机证据。

## 20. 2026-08-20 阶段验证证据

目标环境为 2 号柜 Ascend950DT、CANN 9.1.0。验证使用 `bs=128`、`K=16`、`H=3584`、
1 次 warmup 和 3 次计时 launch：

- pure AICore Kernel、Host 库和硬件 probe 通过目标工具链编译；
- 29/29 Host、layout、schedule、public ABI 和 source-guard 测试通过；
- 8/32/64P 分别先通过 HCCL `all_reduce_test -a aiv_only`；
- 双 epoch 修复后的 8/32/64P fused probe 分别有 8/32/64 个 rank 报告
  `correctness=passed` 和 `weight_correctness=passed`；
- probe 使用不同的 generation 0/1 weight 输入，分别校验首轮输出和连续
  `1 warmup + 3 timed` 序列的末轮输出；该证据不表示每个中间 generation 都被单独读回。

远端日志：

```text
/home/h00580772/tilexr_validation/cab2_fused_weight/logs/hccl_aiv_8p_20260820_015904.log
/home/h00580772/tilexr_validation/cab2_fused_weight/logs/hccl_aiv_32p_20260820_015949.log
/home/h00580772/tilexr_validation/cab2_fused_weight/logs/hccl_aiv_64p_20260820_020046.log
/home/h00580772/tilexr_validation/cab2_fused_weight/logs/combine_fused_weight_8p_20260820_015924.log
/home/h00580772/tilexr_validation/cab2_fused_weight/logs/combine_fused_weight_32p_20260820_020011.log
/home/h00580772/tilexr_validation/cab2_fused_weight/logs/combine_fused_weight_64p_20260820_020110.log
```

本轮证明了 Generic Kernel 在单机 Fullmesh、4 节点和 8 节点路径上的 hidden/weight
正确性及跨节点 Memory 可达性。它没有覆盖 Group/128P、`dstLocal == -1`、错误注入、
Runtime trace、旧双 launch 性能基线或 8P 大 per-peer Fullmesh CQ 回收边界，不能对这些
维度作通过声明。
