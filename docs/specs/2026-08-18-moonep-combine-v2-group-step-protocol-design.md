# MoonEP Combine V2 Group Step 完成协议设计

## 1. 范围

本设计仅适用于 128P `MoonEpCombineV2Group` 路径。非 128P 仍使用
`MoonEpCombineV2`，不改变其协议和 ABI。

设计前提：

- 外部输入和 128P 发送映射可靠。
- Combine 多轮串行执行，UDMA QP 在轮次间复用，不存在同一 QP
  的并行 kernel 使用。
- 每个 step、每个 QP 的 payload WQE、Done WQE 与承接的无 CQ
  Credit WQE 总数小于 16384 个 SQ basic block。
- 各 core 独立执行点对点握手，不在 step 热路径增加跨 core 强同步。

## 2. 问题与目标

当前 Group CLOS 路径先将每个 QP 的最后一条 payload WQE 设为
`ORDERED_COMPLETION`，等待两个 payload CQE 后，再单独发送一个 Done
并等待 Done CQE。之后才会等待对端 Done，再发送并等待 Credit
CQE。这使本端和对端完成条件被强制串行化。

新协议的目标是：

1. payload 只使用 NO WQE，不产生 CQE。
2. Done WQE 跟在本 QP 的 payload 之后，使用
   `ORDERED_COMPLETION`，用一个 CQE 证明该 QP 的 payload 和 Done 均已完成。
3. 本端 Done CQE 和对端 Done 作为两个独立完成条件，公平轮询，
   不固定先后顺序。
4. 对端 Done 全部到达后立即给下一 step 的预期发送端发送一个
   Credit；Credit 不产生、不等待 CQE。
5. 当前 step 必须同时满足本端 Done 完成和对端 Done 到达，才能进入
   下一 step。

## 3. 三类传输完成规则

| 传输类型 | payload | 本端完成条件 | 对端完成条件 |
| --- | --- | --- | --- |
| CLOS | 两个 QP，payload WQE 均为 NO | 两个 Done(SO) CQE | 两个 QP 对应的 Done token |
| Fullmesh | 一个 QP，payload WQE 均为 NO | 一个 Done(SO) CQE | 一个 Done token |
| Self | 本地 GM 拷贝 | 本地拷贝返回 | 无 Done 等待 |

CLOS 的两个 QP 都必须下发 Done。即使某个 QP 没有 payload，也要在
该 QP 下发一条 Done(SO)，使接收端可以使用固定的双 Done 规则。

## 4. 发送阶段

### 4.1 CLOS

每个 lane 的 SQ 顺序为：

```text
QP0: carried Credit(NO) -> payload(NO)... -> Done0(ORDERED_COMPLETION)
QP1:                       payload(NO)... -> Done1(ORDERED_COMPLETION)
```

发送阶段完成以下工作后立即返回：

1. 扫描本 step 的 route，构建并分批下发两个 QP 的 payload(NO) WQE。
2. 为两个 QP 各追加一条 Done(SO) WQE。
3. 先完成 UB 到 SQ 的 MTE3 发布，再更新 head、completion count 并敲
   doorbell。
4. 为每个 lane 保存当前 step 的 `doneSubmittedHead` 和 `doneCqTarget`。

发送阶段不等待 payload CQE 或 Done CQE。

### 4.2 Fullmesh

Fullmesh 只有一个 QP。最后一批 payload 之后追加一条 Done(SO)，保存
对应的 `doneSubmittedHead` 和 `doneCqTarget`，发布后立即返回。

### 4.3 Self

Self 保留本地拷贝逻辑，不创建 Done WQE，不等待 Done token。

## 5. Step 完成状态机

每个 core 维护独立状态：

```text
localDonePendingMask   本端尚未收到 CQE 的 Done lane
remoteDonePendingMask  尚未观察到对端 Done token 的 lane
creditPublished        是否已经发布下一 step Credit
pollCursor             本端 CQ 和对端 Done 的公平轮询位置
```

伪代码：

```cpp
while (localDonePendingMask != 0U || remoteDonePendingMask != 0U) {
    PollOneLocalDoneCq();
    PollOneRemoteDoneToken();

    if (remoteDonePendingMask == 0U &&
        !creditPublished && step + 1U < stepCount) {
        PublishNextCreditNoCompletion(step);
        creditPublished = true;
    }

    CheckTimeoutAndFailure();
    AdvancePollCursor();
}
```

轮询助手必须是单次尝试、立即返回的非阻塞函数，不能在某一个 lane
内部循环到完成，否则仍会将两个条件串行化。

对端 Done 先全部到达时，Credit 应立即发布，无需等待本端 Done
CQE。发布 Credit 后继续轮询本端 Done CQE，直到两类条件都完成。

## 6. Credit 无 CQ 协议

### 6.1 发布

- 每个 core 在每个非最后 step 只发布一个 Credit。
- Credit 目标是下一 step 中将向本 rank/core 发送数据的预期发送端。
- 本地 Credit 直接写入本 rank GM signal。
- 远端 Credit 在 CLOS six-port SQ 上使用不带 completion bit 的 WQE。
- Credit 发布只增加 SQ `head`，不增加 completion count，不更新
  `cqTarget`，不轮询 CQ。

### 6.2 累积回收

Credit WQE 不需要独立 CQE。同一 CLOS SQ 后续最近的 Done(SO) CQE
通过 `entryIdx` 一次将 SQ tail 推进到 Done 之后，同时回收之前的
Credit(NO) 和 payload(NO) WQE。

若当前调用在最后一个 Credit 之后只执行 Fullmesh/Self step，则允许这些
Credit 以少量 outstanding 状态保留在 CLOS SQ。下一次串行调用的 Step 0
固定为 CLOS，其 Done(SO) CQE 会连同这些承接 Credit 一起回收。

因此不增加轮末 drain CQE，也不为 Credit 单独生成 CQE。

### 6.3 状态约束

- `InitLaneStates()` 必须允许有界的承接 Credit outstanding，不能强制
  `head == tail`。
- SQ 可用空间必须以 `head - tail` 计算，并将承接 Credit 计入上限。
- Credit source 继续按 epoch、transition step 和 core 分槽，不在 WQE
  可见前覆写。
- 本设计依赖串行调用和下一轮 Step 0 CLOS Done 的最终回收。

## 7. CQ 状态记账

当 Credit 可能在当前 step Done CQE 之前被追加到 SQ 时，当前的
`WaitLaneCq()` 不再适用，因为它强制 `state.head == state.tail`。

每个 Done 必须保存提交时快照：

```text
doneSubmittedHead[lane]  Done WQE 之后的 SQ 绝对 head
doneCqTarget[lane]       该 Done 对应的 CQ 绝对 target
```

消费 Done CQE 时验证 `tail == doneSubmittedHead[lane]`。当前 `head` 可以更大，
其差值是 Done 之后追加的无 CQ Credit。后续 Done CQE 可以通过更新的
`entryIdx` 累积推进 tail。

## 8. Done token 布局

- CLOS Done 继续使用 `epoch + source rank + lane` 索引，分别写入
  six-port 和 two-port 的两个 slot。
- Fullmesh 只使用约定的单个 Done slot。
- Done token 保留 `magic + step` 校验和 DCCI，不通过清零复用 slot。
- CLOS 接收端只有在两个 lane 的 token 都匹配时才将
  `remoteDonePendingMask` 置零。

## 9. 最后一个 Step

最后一个 step 与其他 step 使用相同的 payload/Done 发布和双条件轮询逻辑：

- 必须完成本端 Done CQE。
- 必须完成对端 Done token 等待。
- 不计算下一发送端，不发布 Credit，不等待 Credit。

## 10. Profiling 语义

新协议下，`Step N send` 应在 payload 和 Done WQE 都已写入 SQ 并敲完
doorbell 后结束，不包含 Done CQE 等待。

由于本端 Done 和对端 Done 没有固定先后关系，应记录独立时间点，不将
它们伪装成两个串行 stage：

```text
stepLocalDoneEnd[N]   本端所有 Done CQE 完成
stepRemoteDoneEnd[N]  对端所有 Done token 到达
stepCreditPostEnd[N]  Credit WQE 已发布（最后 step 不记录）
stepReadyEnd[N]       本端和对端 Done 条件都完成
```

若扩展 `MoonEpCombineV2ProfileRecord` 布局，必须升级 profile version，并同步更新
Host layout、hardware probe、trace 解析和 source guard。

## 11. 实现边界

预期调整的 Group kernel 责任如下：

- `SubmitRemotePayloadBatch()`：payload 保持 NO，不再将 terminal payload 改为 SO。
- `SendClosStep()`：两个 QP 追加 Done(SO)，保存快照，发布后返回。
- `SendFullmeshStep()`：单 QP Done(SO) 发布后返回。
- `WaitSingleInboundDone()`：拆为按传输类型和 lane 单次轮询的非阻塞助手。
- `WaitLaneCq()`：热路径不再阻塞等待至 `head == tail`，改为按 step
  Done 快照推进。
- `PublishNextCredit()`：远端分支改为无 completion 提交，不复用当前
  同步 `SubmitClosControl()`。
- `Process()`：每个 step 使用双条件公平轮询状态机。

Host API、kernel launch ABI、workspace 大小和 128P 发送映射保持不变。

## 12. 验收标准

1. CLOS 每个 step 的 payload WQE 均不请求 CQE，两个 Done WQE 均为
   `ORDERED_COMPLETION`。
2. Fullmesh 只等待一个本端 Done CQE 和一个对端 Done token。
3. Self 不生成或等待 Done。
4. 本端 Done CQE 和对端 Done 的轮询无阻塞偏置，任一条件先完成都可以
   先更新状态。
5. 对端 Done 全部到达后，非最后 step 立即发布一个无 CQ Credit。
6. 最后 step 完成本端和对端 Done，不发布 Credit。
7. 后续 Done CQE 能累积回收先前的无 CQ Credit，串行多轮调用不因
   承接 outstanding 被判定为 QP 非空错误。
8. 非 128P 路径行为不变。
