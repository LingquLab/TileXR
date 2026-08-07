# MoonEP Planner V2 下游契约

## 1. 目的与适用范围

本文定义当前 TileXR MoonEP Planner V2 与 **Expert Migration、Token Dispatch、Token Combine** 之间的下游数据契约。本文描述的是当前 public headers 和 `src/moonep/planner_v2` 实现已经具备的语义，不把上游梳理文档中的设想误写成已实现能力。

核心结论如下：

- Planner 只生成布局和迁移意图元数据，不搬运专家权重，也不搬运 token hidden/route weight。
- Planner 只汇聚每个 Rank 的 `tokensPerExpert[E]`、调用头和 Rank ID；**没有 route all-gather，也没有 `[R*S*K]` global token pool**。
- `dst[S*K]` 是当前源 Rank 的逐 route 发送计划；`cuSeqlens[E+B]` 是当前目的 Rank 的接收/专家布局。
- V2 metadata 新增完整的目的视角 `remoteExperts[R,B]` 和当前 owner Rank 视角的 `expertTargets[E/R,ceil(R/64)]`。
- Planner 不输出 `tokenRemap`。`tokenRemap` 必须由 Dispatch 在实际落槽时写入，Combine 只消费 Dispatch 已提交的映射。
- 当前 MoonEP stage ABI 中只有 Planning 是 native；Dispatch、Prefetch Weight、Combine、Reduce Grad 仍为本地 stub，不构成真实的跨 Rank 数据搬运实现。

## 2. 记号与基础约束

| 记号 | 含义 |
| --- | --- |
| `R` | communicator Rank 数，即 `world`/`rankSize` |
| `S` | 每个 Rank 的本地 token 数 |
| `K` | 每个 token 的 top-k route 数 |
| `E` | 全局专家数，要求 `E % R == 0` |
| `Epr` | 每个 owner Rank 的专家数，`Epr = E / R` |
| `B` | 每个目的 Rank 的 remote-expert/prefetch slot 数，即 `prefetchSlots` |
| `Cap` | 每个 Rank 的 route 容量，当前要求 `Cap = S * K` |
| `NvS` | `dst` 编码的每 Rank slot stride，也是目的 Rank token-remap/layout 的地址域上界；要求 `NvS >= Cap` |
| `W` | expert target bitmap 的 64-bit word 数，`W = ceil(R / 64)` |
| `rank` | 当前调用所在 Rank |

除非特别说明，二维数组均按 C row-major 展平。所有 metadata `*Count` 都是**元素个数**，不是字节数。

## 3. 总体数据流和职责边界

```text
local topk[S,K] ───────────────┐
local tokensPerExpert[E] ──┐   │
                           │   │
                    Planner V2
          汇聚 [R,E] load，生成布局/迁移意图
                           │
          ┌────────────────┼─────────────────────┐
          │                │                     │
     dst[S*K]       cuSeqlens[E+B]      remoteExperts[R,B]
          │                                      expertTargets[E/R,W]
          │                │                     │
          ▼                ▼                     ▼
       Dispatch        Expert compute      Expert Migration
          │
          ├─ 实际发送/本地复用 hidden
          ├─ 写 route weight
          └─ 写 tokenRemap[NvS] = globalTokenId

 expert outputs + tokenRemap[NvS] ───────────────► Combine
```

### 3.1 Planner 的职责

Planner：

1. 读取当前 Rank 的 `topkExperts[S,K]` 和 `tokensPerExpert[E]`。
2. 通过 Planner peer mailbox 发布并汇聚：
   - `PlanCallHeader`；
   - 每个 Rank 的 `tokensPerExpert[E]`；
   - 每个 Rank 的 `globalRankId`；
   - Planner status。
3. 基于全局负载计数 `[R,E]` 计算专家-token segment 的目标 Rank、slot 基址和远端专家需求。
4. 为当前 Rank 的本地 routes 生成 `dst[S*K]`。
5. 为当前 Rank 作为目的 Rank时生成 `cuSeqlens[E+B]`。
6. 在 metadata V2 路径生成完整的 `remoteExperts[R,B]`，并生成当前 owner Rank 的 `expertTargets[E/R,W]`。

Planner **不负责**：

- 汇聚所有 Rank 的 `topkExperts` 或 route 列表；
- 构造 `[R*S*K]` global token pool；
- 读取或发送 token hidden；
- 读取或发送专家权重；
- 填写 `tokenRemap`；
- 执行专家计算或 Combine。

### 3.2 Expert Migration 的职责

Expert Migration 消费 Planner 的专家迁移意图：

- 目的视角可使用 `remoteExperts[dstRank,B]` 判断目的 Rank 的每个 remote slot 需要哪个全局专家；
- owner/source 视角可使用当前 Rank 的 `expertTargets[E/R,W]` 判断每个本地专家需要复制到哪些目的 Rank；
- 负责实际权重地址解析、传输、同步、错误处理和生命周期管理；
- 必须在依赖这些 remote expert 的专家计算开始前完成相应权重就绪。

Planner 只给出“需要复制到哪里”，不提供权重 tensor、字节数、传输句柄，也不执行真实复制。

### 3.3 Dispatch 的职责

Dispatch 消费本地 `tokens[S,H]`、本地 route 顺序、`dst[S*K]` 和目的布局：

- 解码每条 route 的目的 Rank 和 `recvSlot`；
- 对 primary route 发送 hidden，对同一 token、同一目的 Rank 的后续 route 复用 hidden；
- 无论 primary 还是 deduplicated route，都要保留各自的 route 语义并写 route weight；
- 在目的 Rank 的对应 `recvSlot` 写入该 route 的 `globalTokenId`，从而建立 `tokenRemap`；
- 对 `~rawDst` route，负责让其目的 slot 能取得 primary hidden，例如通过目的端本地 fan-out/别名/复制；`~rawDst` 只抑制重复的跨 Rank hidden 发送，不删除 route。

### 3.4 Combine 的职责

Combine 消费专家输出和 Dispatch 写好的 `tokenRemap`：

- 跳过值为 `UINT64_MAX` 的无效/未使用 slot；
- 解码有效 `globalTokenId` 得到原始 `srcRank`、`tokenId` 和 `topKId`；
- 将每个专家 route 的输出发送/累加回原始 token；
- 按模型定义处理 top-k 权重和归并精度。

Combine 不应假设 Planner 已经填写 `tokenRemap`，也不应把 `dst` 当作反向映射。

## 4. `dst[S*K]` 契约

### 4.1 索引顺序

`dst` 是 `int32_t[S*K]`，等价于 `[S,K]`：

```text
routeIndex = tokenId * K + topKId
```

每个 Rank 的 Planner 调用只输出**该 Rank 本地 token routes** 的 `dst`，不是 `[R,S,K]` 全局 route 表。

### 4.2 `rawDst` 编码

每条 route 的非负原始目的编码为：

```text
rawDst  = dstRank * NvS + recvSlot
dstRank = rawDst / NvS
recvSlot = rawDst % NvS
```

约束：

```text
0 <= dstRank < R
0 <= recvSlot < NvS
0 <= rawDst <= INT32_MAX
```

`recvSlot` 是目的 Rank 的最终 route slot；它已经包含专家 group 基址、该专家内 offset 和 Planner 的布局决定。Dispatch 不应再次按专家负载重算 slot。

### 4.3 `rawDst` 与 `~rawDst`

实际存入 `dst[routeIndex]` 的值为：

```text
encodedDst = rawDst     // primary：需要发送 hidden
encodedDst = ~rawDst    // deduplicated：不重复发送 hidden
```

解码必须使用位取反，而不是绝对值或一元负号：

```text
rawDst = encodedDst >= 0 ? encodedDst : ~encodedDst
```

因为 `~rawDst == -rawDst - 1`，例如 `rawDst == 0` 时编码为 `-1`。

primary 的判定范围是 **同一个本地 token、同一个目的 Rank**：Planner 按 `topKId = 0..K-1` 扫描；该 token 首次路由到某个 `dstRank` 时写 `rawDst`，之后再次路由到同一 `dstRank` 时写 `~rawDst`。如果同一 token 分别路由到多个目的 Rank，则每个目的 Rank 都各有一个 primary。

当前下游 helper 的固定语义是：

| `encodedDst` | `sendHidden` | `writeRouteWeight` |
| --- | ---: | ---: |
| `rawDst >= 0` | 1 | 1 |
| `~rawDst < 0` | 0 | 1 |

因此负值不是“无效 route”，也不是“跳过该 top-k 分支”。

## 5. `cuSeqlens[E+B]` 契约

`cuSeqlens` 是当前 Rank 作为**目的 Rank**时的 `int32_t[E+B]` 累计结束 offset。它不是 `[R,E+B]`，也不带额外的前导零元素。

对 group `g`：

```text
begin(g) = (g == 0) ? 0 : cuSeqlens[g - 1]
end(g)   = cuSeqlens[g]
paddedCount(g) = end(g) - begin(g)
```

空 group 的结束值与前一 group 相同。

### 5.1 group 顺序

- `g in [0,E)`：全局专家 ID 对应的固定 group。
- `g in [E,E+B)`：当前目的 Rank 的 remote-expert slot，`slot = g - E`。

对于当前目的 Rank：

- 本地/home 专家的 tokens 保持在其全局专家 ID group；
- 若全局专家 `e` 是该目的 Rank 的 remote expert，则 `e` 的 `[0,E)` 固定 group 为空，实际 tokens 放入 `E + slot`，其中 `remoteExperts[rank,slot] == e`；
- `B` 个 appended group 与 `remoteExperts` 当前 Rank 行一一对应；unused slot 对应空 group。

每个非空 group 的 token 数按 `tokenPadding` 向上对齐后累加；最终值不得超过 `NvS`。因此 `cuSeqlens` 表示的是**带 padding 的物理布局边界**，不是原始 token count 数组。

## 6. `remoteExperts[R,B]` 契约

`remoteExperts` 是 metadata V2 的 `int32_t[R*B]`，按目的 Rank 展开：

```text
remoteExperts[dstRank * B + slot]
```

语义：目的 Rank `dstRank` 的 appended remote slot `slot` 需要加载的**全局专家 ID**。

- 方向是 destination-oriented，不是 owner-oriented。
- 只列出 home Rank 不是 `dstRank`、但在 `dstRank` 上实际分配了 token 的专家。
- 每行最多 `B` 个有效专家；unused slot 填 `-1`。
- 当前实现按该目的 Rank 上的分配 token 数降序排列，数量相同时按全局专家 ID 升序排列。
- `remoteExperts[dstRank,slot]` 与该 Rank 的 `cuSeqlens[E+slot]` group 严格对应。

每个 Rank 的 metadata V2 调用都会基于相同的 `[R,E]` load 和确定性算法生成完整 `[R,B]` 矩阵；这不是通过 route all-gather 得到的。

## 7. `expertTargets[E/R,ceil(R/64)]` 契约

`expertTargets` 是 metadata V2 在**当前 owner Rank**上的 `uint64_t[Epr*W]` bitmap：

```text
Epr = E / R
W   = ceil(R / 64)
localExpert = globalExpert - rank * Epr
word = dstRank / 64
bit  = dstRank % 64
```

当且仅当当前 Rank 拥有的全局专家 `globalExpert` 出现在某个 `remoteExperts[dstRank,:]` 中时：

```text
expertTargets[localExpert * W + word] |= 1ULL << bit
```

契约要点：

- 第一维是当前 Rank 的本地专家 ordinal，不是全局专家 ID。
- bitmap 位表示**目的 Rank**。
- 该 bitmap 是 `remoteExperts[R,B]` 对当前 owner Rank 的转置/筛选视图。
- home Rank 本身不会因为本地常驻而自动置位；只有需要远端副本的目的 Rank 置位。
- `R` 不是 64 的整数倍时，最后一个 word 的高位保持 0。

## 8. `globalTokenId` 与 `tokenRemap`

### 8.1 `globalTokenId` 编码

当前实现不是可变 bit-field，而是稳定的 row-major 稠密编码：

```text
globalTokenId = ((uint64_t(srcRank) * S + tokenId) * K) + topKId
```

反解：

```text
topKId  = globalTokenId % K
tokenBase = globalTokenId / K
srcRank  = tokenBase / S
tokenId  = tokenBase % S
```

有效 ID 必须满足对应的 `srcRank/tokenId/topKId` 范围，并且不得等于保留哨兵。

`src/include/tilexr_ep_plan.h` 公开并由 `libtilexr-moonep-planner.so` 实现以下 C++ helper：

```cpp
TileXREp::Plan::EncodeMoonEPGlobalTokenId(...);
TileXREp::Plan::DecodeMoonEPGlobalTokenId(...);
TileXREp::Plan::DecodeMoonEPDst(...);
TileXREp::Plan::BuildMoonEPRouteDescriptor(...);
```

`BuildMoonEPRouteDescriptor` 一次性返回 `srcRank/tokenId/topKId/globalTokenId/dstRank/recvSlot/sendHidden/writeRouteWeight`，供 Dispatch/Combine 避免各自重复实现编码规则。

### 8.2 `UINT64_MAX` 哨兵

public header 固定定义：

```text
TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID = UINT64_MAX
```

`UINT64_MAX` 只表示 token-remap slot 无效、未写或 padding/hole；编码 helper 明确拒绝生成或解码该值。下游不得把它解释为合法的最后一个 token。

### 8.3 `tokenRemap` 由 Dispatch 填写

metadata V2 **没有** `tokenRemap` 字段，Planner 也不填任何 remap buffer。推荐的每目的 Rank 契约为：

```text
tokenRemap uint64_t[NvS]
```

Dispatch 每轮应先将其初始化为 `UINT64_MAX`。对每条实际到达目的 Rank 的 route：

```text
tokenRemap[recvSlot] = globalTokenId(srcRank, tokenId, topKId)
```

要求：

- 每个有效 route 的 `(dstRank,recvSlot)` 唯一；重复写同一 slot 是计划/Dispatch 错误。
- primary 和 `~rawDst` route 都必须写各自的 `globalTokenId`。
- `~rawDst` route 虽然不重复发送 hidden，但仍有独立 `recvSlot`、route weight 和 remap 项。
- `NvS > S*K`、partial plan、padding 或未使用 slot 可以继续保持 `UINT64_MAX`。
- 上游梳理文档把 `glb_token_remap` 写成 `[S*K]`；这只在兼容配置 `NvS == S*K` 时等价。通用 V2 契约应按 `[NvS]` 分配。

## 9. 无 route all-gather 的含义

Planner mailbox 当前发布的 route 相关输入只有每 Rank 的 `tokensPerExpert[E]` 计数；`topkExperts[S,K]` 始终是本地输入，没有被发布到其他 Rank。

因此当前实现中不存在：

- `topkExperts[R,S,K]` all-gather；
- `globalTokenId -> expertId` 的全局 route 表；
- `[R*S*K]` global token pool；
- Planner 侧按全局 token ID 选择具体远端 token；
- Planner 侧生成完整的全局 `token_trans[N,3]` 或目的 Rank `tokenRemap`。

Planner 能做的是依据全局 **count** 决定每个专家从哪个源 Rank 切出多少连续 ordinal segment、分配到哪个目的 Rank；每个源 Rank 再用自己的本地 top-k 顺序把本地 route ordinal 映射到这些 segment，生成本地 `dst[S*K]`。

这也是 `tokenRemap` 必须留给 Dispatch 的原因：只有 Dispatch 在处理实际源 route 时同时拥有 `srcRank/tokenId/topKId` 和最终 `dstRank/recvSlot`。

## 10. 无真实搬运的当前实现边界

必须区分两种“传输”：

1. Planner 内部确实通过 IPC peer mailbox/MTE 交换调用头、load 计数、Rank ID、status 和 barrier；这是**控制面 metadata 交换**。
2. 当前代码没有实现基于该 Plan 的专家权重迁移或 route-aware token 数据面搬运。

`TileXRMoonEpGetCapabilitiesV1` 当前报告：

- native：Planning；
- stub：Dispatch、Prefetch Weight、Combine、Reduce Grad。

这些 stub 只做参数/shape 校验，并在本设备上执行 zero + prefix D2D copy（或同指针 no-op）；它们不解析 `dst`，不使用 `remoteExperts/expertTargets`，不填写 `tokenRemap`，不执行跨 Rank token/权重传输，也不实现真实 Combine 归并。因此现有 stub 成功不能作为下游数据面已打通的证据。

## 11. V1/V2 ABI 关系

“V1/V2”在当前代码中有多层命名，必须避免混用。

### 11.1 MoonEP stage ABI V1

`src/include/tilexr_moonep.h` 定义：

```text
TILEXR_MOONEP_ABI_VERSION_V1 = 1
TileXRMoonEpPlanV1
TileXRMoonEpPlanningArgsV1
TileXRMoonEpDispatchArgsV1
TileXRMoonEpPrefetchWeightArgsV1
TileXRMoonEpCombineArgsV1
```

该 ABI 是面向完整 MoonEP stage 的 C ABI。64 位平台上的冻结布局为：

```text
sizeof(TileXRMoonEpPlanV1) == 104
alignof(TileXRMoonEpPlanV1) == 8
```

`TileXRMoonEpPlanV1` 只有裸指针，没有 element count，也没有独立的 `remoteExperts`、`expertTargets` 或 `tokenRemap` 字段。兼容层固定派生：

```text
B = E / R
NvS = S * K
tokenPadding = 1
```

当前只有 V1 Planning 路径调用 native Planner，其他 stage 是 stub。V1 workspace query 的完整声明为：

```cpp
int TileXRMoonEpPlanningGetWorkspaceSizeV1(
    TileXRCommPtr comm, int64_t s, int64_t k, int64_t e,
    uint64_t *workspaceBytes, int64_t *dispatchedCapacity);
```

### 11.2 历史 Planner V2 函数 ABI

`src/include/tilexr_moonep_planner.h` 中的：

```cpp
int TileXRMoonEpPlannerGetWorkspaceSizeV2(
    TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount,
    uint64_t *workspaceBytes, int64_t *dispatchedCapacity);

int TileXRMoonEpPlannerV2(/* 历史 Planner V2 参数，签名保持不变 */);
```

是已有的 Planner 函数签名，名称中的 `V2` 不等于 metadata struct 的 `abiVersion == 2`。这些函数继续作为 MoonEP V1 Planning 的兼容桥接层存在。

### 11.3 optimized Planner legacy ABI

`src/include/tilexr_ep_plan.h` 中已有：

```cpp
int TileXRMoeEpPlanV2GetWorkspaceSize(
    int64_t rankSize, int64_t s, int64_t topK, int64_t expertNum,
    const TileXRMoonEPPlanConfig *config,
    uint64_t *localWorkspaceBytes, uint64_t *registeredMetaBytes);

int TileXRMoeEpPlanV2(..., TileXRMoonEPPlanDesc *plan, ...);
```

`TileXRMoonEPPlanDesc` 没有 `structSize/abiVersion/count`。调用该 legacy optimized API 时，新增的 `remoteExperts/expertTargets` 输出被禁用，旧 `plan.expertsToCopy` 保持当前 Rank 行语义，见下一节。

### 11.4 metadata V2 ABI

新增接口：

```text
TileXRMoeEpPlanV2WithMetadata(..., TileXRMoonEPPlanMetadataV2 *metadata, ...)
```

要求：

```text
sizeof(TileXRMoonEPPlanMetadataV2) == 208
alignof(TileXRMoonEPPlanMetadataV2) == 8
metadata.structSize == sizeof(TileXRMoonEPPlanMetadataV2)
metadata.abiVersion == TILEXR_MOONEP_PLAN_METADATA_V2_ABI_VERSION == 2
```

并精确校验所有 element count：

| 字段 | 元素数 |
| --- | ---: |
| `dst` | `S*K` |
| `cuSeqlens` | `E+B` |
| `remoteExperts` | `R*B` |
| `expertTargets` | `(E/R)*ceil(R/64)` |
| `remoteStats` | `2` |
| `status` | `8` |
| `dupGroups` | `3*NvS` |
| `dupLoffs` | `NvS` |
| `dupCounts` | `2` |

metadata V2 是新增并存 ABI，不修改 `TileXRMoonEpPlanV1`、`TileXRMoonEPPlanDesc` 或历史函数签名。它同样不包含 `tokenRemap`；这是有意保留的 Dispatch 输出边界。

## 12. 旧 `expertsToCopy` 的兼容语义

这里需要区分 optimized descriptor 与更外层 MoonEP V1 buffer。

### 12.1 `TileXRMoonEPPlanDesc::expertsToCopy`

旧 optimized `TileXRMoonEPPlanDesc::expertsToCopy` 的有效长度是 `B`，表示：

```text
remoteExperts[currentRank, 0:B]
```

也就是**当前 Rank 作为目的 Rank**需要复制进来的 remote expert 行，不是完整 `[R,B]`。

- legacy `TileXRMoeEpPlanV2`：kernel 直接把当前 Rank 行写入该 `B` 元素 buffer；
- metadata V2：kernel 令内部兼容指针指向
  `metadata.remoteExperts + currentRank * B`，所以旧行语义仍保持不变，同时另行填完整 `remoteExperts[R,B]`。

下游不得把 legacy optimized `expertsToCopy[B]` 当作 owner 侧发送 bitmap；需要 owner 视角时使用 `expertTargets[E/R,W]`。

### 12.2 MoonEP V1 兼容层的 outward buffer

更外层 `TileXRMoonEpPlannerV2`/`TileXRMoonEpPlanV1::expertsToCopy` 现有 Torch/demo 调用按 `[R,B]` 分配。兼容层在 optimized Planner 完成后，把内部 `remoteExpertSet[R,B]` 异步复制到该 outward buffer。

因此：

- “旧 `expertsToCopy` 当前 Rank 行”专指 `TileXRMoonEPPlanDesc` 的 legacy optimized 契约；
- MoonEP V1 compatibility outward buffer 当前仍可得到完整 `[R,B]`；
- metadata V2 用明确命名的 `remoteExperts[R,B]` 消除这两层语义歧义。

## 13. 下游实现必须满足的检查表

### Expert Migration

- [ ] 按 `remoteExperts[dstRank,B]` 或当前 owner 的 `expertTargets[E/R,W]` 生成真实权重复制任务。
- [ ] 忽略 `remoteExperts == -1` 的 unused slot。
- [ ] 保证 `remoteExperts[rank,slot]` 的权重与 `cuSeqlens[E+slot]` 对应的专家计算 group 一致。
- [ ] 不把 Planner metadata mailbox 交换当作权重已搬运。

### Dispatch

- [ ] 按 `tokenId*K+topKId` 读取本地 `dst`。
- [ ] 使用 `encoded >= 0 ? encoded : ~encoded` 解码，禁止 `abs(encoded)`。
- [ ] 每个 `(token,dstRank)` 只跨 Rank 发送一次 hidden，但每条 route 都写 weight 和 remap。
- [ ] 初始化 `tokenRemap[NvS]` 为 `UINT64_MAX`。
- [ ] 在最终 `recvSlot` 写 `globalTokenId`，并检测 slot collision。
- [ ] 为 `~rawDst` route 完成目的端 hidden fan-out/复用。

### Combine

- [ ] 只消费 Dispatch 已填写的 `tokenRemap`。
- [ ] 跳过 `UINT64_MAX`。
- [ ] 用同一轮的 `R/S/K` 解码 `globalTokenId`。
- [ ] 按 `srcRank/tokenId/topKId` 做正确的跨 Rank 回传和 top-k 归并。

### ABI

- [ ] V1 与 metadata V2 分开校验各自的 `structSize/abiVersion` 规则。
- [ ] metadata count 使用元素数，不使用字节数。
- [ ] legacy optimized `expertsToCopy` 只按 `B` 元素当前 Rank 行使用。
- [ ] 不要求 Planner 提供 `tokenRemap` 或 global route pool。

## 14. 依据

本文按以下当前工作区证据整理：

- Public headers：
  - `src/include/tilexr_ep_plan.h`
  - `src/include/tilexr_moonep_planner.h`
  - `src/include/tilexr_moonep.h`
- Planner V2 实现：
  - `src/moonep/planner_v2/common/ep_plan_algorithm_impl.h`
  - `src/moonep/planner_v2/common/ep_plan_downstream.h`
  - `src/moonep/planner_v2/common/ep_plan_downstream.cpp`
  - `src/moonep/planner_v2/host/ep_plan_host.cpp`
  - `src/moonep/planner_v2/host/tilexr_moonep_planner.cpp`
  - `src/moonep/planner_v2/kernels/tilexr_moonep_planner_kernel.cpp`
- MoonEP stage 能力与 stub 实现：
  - `src/moonep/host/tilexr_moonep.cpp`
- 契约/ABI 与生产库消费测试：
  - `tests/ep/unit/test_tilexr_ep_plan_downstream.cpp`
  - `tests/ep/unit/test_tilexr_moonep_planner_public_abi.cpp`
  - `tests/ep/integration/test_tilexr_ep_plan_public_helper_consumer.cpp`
- 上下游梳理材料：MoonEP 上下游配合梳理文档及其提取文本；本规范不依赖不可移植的本机绝对路径。
