# MoonEP Combine V2 R9-Q 点对点 Grant 性能探针

状态：实现与验证完成

日期：2026-08-17

## 1. 目标

R8 在 0/2 柜 Ascend950 128P、BS8192、K16、H3584、Exp256、BF16、no-reduce、
warmup20、iterations80 下的最大耗时为 `5.231500 ms`。Profile 显示大多数 round 的
16 个 core 在 `READY_END` 前只相差约 1-3 us，说明每轮 Server-Grant collective 把整卡
推进速度限制为最慢 core。

R9-Q 只验证以下假设：

```text
每核等待 4 张源卡的 Grant
+ 每轮 3 次 SyncAll<true>()
+ collective status 共享 cache line
= Server-Pair 流水退化为整卡规整 barrier
```

本版本把每个接收 core 的等待集合从四个 Grant 缩为一个直接 Grant，并删除每 round 的
卡内 collective。发送端暂时保持 R8 的四卡广播，因此性能收益是最终点对点协议收益的
保守下界。

## 2. 范围与非目标

范围：

- 仅改变 32P/64P/128P parity Server-Pair 的 Server-Grant admission；
- 每个 target card/core 只等待 predecessor server 中相同 local-rank card/core 的 signal；
- leading barrier 和 phase barrier 保持不变；
- 保持每个 round 的 Grant 发布和 final closure，确保与 R8 的 A/B 变量最少；
- 调整失败控制流，使所有 core 在同一 phase collective 汇合前执行相同循环骨架。

非目标：

- 不减少 `PublishServerGrants()` 的四个 target card；
- 不把 Grant 合并回 payload final batch；
- 不删除 phase 末尾或最终 round 的 Grant closure；
- 不修改 peer、Done、Fullmesh、shared-QP、workspace、Host API 或 Python API；
- 不修改远端 `compile.sh`、`sync_runtime.sh` 和 `run.sh`。

## 3. Direct Grant 映射

设 target card 为 `C`：

```text
directSourceCardLane = (C.rank % 8) / 2
directSourceRank = ServerGrantSourceRank(C.rank, directSourceCardLane, R)
directGrantOrdinal = directSourceCardLane * 16 + C.core
```

这选择 predecessor server 中与 C 相同 local rank 的源卡 A。对于 phase 内非末尾 round：

```text
pair(A.server, phaseStep) == pair(C.server, phaseStep + 1)
peer(A.rank, round, core) == peer(C.rank, round + 1, core)
```

因此 A/core 完成到 B 的数据后产生的 signal 可以直接授权 C/core 下一 round 向同一 B
发送。R9-Q 仍保留 phase 末尾和最终 closure；这些 signal 只用于保持 A/B 变量一致，
不作为最终协议设计结论。

## 4. Kernel 控制流

R8：

```text
data CQ -> publish 4 Grant WQEs -> Grant CQ
-> 每核等待 4 个 Grant -> Begin/EndCollectiveStage -> 3 x SyncAll
```

R9-Q：

```text
data CQ -> publish 4 Grant WQEs -> Grant CQ
-> 每核等待 1 个 direct Grant -> 下一 round
```

热路径不得调用 `BeginCollectiveStage()`、`EndCollectiveStage()` 或 `SyncAll<true>()`。
phase barrier 仍由 16 个 core 无条件进入。单 core 失败时不能立即跳出 phase 循环；失败
core 只停止后续数据工作，并与其他 core 在 phase barrier 或最终 Done collective 汇合，
避免进入不同的硬件 barrier 序列。

## 5. 修改面

| 文件 | 修改 |
| --- | --- |
| `src/moonep/common/moonep_combine_schedule.h` | direct source lane/rank/ordinal helper |
| `src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h` | 单 signal wait、绕过 round collective、固定 phase 控制流 |
| `tests/moonep_combine_v2/unit/test_combine_v2_schedule.cpp` | direct 映射、inverse 和同 peer 穷举 |
| `tests/moonep_combine_v2/unit/test_combine_v2_source_guard.cpp` | 热路径无 SyncAll、phase 汇合拓扑检查 |

## 6. 验证顺序

1. Host schedule oracle：32P/64P/128P direct source 唯一、地址合法，phase 内相邻 round
   指向同一 peer；
2. source guard：direct wait 只读取一个 ordinal，且不调用 collective helper；
3. 现有 Combine V2 Host/unit/source-guard 门禁；
4. CANN 9.1 target compile；
5. 0/2 柜 128P production case，要求 128/128 correctness passed；
6. 与 R8 `5.231500 ms` 比较 `max_ms`。

远端验证只执行现有四条命令：

```bash
mutagen sync flush tilexr-combine-v2-cab0-2-fast
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/compile.sh'
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/sync_runtime.sh'
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/run.sh'
```

## 7. 结果判定

| R9-Q max_ms | 结论 |
| ---: | --- |
| `< 4.97 ms` | 提升超过约 5%，整卡同步假设成立 |
| `4.97-5.18 ms` | 同步有影响，但 Grant 发布/CQ 或数据热点仍占主要部分 |
| `>= 5.18 ms` | 整卡同步不是主要瓶颈，优先分析数据路径和发布端成本 |

若假设成立，后续版本再依次验证：发送端从四卡广播缩为单卡、恢复双 lane 点对点 Grant、
把 Grant 合并到 final payload batch，以及删除无后继数据的 closure。本探针不提前实施这些
优化。

## 8. 实现与验证结果

R9-Q 已按本方案落地。`Process()` 热路径使用单 signal direct Grant admission，不再调用
`RunServerGrantAdmission()`，每 round 的 `BeginCollectiveStage()`、
`EndCollectiveStage()` 和三次 `SyncAll<true>()` 已移除；leading barrier 和 phase barrier
保持不变。发送端仍由 `PublishServerGrants()` 向四张目标卡发布 Grant。

CANN 9.1 AICore target compile 通过，产物 SHA256 为：

```text
ccb83a552a62db24d18eb23cbfded5f475193630d197008f3c766b59f1f94e81
```

以下六个 Host/unit 门禁均通过：

```text
test_tilexr_moonep_combine_v2_schedule
test_tilexr_moonep_combine_v2_layout
test_tilexr_moonep_combine_v2_host
test_tilexr_moonep_combine_v2_launch
test_tilexr_moonep_combine_v2_public_abi
test_tilexr_moonep_combine_v2_source_guard
```

0/2 柜 Ascend950 128P production case 连续执行两轮：

| 运行 | avg_ms | max_ms | 正确性 | 日志 |
| --- | ---: | ---: | --- | --- |
| R9-Q-1 | 4.967103 | 4.977630 | 128/128 passed | `combine_v2_128p_noprofile_20260817_025341.log` |
| R9-Q-2 | 4.970783 | 4.981410 | 128/128 passed | `combine_v2_128p_noprofile_20260817_025607.log` |

两轮各包含 128 条 `COMBINE_V2_RANK_PERF`，128 个 rank 和最终聚合结果均为
`correctness=passed`，未出现 CQ、Grant、Done、barrier timeout 或正确性失败。两轮
`max_ms` 均值为 `4.979520 ms`，相对 R8 的 `5.231500 ms` 减少 `0.251980 ms`，提升
`4.82%`。

结果落在 `4.97-5.18 ms` 区间：整卡 Grant admission 确有可测成本，但不是剩余性能差距
的全部来源。由于发送端仍为每个 core 发布四个 Grant WQE，本结果只给出点对点 Grant
最终优化收益的保守下界。下一步应单独验证发送端单目标 Grant，再评估与 payload final
batch 合并；本轮不扩大实现范围。
