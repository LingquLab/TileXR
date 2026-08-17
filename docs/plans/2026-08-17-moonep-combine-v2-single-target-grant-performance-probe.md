# MoonEP Combine V2 R10-Q 单目标 Grant 性能穿刺

状态：实现与验证完成

日期：2026-08-17

## 1. 假设

R9-Q 把每个 core 的 Grant 等待从四个 signal 缩为一个，并移除每 round 卡内
collective；128P `max_ms` 两轮均值为 `4.979520 ms`，相对 R8 提升 `4.82%`。但发送端
仍为每个 core、每个 round 向四张目标卡分别发布一个 Grant WQE。

R10-Q 只改变发送目标数：每个 source card/core 仅向实际执行
`WaitServerGrantDirect()` 的唯一 target card/core 发布一个 Grant。若四目标发布和对应 CQ
构成剩余成本，128P `max_ms` 应进一步低于 R9-Q；否则瓶颈主要位于 payload、Grant CQ
固定成本或其他数据热点。

## 2. 唯一映射

```text
sourceCardLane = (sourceRank % 8) / 2
targetRank = ServerGrantTargetRank(sourceRank, sourceCardLane, rankSize)
sourceRank == DirectGrantSourceRank(targetRank, rankSize)
```

该映射保持 source/target 的 local rank 相同，并且是 R9-Q direct wait 映射的严格逆。
32P/64P/128P 中每个 source 恰有一个 target，每个 target 也恰有一个 source。32P 的
`phaseStepCount=1`，因此 target 是本 rank，只需发布本地 signal；64P/128P 的 target
为非自身 rank，发布一个远端 WQE。

## 3. 单变量边界

- `PublishServerGrants()` 从四目标循环缩为 32P 一个本地 signal 或 64P/128P 一个
  ordered-completion WQE；
- signal ordinal、source/receive workspace 分区和 guard 不变；
- 每 round Grant CQ drain、direct wait、phase barrier 和 final closure 不变；
- payload、Done、Fullmesh、shared-QP、Host/Python API 和 blockDim 不变；
- 不修改远端 `compile.sh`、`sync_runtime.sh` 或 `run.sh`。

## 4. 验证顺序

1. schedule oracle 穷举 32P/64P/128P 唯一映射和双向 inverse；
2. source guard 确认 publisher 只有一个 target、一个 WQE 且保留 ordered completion；
3. 执行六个 Combine V2 Host/unit/source-guard 门禁；
4. CANN 9.1 AICore target compile；
5. 0/2 柜 Ascend950 128P、BS8192、K16、H3584、Exp256、BF16、no-reduce、
   warmup20、iterations80 正确性和性能测试；
6. 与 R9-Q `4.979520 ms` 及 R8 `5.231500 ms` 比较。

只执行现有部署和验证命令：

```bash
mutagen sync flush tilexr-combine-v2-cab0-2-fast
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/compile.sh'
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/sync_runtime.sh'
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/run.sh'
```

## 5. 判定

- 正确性门禁：128/128 rank `correctness=passed`，无 CQ、Grant、Done 或 barrier timeout；
- 性能主指标：同口径 `max_ms`；
- 若性能无明显改善，停止在该探针，不叠加 payload batch 或 closure 优化；
- 本轮结论仅覆盖 128P no-reduce，不外推到 32P/64P、Reduce 或 FP32 route-weight。

## 6. 实现与验证结果

R10-Q 已完成单目标 publisher 实现。64P/128P 每个 source card/core 只向
`MoonEpCombineV2DirectGrantTargetRank()` 返回的唯一 rank 发布一个
ordered-completion WQE；32P direct target 为本 rank，因此只写 receive partition 的本地
signal。R9-Q direct wait、每 round CQ drain、phase barrier 和 final closure 均未改变。

CANN 9.1 AICore target compile 通过，安装库 SHA256 为：

```text
d49c496d6b03759516c833b3664cb038079f470bd2d6ba4c4fe7c2900c48a97e
```

schedule、layout、host、launch、public ABI 和 source-guard 六个测试 target 重新构建并
全部通过。现有 `compile.sh` 只构建 perf target，不能直接把 build 目录中已有的测试
二进制当作当前源码证据；必须显式构建对应测试 target 后再执行。

0/2 柜 Ascend950 128P production case 连续执行两轮：

| 运行 | avg_ms | max_ms | 正确性 | 日志 |
| --- | ---: | ---: | --- | --- |
| R10-Q-1 | 4.910772 | 4.918600 | 128/128 passed | `combine_v2_128p_noprofile_20260817_031358.log` |
| R10-Q-2 | 4.911296 | 4.921610 | 128/128 passed | `combine_v2_128p_noprofile_20260817_031528.log` |

两轮各包含 128 条 `COMBINE_V2_RANK_PERF`；128 个 rank 和聚合结果均为
`correctness=passed`，未出现 timeout 或正确性失败。两轮 `max_ms` 均值为
`4.920105 ms`：

- 相对 R9-Q `4.979520 ms` 减少 `0.059415 ms`，提升 `1.19%`；
- 相对 R8 `5.231500 ms` 减少 `0.311395 ms`，提升 `5.95%`。

结果证明四目标 Grant 发布和附带 WQE/CQ 工作存在约 `0.059 ms` 的稳定成本，但它不是
剩余性能差距的主要来源。后续若继续穿刺，应把 Grant 合入 payload final batch 作为新的
单变量；不能把本轮收益外推为删除全部 Grant CQ 的收益。

## 7. Profiling 扩展验证

状态：完成

本轮不再改变 R10-Q 算法，只补充一轮关闭 profiling 的宏观性能证据，并用隔离的
`build_profile_r10`/`install_profile_r10` 构建 profiling 版本，避免覆盖前述无扰动产物。
profiling case 保持 128P、BS8192、K16、H3584、Exp256、BF16、no-reduce、80 iterations，
只把 warmup 设为 0 并添加 `--profile`。

验收要求：

- 新无 profiling 运行包含 128 条 rank 性能记录且 128/128 正确性通过；
- profiling 运行包含 128 条最后一轮 `COMBINE_V2_SAMPLE`、128 条
  `COMBINE_V2_RANK_PERF` 和 2048 条 `COMBINE_V2_PROFILE`；
- exporter 生成 fastest、P50、slowest 三个彼此独立的 Chrome Trace JSON；
- JSON 每个包含 16 条 active AIV core 轨道，并保留当前源码、二进制 SHA、shape 和日志
  provenance；
- profiling 耗时只用于阶段拆解，不与关闭 profiling 的 `max_ms` 混作宏观性能结论。

最新关闭 profiling 的复测仍为 128/128 正确性通过，`avg_ms=4.916513`、
`max_ms=4.925600`，日志为 `combine_v2_128p_noprofile_20260817_032121.log`。该结果与
前两轮 `max_ms=4.918600/4.921610` 的方向和量级一致。

profiling 版本安装库 SHA256 为：

```text
1c99bd178aba40831383923b9c9046ab21e487a5bbc266dcd9442adbcbd727f7
```

profiling 运行的 `avg_ms=5.356000`、`max_ms=5.374750`，128/128 正确性通过。日志
`combine_v2_128p_profile_r10_single_target_20260817_0325.log` 包含 128 条
`COMBINE_V2_SAMPLE`、128 条 `COMBINE_V2_RANK_PERF` 和 2048 条
`COMBINE_V2_PROFILE`，无 timeout。profiling 时间包含埋点开销，只用于下表拆解：

| 角色 | rank/core | kernel us | setup us | full-sync us | 8 step send us | wait + inbound us | finalize us | wait 占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fastest | 8/10 | 4632.520 | 8.001 | 187.391 | 295.350 | 4139.849 | 1.929 | 89.36% |
| P50 | 87/5 | 4783.730 | 7.849 | 84.255 | 291.709 | 4397.948 | 1.969 | 91.94% |
| slowest | 59/9 | 4881.838 | 8.017 | 169.055 | 295.282 | 4407.600 | 1.884 | 90.29% |

三个 trace 分别为：

```text
artifacts/moonep/r10_single_target_profile_20260817_0325/traces/
  combine_v2_r10_single_target_no_reduce_128p_iteration79_fastest_rank8_edge_trace.json
  combine_v2_r10_single_target_no_reduce_128p_iteration79_p50_rank87_edge_trace.json
  combine_v2_r10_single_target_no_reduce_128p_iteration79_slowest_rank59_edge_trace.json
```

每份 JSON 含 16 条 AIV core 轨道和 471 个 trace event。fastest 的最大 wait 是 step0
`831.334 us`；P50 是 step0 `946.649 us` 和 step4 `931.457 us`；slowest 是 step3
`906.870 us`。热点 round 随 rank 移动，而 send 总和稳定在约 `292-295 us`，说明剩余
差异主要累积在 wait 区间。当前 `Step N wait` 同时包含 data CQ、单目标 Grant 发布/CQ、
direct Grant wait，以及 phase 边界上的 barrier；若要继续拆分这些成分，必须新增独立
profile point，不能从本轮 JSON 反推。
