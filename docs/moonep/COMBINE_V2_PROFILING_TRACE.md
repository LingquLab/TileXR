# MoonEP Combine V2 Profiling 与 Chrome Trace 导出

本文只记录 profiling 模式的构建、采集和三个性能拆解 JSON 的生成。常规多机测试
流程见 [COMBINE_V2_MULTIHOST_PERF.md](COMBINE_V2_MULTIHOST_PERF.md)。

## 1. 开启 Profiling

profiling 必须同时在编译和运行阶段开启。编译时添加 `--enable-profiling`：

```bash
bash "${REMOTE_ROOT}/source/tools/moonep/build_combine_v2_perf.sh" \
  --source-dir "${REMOTE_ROOT}/source" \
  --build-dir "${REMOTE_ROOT}/build" \
  --install-dir "${REMOTE_ROOT}/install" \
  --cann-path "${CANN_PATH}" \
  --jobs 16 \
  --enable-profiling
```

将 `install/` 平铺同步到 hostfile 中的全部节点后，按常规 launcher 命令运行，并
添加 `--profile`。no-reduce 不添加 `--reduce-hidden`；reduce 测试同时添加
`--reduce-hidden`。

```bash
bash "${REMOTE_ROOT}/source/tools/moonep/run_combine_v2_perf_multihost.sh" \
  --hostfile "${REMOTE_ROOT}/hostfile" \
  --install-dir "${REMOTE_ROOT}/install" \
  --cann-path "${CANN_PATH}" \
  --ssh-user root \
  --bs 8192 --warmup 0 --iterations 80 \
  --experts 256 --hidden-size 3584 \
  --comm-domain 141 --comm-id "${PRIMARY_HOST}:10067" \
  --timeout 600 \
  --log-file "${LOG_FILE}" \
  --profile
```

profiling 会扰动耗时，只用于阶段归因；宏观性能以关闭 profiling 的测试为准。
计时阶段连续提交全部轮次，轮次之间不执行 Host barrier、同步或 profile D2H。
全部轮次结束后只拷回最后一轮 profile。正确性失败不应阻止最后一轮
`COMBINE_V2_SAMPLE`、`COMBINE_V2_PROFILE` 和 rank 性能记录输出。

Profile v5 在原有 26 个累计阶段时间点和 8 个 metric 之外，使用 record 的诊断槽记录
Fullmesh 路由与五个绝对 cycle 边界：

```text
FM_WQE_BUILD_END
< FM_SUBMIT_END
< FM_CQ_SUCCESS
< CLOS_GRANT_SUBMIT
< CLOS_GRANT_CQ_SUCCESS
```

`COMBINE_V2_PROFILE` 同步输出 `transport`、`fm_step`、`fm_peer`、
`fm_successor`、`fm_logical_qp` 和上述五个边界。非 Fullmesh Core 的 route 为 `-1`、
边界为 `0`。当 successor 是本卡时，`CLOS_GRANT_CQ_SUCCESS` 表示两个本地 grant 已发布
并完成 cache clean；此时没有 CLOS WQE/CQE。trace event 的 `grant_transport=local` 用于
区分该语义。

Server-Grant 调度不执行 Legacy/CLOS Grant，因此 `CLOS_GRANT_SUBMIT` 和
`CLOS_GRANT_CQ_SUCCESS` 必须同时为 `0`。该模式仍要求前三个 Fullmesh 边界严格递增，
trace 中省略两个不存在的 Grant 事件并标记 `grant_transport=none`。

## 2. 生成三个 JSON

将包含全部 rank 输出的完整日志同步到本地后执行：

```bash
python tools/moonep/combine_v2_trace.py COMBINE_LOG \
  --split-output-dir TRACE_DIR \
  --prefix combine_v2_no_reduce \
  --host HOST \
  --iteration 79 \
  --world-size 128 \
  --bs 8192 \
  --topk 16 \
  --hidden-size 3584 \
  --experts 256 \
  --reduce disabled
```

reduce 测试把 `--prefix` 改为 `combine_v2_reduce`，并传
`--reduce enabled`。不要传 `--output`；`--split-output-dir` 会直接生成三个彼此
独立、可在 Chrome Tracing 中打开的文件：

```text
<prefix>_iteration79_fastest_rank<RANK>_edge_trace.json
<prefix>_iteration79_p50_rank<RANK>_edge_trace.json
<prefix>_iteration79_slowest_rank<RANK>_edge_trace.json
```

## 3. 选卡与轨道口径

- 选卡只看最后一轮，即 `iteration=79` 的 kernel profile 耗时。
- 每张卡的选卡耗时取 16 个 active Core 中最大的 `t25 - t0`，即最慢 Core 的完整
  kernel 时长；日志标记为 `timing_source=kernel_profile`。
- fastest 是最小耗时 rank，slowest 是最大耗时 rank。
- P50 是 128 个 rank 按耗时升序排列后的第 64 张卡，不是耗时数值的插值中位数。
- 相同耗时按 rank 编号排序，保证结果可重复。
- 16P 及以上每个 JSON 必须包含 16 条 AIV Core 轨道；2-8P 使用与 rank 数相同的
  Core 数。
- 每个 JSON 只含一张选中卡，不要把 fastest、P50 和 slowest 合并成一个 trace。

每张卡以该卡最早的 AIV `t0` 作为时间零点。不同 NPU 的系统 cycle 不保证同步，
因此不能比较三个 JSON 的绝对起点。

## 4. 导出前检查

128P、80 轮的 profiling 日志应包含 128 条最后一轮 `COMBINE_V2_SAMPLE`、128 条
`COMBINE_V2_RANK_PERF`，以及 `128 * 16 = 2048` 条最后一轮
`COMBINE_V2_PROFILE`。导出工具还要求存在与 `BS/K/H/ExpNum/reduce` 一致的
`COMBINE_V2_PERF` provenance 记录；参数不一致、时间戳非正、时间戳非单调或所选
rank 缺少任一 active Core 时会拒绝生成 JSON。

生成后终端应恰好输出三条 `COMBINE_V2_EDGE_TRACE`，并分别标记
`role=fastest`、`role=p50` 和 `role=slowest`。这三个 JSON 是 profiling 分析产物；
等效算法带宽仍直接从关闭 profiling 的 `COMBINE_V2_PERF` 输出读取。

## 5. Fullmesh 实现约束

- 普通 workspace registration 同时发布 CLOS registry 与 Fullmesh generation；profile
  registration 仍只绑定 32 个 CLOS QP，不能扩展或替换普通 Fullmesh 注册。
- 一个 direct route 必须匹配唯一单端口 topology edge 和唯一 local EID。歧义 route
  会导致对端 slot/import 关系不可证明，必须在 capability 发布前失败。
- 2P-8P 的数据 step 可能全部走 Fullmesh，因此 full-sync 必须自行产生并消费 CLOS CQ；
  不得依赖后续数据 CQ 回收 full-sync SQ。

以上约束由 source/compile 测试覆盖，但不能替代 Ascend950 上对实际 EID、CQ 状态、数据
正确性和时序的验证。
