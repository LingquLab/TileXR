# TileXR 单机性能分析模块改动报告

## 背景与目标

本轮单机性能分析工作基于 `lhy-single-profiling` 分支，目标是在 Ascend950 单机多卡环境上跑通 standalone collectives 的算子内性能采集，并同时保留原有 HTML 报告与新增 `ui.perfetto.dev` 可打开的 trace JSON。

当前已验证链路覆盖：

- 70 环境：Ascend950PR_9599，单机 allgather profiling 跑通。
- 62 环境：Ascend950PR_9589，补充芯片名映射后单机 allgather profiling 跑通。
- 本地报告样例：`C:\TileXR\run\prof\collectives-950\report.html`、`C:\TileXR\run\prof\collectives-950-62\report.html`。
- Perfetto 样例：对应目录下的 `perfetto_trace.json` 可上传到 `https://ui.perfetto.dev` 查看。

## 代码改动概览

主要改动来自提交 `f80a607 feat: enable Ascend950 single-node profiling`，涉及 collectives kernel 编译、Ascend950 kernel 兼容、profile report 聚合和 Perfetto trace 导出。

当前工作区还有一个额外未提交补丁，用于 62 机器上的 `Ascend950PR_9589` 识别：

- `src/comm/tilexr_internal.cpp`：增加 `Ascend950PR_9589 -> CHIP_950`。
- `tests/comm/unit/test_tilexr_source_guards.cpp`：增加对应 source guard。

## Ascend950 编译与运行适配

Ascend950 单机 profiling 的关键编译适配在 `src/collectives/kernels/CMakeLists.txt`：

- 新增 `TILEXR_COLLECTIVES_SOC_TYPE` CMake 变量。
- 当 `TILEXR_COLLECTIVES_SOC_TYPE=Ascend950` 时，CCE AIV arch 使用 `dav-c310-vec`。
- 其他平台默认仍使用原来的 `dav-c220-vec`。

相关 CCE 源文件也扩展了 `__DAV_C310_VEC__` 条件，使 Ascend950 编译时能生成 collectives kernel：

- `src/collectives/kernels/tilexr_lccl_op.cpp`
- `src/collectives/kernels/lccl_op.h`
- `src/collectives/kernels/collectives.h`
- `src/collectives/kernels/kernels/collectives.cce`
- `src/collectives/kernels/datacopy_gm2gm.h`

`datacopy_gm2gm.h` 还增加了 `TileXRAtomicTypeSupported`，避免 `dav-c310-vec` 下对不支持类型生成 atomic 指令。

## Profiling 采集链路

运行入口仍然是 `tests/collectives/run_collective_perf.sh` 和 `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`。

典型构建命令：

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S . -B build-profile-950 \
  -DTILEXR_BUILD_COLLECTIVES=ON \
  -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON \
  -DTILEXR_COLLECTIVES_SOC_TYPE=Ascend950
cmake --build build-profile-950 --target tilexr_collective_perf -j"$(nproc)"
```

典型运行命令：

```bash
cd tests/collectives
./run_collective_perf.sh 2 0 ../../build-profile-950/tests/collectives \
  --op allgather \
  --min-bytes 4096 \
  --max-bytes 4096 \
  --iters 2 \
  --warmup-iters 1 \
  --datatype int32 \
  --check 0 \
  --profile 1 \
  --profile-dir ../../run/prof/collectives-950 \
  --profile-sample-every 1 \
  --profile-ai-prompt 1
```

采集流程：

- 每个 measured launch 创建一个 `TileXRCollectivePerfSession`。
- kernel launch 前通过 `PreparePerfTraceLaunch` 准备 device trace buffer。
- kernel 内部写入 stage/core/rank 维度的统计信息。
- launch 完成后调用 `TileXRCollectivePerfWriteReport` 输出单 launch 报告。
- `run_collective_perf.sh` 等所有 rank 结束后调用 `tilexr_collective_profile_report.py` 生成聚合报告。

## 输出产物

每个 rank/launch 目录会生成：

- `trace.json`：单次 launch 的结构化 trace，schema 为 `tilexr_perf_trace_report.v1`。
- `summary.csv`：stage/core 维度统计表。
- `analysis.md`：单 launch 文本分析。
- `report.html`：单 launch HTML drilldown。
- `ai_prompt.md`：可选，开启 `--profile-ai-prompt 1` 时生成。

profile 根目录会生成聚合产物：

- `report.html`：保留原有 HTML 呈现，包含 bottleneck-first 摘要、timeline、drilldown 链接。
- `trace_index.json`：聚合后的中间索引，schema 为 `tilexr_perf_trace_run.v1`。
- `analysis.md`：跨 rank/launch 的文本摘要。
- `ai_prompt.md`：可选聚合 prompt。
- `perfetto_trace.json`：新增 Perfetto/Chrome trace event 格式，给 `ui.perfetto.dev` 使用。

聚合摘要现在额外包含 rank-level kernel summary：

- `summary.rank_kernel`：按 rank 汇总 `kernel_total`，记录 launch 数、平均 kernel us、最大 kernel us、最慢 launch。
- `summary.slowest_rank`：按平均 kernel us 排序选出的最慢 rank。
- `report.html`：新增 `Rank-Level Summary` 表格，用于快速定位慢 rank 后再跳转到单 launch drilldown。
- `analysis.md` / `ai_prompt.md`：同步输出 slowest rank 和 rank kernel totals。

## Perfetto Trace 支持

`tests/collectives/tilexr_collective_profile_report.py` 新增 `render_perfetto_trace()`：

- 每个 rank 映射为 Perfetto process。
- 每个 rank/core 映射为 Perfetto thread，thread 名为 `rankN/coreM`。
- 每个 profiling stage 映射为 `ph: "X"` duration event，事件名为 `launchN/rankR/stage`。
- 每个 rank 增加 `launch_windows` thread，并写入 `launchN/rankR/window` 对齐窗口。
- measured launch 之间增加固定 gap，方便在 `ui.perfetto.dev` 中按 launch 展开和搜索。
- 事件保留 `launch_id`、`launch_offset_us`、`normalized_ts`、`rank`、`core`、`stage`、`stage_id`、`sum_us`、`raw_cycles`、`max_cycles`、`message_bytes`、`rank_size` 和源 trace 路径。

注意：不同 NPU 的原始 cycle offset 不假设同步，因此 HTML 和 Perfetto 都按 rank/launch 内部归一化时间展示，更适合观察单 rank 内核阶段耗时、stage 分布和慢 core，而不是直接比较跨 NPU 的绝对开始时间。

## 测试覆盖

已补充/更新的测试包括：

- `tests/collectives/unit/test_collective_profile_report.py`
  - 校验聚合 HTML、`trace_index.json`、`analysis.md`。
  - 校验 rank-level summary、slowest rank、Perfetto launch window 和 launch/rank/stage 事件命名。
  - 校验 sparse launch、multi-size launch、missing trace diagnostics。
- `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`
  - 校验 README 和 profile report helper 保留 rank summary / Perfetto marker 关键字段。
- `tests/collectives/unit/test_tilexr_collectives_api.cpp`
  - 校验 Ascend950 CCE arch 配置。
  - 校验 `__DAV_C310_VEC__` 相关源码路径。
- `tests/comm/unit/test_tilexr_source_guards.cpp`
  - 校验已观察到的 `Ascend950PR`、`Ascend950PR_9589`、`Ascend950PR_9599` variant 被映射为 `CHIP_950`。

## 当前限制

- 当前单机 profiling 路径依赖 standalone collectives kernel 内的 trace hook，主要用于分析 allgather/allreduce/reducescatter/broadcast/alltoall 这类已有 standalone 算子。
- warmup launch 不写 profiling trace，聚合报告只展示 measured launch。
- rank-level summary 已能直接看慢 rank，但更细的自动归因仍需要结合 stage/core drilldown 分析。
- 多机 62+70 的 communicator bootstrap 可以成功，但 2 机 collective kernel 目前会在 kernel 阶段失败或超时；这属于跨节点数据面支持问题，不属于本报告覆盖的单机 profiling 范围。

## 建议后续

- 短期：继续用 70 的单机 allgather/profile 结果检查 rank-level summary 的可读性，并按真实数据调整排序和字段展示。
- 短期：在 HTML 中补充 rank 间差值/百分比，例如 slowest rank 相比 fastest rank 慢多少。
- 中期：为 Perfetto 增加更多 marker，例如 message size group、op group、rank delta marker，进一步降低定位慢 stage 的成本。
- 中期：另起多机 profiling 任务，先解决 collective kernel 跨节点数据面，再复用当前 profile report 聚合链路。
