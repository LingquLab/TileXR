# TileXR 多机性能分析模块改动整理

## 背景与目标

本轮多机性能分析工作以 `C:\TileXR` 的 `lhy-single-profiling` 分支为基本盘，目标是在 62 和 70 两台机器上跑通 2 机 kernel-level profiling，并输出统一的 HTML 报告和 `perfetto_trace.json`。

当前已验证链路：

- 62：`root@141.62.24.62`，远端路径 `/home/l00929943/TileXR`。
- 70：`root@141.62.24.70`，远端路径 `/home/l00929943/TileXR`。
- 算子：`profile-probe`。
- 输出目录：`/home/l00929943/TileXR/run/prof/collectives-2host-profile-probe-62-70-kernel-align`。
- 本地报告：`C:\TileXR\run\prof\collectives-2host-profile-probe-62-70-kernel-align\report.html`。
- 本地 Perfetto：`C:\TileXR\run\prof\collectives-2host-profile-probe-62-70-kernel-align\perfetto_trace.json`。

注意：`profile-probe` 是多机 profiling/report smoke 算子。它会初始化 ACL 和 socket communicator，并在每个 rank 上执行真实 AIV kernel，记录与单机 profiling 一致的 7 个 kernel stage；但它只做本地 device memory copy，不触碰跨机 `peerMems[]`，不能当作 allgather 跨机带宽结果。

## 必须保留的改动

### 1. `profile-probe` 算子闭包

这些改动是多机 profiling 能稳定跑通的核心，提交时必须保留：

- `src/include/tilexr_types.h`
  - 新增 `TileXRType::PROFILE_PROBE = 11`。
  - 新增 `TileXRProfileProbe` 类型名映射。
- `src/include/tilexr_collectives.h`
  - 新增 public API `TileXRProfileProbe(...)`。
- `src/collectives/host/tilexr_collectives.cpp`
  - 新增 host 侧 `TileXRProfileProbe` 实现。
  - 使用 `GetProfileProbeBlockNum` 固定 probe block 数。
  - 以 `INT8` kernel signature launch，避免 profile probe 与 dtype 组合膨胀。
- `src/collectives/host/collective_utils.{h,cpp}`
  - 新增 `GetProfileProbeBlockNum`。
- `src/collectives/host/collective_kernel.cpp`
  - 注册 `PROFILE_PROBE` kernel。
  - 对 `PROFILE_PROBE` 使用独立 funsig 和 kernel name。
- `src/collectives/kernels/lccl_op.h`
  - include `kernels/lcal_profile_probe.cce`。
  - 新增 `LCCL_PROFILE_PROBE_FUNC_AUTO_DEF()`。
- `src/collectives/kernels/tilexr_lccl_op.cpp`
  - 注册 `TileXRProfileProbe` kernel TU。
- `src/collectives/kernels/kernels/lcal_profile_probe.cce`
  - 新增真实 AIV probe kernel。
  - 输出和单机一致的 7 个 stage：`kernel_total`、`chunk_total`、`post_sync`、`local_input_to_ipc`、`flag_poll_wait`、`peer_ipc_to_output`、`chunk_barrier`。
  - 只做本地 GM -> UB -> GM copy，不访问 `peerMems[]`。
- `src/collectives/kernels/CMakeLists.txt`
  - `TILEXR_COLLECTIVES_1OP_BIN_SIZE` 从 `5242880` 增大到 `10485760`，否则新增 CCE binary 后可能被截断。

### 2. kernel trace 数据面修复

这些改动保证多机和单机报告的 kernel 粒度一致，也是必须保留的：

- `src/include/tilexr_perf_trace.h`
  - 新增 `TILEXR_PERF_TRACE_STATS_OFFSET = 128`。
  - `TileXRPerfCoreStageStats` 扩展到 96 bytes，并保持 32-byte 对齐。
  - 新增 `aux2/aux3`，防止 ABI/layout 与 kernel 侧不一致。
- `src/collectives/kernels/perf_trace_kernel.h`
  - 使用固定 GM stats offset，避免 kernel 侧读取 header 字段导致不稳定。
  - 通过 UB 临时 buffer 做 stats slot 的 GM <-> UB 更新。
  - 使用 `GetBlockNum()` 和固定 stage count 计算 slot。
  - 保留无 profiling 编译路径的 no-op overload。
- `src/collectives/host/perf_trace_session.cpp`
  - host 侧 header 使用同一个 `TILEXR_PERF_TRACE_STATS_OFFSET`。
  - 支持 `TILEXR_COLLECTIVES_DISABLE_KERNEL_PROFILING=1`，用于只保留 host metadata 的调试对照。
  - 新增 incomplete report 写出入口。
- `src/include/tilexr_collectives_perf.h`
  - 对外声明 `TileXRCollectivePerfWriteIncompleteReport(...)`，让测试工具在 kernel launch 失败时仍能写出可聚合的 trace metadata。
- `src/collectives/host/perf_trace_report.{h,cpp}`
  - 只保留 `count != 0` 的 stats，避免 count=0 的脏 slot 进入报告。
  - `trace.json` 写出 `aux2/aux3`。
  - 支持 incomplete trace 的 `incomplete` / `incomplete_reason`。

### 3. 多机运行入口与测试工具

这些改动让 62/70 可以一键启动 2 机 profiling，必须保留：

- `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`
  - 新增 `--comm-mode local|socket`。
  - 新增 `--device-id`。
  - 新增 `--op profile-probe`。
  - socket 模式下使用 `TileXRCommInitRank`。
  - 写出 `rank<N>/host_info.json`，包含 host/IP/comm_mode。
  - 在 measured launch 失败时写 incomplete trace，便于聚合报告诊断。
  - `profile-probe` 支持 `--check 1`，用于确认 profiling 没有导致执行错误。
- `tests/collectives/run_collective_perf_multihost.sh`
  - 读取 `TILEXR_MULTIHOST_PEERS`，按 rank 通过 SSH 启动远端进程。
  - 设置 `--comm-mode socket`、`TILEXR_COMM_ID`、`ASCEND_PROCESS_LOG_PATH`。
  - 优先用当前构建产物的 `LD_LIBRARY_PATH`，并将 driver lib 放前面。
  - 远端 rank 完成后回收 `rank<N>/` profiling 目录。
  - 调用 `tilexr_collective_profile_report.py` 生成聚合报告。
- `tests/collectives/CMakeLists.txt`
  - 安装 `run_collective_perf_multihost.sh`。
  - 增加 rpath-link，避免测试工具链接时误用 CANN stub/缺失 driver 依赖。
- `tests/collectives/README.md`
  - 补充 multi-host profiling 使用说明和限制。

### 4. 聚合报告和 Perfetto 输出

这些改动让多机信息真正呈现在 HTML 和 Perfetto 中，必须保留：

- `tests/collectives/tilexr_collective_profile_report.py`
  - 聚合 `rank*/host_info.json`。
  - HTML 显示 `rank@host` 和 host IP。
  - `Rank-Level Summary` 用 `kernel_total` 汇总每个 rank 的 avg/max kernel us。
  - `Trace Status` 显示 incomplete/synthetic trace。
  - `perfetto_trace.json` 使用 host/rank/stage 命名，例如 `launch0/rank1@141.62.24.70/kernel_total`。
  - 每个 rank 增加 `launch_windows` thread，便于 `ui.perfetto.dev` 中按 launch 对齐查看。
  - 保留诊断：missing trace、incomplete trace、group incompatible 等。

### 5. 回归测试与 source guard

这些测试是提交多机 profiling 时保护行为不退化的必要部分，建议全部保留：

- `tests/collectives/unit/test_collective_profile_report.py`
  - 覆盖 host metadata、Perfetto、incomplete trace、count=0 过滤。
  - 新增 `test_multihost_report_preserves_single_host_kernel_stage_granularity`，确保多机报告保留单机同款 7 个 kernel stage。
- `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`
  - 确认 profile probe CCE 文件、注册宏、kernel 注册和 stats layout 约束。
- `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`
  - 确认 CLI、multi-host script、README、Perfetto host label 等关键字符串存在。
- `tests/collectives/unit/test_tilexr_perf_trace_layout.cpp`
  - 锁定 trace header/stats ABI 和 32-byte 对齐。
- `tests/collectives/unit/test_prepare_host_launch_context.cpp`
  - 覆盖 stats offset 和 disable kernel profiling 环境变量。
- `tests/collectives/unit/test_collective_perf_session.cpp`
  - 覆盖 incomplete report 写出。
- `tests/collectives/unit/test_collective_perf_report.cpp`
  - 覆盖 count=0 stats 过滤和 aux 字段写出。

## 提交前 staging 清单

如果后续要提交“多机性能分析模块跑通”这一组改动，建议至少包含下面这些文件。这里按 git 状态拆分，方便提交前直接核对。

新增文件必须保留：

- `docs/multi-host-profiling-report.md`
- `src/collectives/kernels/kernels/lcal_profile_probe.cce`
- `tests/collectives/run_collective_perf_multihost.sh`

已修改文件必须保留：

- `src/collectives/host/collective_kernel.cpp`
- `src/collectives/host/collective_utils.cpp`
- `src/collectives/host/collective_utils.h`
- `src/collectives/host/perf_trace_report.cpp`
- `src/collectives/host/perf_trace_report.h`
- `src/collectives/host/perf_trace_session.cpp`
- `src/collectives/host/tilexr_collectives.cpp`
- `src/collectives/kernels/CMakeLists.txt`
- `src/collectives/kernels/lccl_op.h`
- `src/collectives/kernels/perf_trace_kernel.h`
- `src/collectives/kernels/tilexr_lccl_op.cpp`
- `src/include/tilexr_collectives.h`
- `src/include/tilexr_collectives_perf.h`
- `src/include/tilexr_perf_trace.h`
- `src/include/tilexr_types.h`
- `tests/collectives/CMakeLists.txt`
- `tests/collectives/README.md`
- `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`
- `tests/collectives/tilexr_collective_profile_report.py`
- `tests/collectives/unit/test_collective_perf_report.cpp`
- `tests/collectives/unit/test_collective_perf_session.cpp`
- `tests/collectives/unit/test_collective_profile_report.py`
- `tests/collectives/unit/test_prepare_host_launch_context.cpp`
- `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`
- `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`
- `tests/collectives/unit/test_tilexr_perf_trace_layout.cpp`

这组文件形成闭环：算子枚举/API、host launch、device kernel、trace ABI、报告聚合、multi-host 启动脚本、README 和回归测试都必须同时存在。只提交其中一部分，容易出现编译可过但远端跑不通，或者能跑但 HTML/Perfetto 缺少 rank/kernel 粒度信息。

## 不应提交的内容

以下文件是同步/调试临时产物，不属于代码改动，应在提交前删除或保持 untracked：

- `.tilexr_profile_sync.tar.gz`
- `.tilexr_profile_sync.tar.gz.b64`
- `.tilexr_profile_probe_sync.tar.gz`
- `.tilexr_profile_probe_sync.tar.gz.b64`
- `.tilexr_profile_probe_sync.ascii.b64`
- `.tilexr_profile_probe_fix2.tar.gz`
- `.tilexr_profile_probe_fix2.ascii.b64`
- `.tilexr_profile_probe_fix3.tar.gz`
- `.tilexr_profile_probe_fix3.ascii.b64`

运行报告目录也不建议进代码提交，除非后续明确要提交样例报告：

- `run/prof/collectives-2host-profile-probe-62-70*`
- `run/prof/collectives-2host-profile-probe-ab-profile*`

## 已验证命令

本地报告聚合测试：

```bash
cd C:\TileXR
python -m unittest tests.collectives.unit.test_collective_profile_report -v
```

62/70 远端构建：

```bash
cd /home/l00929943/TileXR
source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pt311 2>/dev/null || true
source /usr/local/Ascend/cann/set_env.sh
cmake --build build-profile-950 --target \
  test_tilexr_collectives_kernel_ownership \
  test_tilexr_collectives_tools_sources \
  tilexr_collective_perf -j8
```

62/70 远端 source guard：

```bash
cd /home/l00929943/TileXR/build-profile-950/tests/collectives
./test_tilexr_collectives_kernel_ownership
./test_tilexr_collectives_tools_sources
```

2 机 profiling 运行：

```bash
cd /home/l00929943/TileXR
source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pt311 2>/dev/null || true
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$(pwd)/build-profile-950/src/collectives:$(pwd)/build-profile-950/src/comm:${LD_LIBRARY_PATH:-}
export TILEXR_MULTIHOST_PEERS='0,root@141.62.24.62,141.62.24.62,0;1,root@141.62.24.70,141.62.24.70,0'
export TILEXR_COMM_ID='141.62.24.62:10067'
export TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=300
bash tests/collectives/run_collective_perf_multihost.sh \
  /home/l00929943/TileXR/run/prof/collectives-2host-profile-probe-62-70-kernel-align \
  /home/l00929943/TileXR/build-profile-950/tests/collectives \
  --op profile-probe --min-bytes 4096 --max-bytes 4096 \
  --iters 2 --warmup-iters 0 --datatype int32 --check 1 \
  --profile 1 --profile-sample-every 1 --profile-ai-prompt 1
```

## 当前验证结论

最新 2 机结果：

- `diagnostics=[]`。
- `op_name=TileXRProfileProbe`。
- `rank_size=2`。
- `launch_ids=[0,1]`。
- `bars=112`。
- 每个 rank/launch 都包含完整 7 个 kernel stage。
- `perfetto_trace.json` 中每个 rank/launch/stage 都有 duration event。
- `Rank-Level Summary` 示例：
  - rank0 avg kernel `93.740 us`，max `112.580 us`。
  - rank1 avg kernel `92.170 us`，max `106.740 us`。

profiling on/off 对照：

- `--profile 0`：`errors=0`，host avg `9.607 us`。
- `--profile 1`：`errors=0`，host avg `393.650 us`，trace 完整。
- 结论：profiling 会明显放大 host 端统计耗时，但没有导致算子校验失败、trace incomplete 或 kernel stage 缺失。

plog 中可见 HCCP deinit/unimport ERROR 和 UDMA topology fallback WARN；这些在 profiling on/off 都存在，且 rank 日志 `errors=0`，目前判断为退出/资源回收阶段或 UDMA/HCCP 清理路径噪声，不是多机 profiling 模块引入的执行异常。

## 已知限制

- `profile-probe` 不是跨机 allgather 数据面，不报告跨机通信带宽。
- 当前 report 将不同 rank/launch 的时间线分别归一化，不假设两台机器的 device cycle offset 同步。
- 62 机器系统时间异常，会影响构建时间戳和 plog 文件名；必要时用 `touch` 关键源文件强制重编。
- `run_collective_perf_multihost.sh` 当前面向 profiling run；若 `--profile 0`，rank profile 目录不会生成，脚本的回收阶段会失败。做 profiling off 对照时建议手动启动 rank 或后续单独增强脚本兼容。

## 提交建议

提交多机性能分析模块时建议包含：

- 上述 `profile-probe` 算子闭包。
- kernel trace 数据面修复。
- `tilexr_collective_perf` socket/profile-probe/host_info/incomplete report 支持。
- `run_collective_perf_multihost.sh`。
- `tilexr_collective_profile_report.py` 的 host/rank/Perfetto/incomplete 聚合能力。
- README 和单测/source guard。
- 本文档 `docs/multi-host-profiling-report.md`。

提交前建议清理 `.tilexr_*` 临时包，并确认没有把 `run/prof` 运行产物加入 git。
