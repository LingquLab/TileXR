# TileXR 多机性能分析模块使用说明

## 适用范围

本文档对应上交分支 `lhy-multihost-profiling`，用于在 62/70 两台 Ascend 机器上跑通 2 机 kernel-level profiling，并生成统一的 HTML 报告和 `ui.perfetto.dev` 可打开的 `perfetto_trace.json`。

当前推荐先使用 `profile-probe` 算子做多机 profiling 链路验证。它会初始化 ACL 和 socket communicator，并在每个 rank 上执行真实 AIV kernel，采集与单机 profiling 对齐的 kernel stage：

- `kernel_total`
- `chunk_total`
- `post_sync`
- `local_input_to_ipc`
- `flag_poll_wait`
- `peer_ipc_to_output`
- `chunk_barrier`

注意：`profile-probe` 是 profiling/report smoke 算子，只做本地 GM -> UB -> GM copy，不访问跨机 `peerMems[]`，不能作为跨机 allgather 带宽结果。如果需要看真实跨机 collective 数据面，需要在该链路基础上继续接入对应 collective kernel。

## 分支与代码

本地基本盘：

```powershell
cd C:\TileXR
git switch lhy-multihost-profiling
```

建议提交/同步到远端的核心文件包括：

- `src/collectives/host/*perf_trace*`
- `src/collectives/host/tilexr_collectives.cpp`
- `src/collectives/host/collective_kernel.cpp`
- `src/collectives/kernels/perf_trace_kernel.h`
- `src/collectives/kernels/kernels/lcal_profile_probe.cce`
- `src/include/tilexr_collectives*.h`
- `src/include/tilexr_perf_trace.h`
- `src/include/tilexr_types.h`
- `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`
- `tests/collectives/tilexr_collective_profile_report.py`
- `tests/collectives/run_collective_perf_multihost.sh`
- `tests/collectives/unit/*profile*`
- `tests/collectives/README.md`
- `docs/multi-host-profiling-report.md`
- `docs/multihost-profiling-usage.md`

不要提交根目录 `.tilexr_*` 同步包，也不要提交 `run/prof/` 运行产物。

## 远端环境准备

两台机器默认路径：

- 62：`root@141.62.24.62:/home/l00929943/TileXR`
- 70：`root@141.62.24.70:/home/l00929943/TileXR`

每台机器进入仓库后加载环境：

```bash
cd /home/l00929943/TileXR
source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pt311 2>/dev/null || true
source /usr/local/Ascend/cann/set_env.sh
```

如果 62 的系统时间异常导致增量编译没有重新生成 kernel，可以在远端执行：

```bash
touch src/collectives/kernels/kernels/lcal_profile_probe.cce
touch src/collectives/kernels/perf_trace_kernel.h
```

## 构建

在 62 和 70 两边都执行：

```bash
cd /home/l00929943/TileXR
source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pt311 2>/dev/null || true
source /usr/local/Ascend/cann/set_env.sh

cmake -S . -B build-profile-950 \
  -DTILEXR_BUILD_COLLECTIVES=ON \
  -DTILEXR_BUILD_TESTS=ON \
  -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON \
  -DBUILD_TESTING=OFF

cmake --build build-profile-950 --target \
  test_tilexr_collectives_kernel_ownership \
  test_tilexr_collectives_tools_sources \
  tilexr_collective_perf -j8
```

建议先跑 source guard：

```bash
cd /home/l00929943/TileXR/build-profile-950/tests/collectives
./test_tilexr_collectives_kernel_ownership
./test_tilexr_collectives_tools_sources
```

## 2 机 profiling 运行

建议从 62 发起，`TILEXR_COMM_ID` 使用 62 的 IP 和一个空闲端口：

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
  /home/l00929943/TileXR/run/prof/collectives-2host-profile-probe-62-70 \
  /home/l00929943/TileXR/build-profile-950/tests/collectives \
  --op profile-probe --min-bytes 4096 --max-bytes 4096 \
  --iters 2 --warmup-iters 0 --datatype int32 --check 1 \
  --profile 1 --profile-sample-every 1 --profile-ai-prompt 1
```

`TILEXR_MULTIHOST_PEERS` 格式为：

```text
rank,ssh_target,host_ip,device_id;rank,ssh_target,host_ip,device_id
```

例如 `0,root@141.62.24.62,141.62.24.62,0` 表示 rank0 通过 `root@141.62.24.62` 登录，在 host IP `141.62.24.62` 上使用 device 0。

## 产物说明

运行完成后，profile 根目录会包含：

```text
run/prof/collectives-2host-profile-probe-62-70/
  report.html
  perfetto_trace.json
  trace_index.json
  analysis.md
  ai_prompt.md
  rank0/
    host_info.json
    launch0/trace.json
    launch0/report.html
  rank1/
    host_info.json
    launch0/trace.json
    launch0/report.html
  multihost_rank0.log
  multihost_rank1.log
  plog/
```

重点看两个聚合产物：

- `report.html`：本地浏览器直接打开，先看 `Rank-Level Summary` 是否有慢 rank，再看 timeline 和 per-launch drilldown。
- `perfetto_trace.json`：上传到 `https://ui.perfetto.dev`，按 `rank@host`、`launch`、`kernel_total` 搜索定位慢 rank/慢 stage。

Perfetto 事件命名示例：

```text
launch0/rank0@141.62.24.62/kernel_total
launch0/rank1@141.62.24.70/kernel_total
launch0/rank1@141.62.24.70/flag_poll_wait
```

每个 rank 还会有 `launch_windows` thread，用于在 Perfetto 中对齐查看每次 launch。

## 本地拉回查看

如果需要从 62 拉回到 Windows：

```powershell
scp -r root@141.62.24.62:/home/l00929943/TileXR/run/prof/collectives-2host-profile-probe-62-70 C:\TileXR\run\prof\
```

然后打开：

```text
C:\TileXR\run\prof\collectives-2host-profile-probe-62-70\report.html
C:\TileXR\run\prof\collectives-2host-profile-probe-62-70\perfetto_trace.json
```

## 重新生成聚合报告

如果 `rank*/launch*/trace.json` 已经存在，只想重新生成 HTML/Perfetto：

```bash
cd /home/l00929943/TileXR
python3 tests/collectives/tilexr_collective_profile_report.py \
  /home/l00929943/TileXR/run/prof/collectives-2host-profile-probe-62-70 \
  --warmup-iters 0 \
  --iters 2 \
  --profile-sample-every 1 \
  --emit-ai-prompt
```

## 判断结果是否正常

一次正常的 `profile-probe` 结果应满足：

- `rank_size=2`。
- `diagnostics=[]`，或没有影响聚合的 fatal diagnostic。
- 每个 rank/launch 都包含 7 个 kernel stage。
- HTML 的 `Rank-Level Summary` 能看到每个 rank 的 avg/max kernel us。
- Perfetto 中能搜索到 `rank0@.../kernel_total` 和 `rank1@.../kernel_total`。
- `tilexr_collective_perf` 日志中 `errors=0`。

profiling 会明显放大 host 端 `avg(us)`，这是因为采集、同步、copy-back 和报告写出都在测量路径附近发生。判断算子是否异常时优先看 `errors=0`、trace 是否完整、kernel stage 是否齐全，以及 profiling on/off 是否都存在相同的退出阶段 plog 噪声。

## 常见问题

### SSH 能连但脚本失败

确认两台机器都能免密互连，并且 `TILEXR_MULTIHOST_PEERS` 里的 `ssh_target` 可以从发起机器直接登录。

### `aclInit` 或 `ascend_hal` 相关失败

确认执行过：

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$(pwd)/build-profile-950/src/collectives:$(pwd)/build-profile-950/src/comm:${LD_LIBRARY_PATH:-}
```

driver lib 需要排在 CANN stub lib 前面，避免误加载 stub `libascend_hal.so`。

### 找 plog

可以在 profile 目录或仓库下使用：

```bash
find $PWD -name "plog"
```

本脚本会设置：

```bash
ASCEND_PROCESS_LOG_PATH="${profile_dir}/plog/rank${rank}"
```

### `--profile 0` 对照

当前 `run_collective_perf_multihost.sh` 面向 profiling run。`--profile 0` 不会生成 `rank<N>/` profile 目录，脚本回收阶段可能失败。需要做 profiling off 对照时，建议临时手动启动两个 rank，或后续单独增强脚本兼容 profile-off 模式。

## 当前限制

- 当前已跑通并建议上交的是多机 multi-rank kernel-level profiling 链路，不是多 ACL stream 并发性能分析工具。
- `profile-probe` 不代表跨机 allgather 数据面性能。
- 跨机器 device cycle 不假设同步，因此聚合报告按 rank/launch 独立归一化时间线。
- 多机真实 collective 数据面 profiling 需要继续把同一套 trace 数据面接入跨机 collective kernel。
