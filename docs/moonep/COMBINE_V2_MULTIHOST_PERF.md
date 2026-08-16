# MoonEP Combine V2 多机性能测试流程

本文只记录可复用的部署、编译、空闲检测、测试和结果判定流程。profiling 构建与
Chrome Trace JSON 导出见
[COMBINE_V2_PROFILING_TRACE.md](COMBINE_V2_PROFILING_TRACE.md)。

## 1. 基准口径

常用 128P case：

```text
world_size=128, BS=8192, K=16, H=3584, ExpNum=256, BF16
warmup=0, iterations=80, reduce=disabled
```

`K=16` 和 `BF16` 由 benchmark 固定。省略 `--reduce-hidden` 即测试 no-reduce。

性能口径是先对每个 rank 的 80 轮耗时求均值，再取 128 个 rank 均值中的最大值
作为整组延迟 `max_ms`。单 rank 等效算法数据量和带宽为：

```text
bytes = BS * K * H * sizeof(BF16)
      = 8192 * 16 * 3584 * 2
      = 939,524,096 bytes

alg_bw_GBps = 939.524096 / max_ms
```

## 2. 目录与 Hostfile

每次测试使用独立目录，避免复用其他任务的源码、产物、日志和 PID 文件：

```text
REMOTE_ROOT/
|-- source/
|-- build/
|-- install/
|-- logs/
`-- hostfile
```

hostfile 每行是 `IP:slots`。128P 使用 16 台服务器，每台 8 个 rank；第一行是
bootstrap 和任务控制节点：

```text
PRIMARY_IP:8
WORKER_IP:8
...
```

hostfile 顺序决定 global rank；每台服务器的 `device` 等于 local rank `0-7`。
编译和运行必须显式使用目标环境实际安装的同一套 CANN。

## 3. 部署源码

在本地 Git Bash 中创建任务专属 Mutagen 会话，只把源码单向同步到主节点：

```bash
SESSION=tilexr-combine-v2-<task>
PRIMARY_HOST=<primary-ip>
REMOTE_ROOT=/home/h00580772/tilexr_combine_v2_<task>

mutagen sync create \
  --name "${SESSION}" \
  --mode one-way-safe \
  --ignore-vcs \
  --ignore '/build*' \
  --ignore '/install*' \
  --ignore '/artifacts*' \
  --compression zstandard \
  'D:/3_codex/tileXR' \
  "root@${PRIMARY_HOST}:${REMOTE_ROOT}/source"

mutagen sync flush "${SESSION}"
```

忽略规则必须锚定仓库根目录；未锚定的 `build_*` 会误排除
`tools/moonep/build_combine_v2_perf.sh`。

不要改动或终止其他 Mutagen 会话。任务完成后只关闭本会话：

```bash
mutagen sync terminate "${SESSION}"
```

## 4. 编译与运行时同步

将命令写入任务专属 Bash 脚本，再通过 SSH 在主节点执行。先编译，不启动 NPU
进程：

```bash
bash "${REMOTE_ROOT}/source/tools/moonep/build_combine_v2_perf.sh" \
  --source-dir "${REMOTE_ROOT}/source" \
  --build-dir "${REMOTE_ROOT}/build" \
  --install-dir "${REMOTE_ROOT}/install" \
  --cann-path "${CANN_PATH}" \
  --jobs 16
```

确定本轮 hostfile 后，从主节点向所有计算节点平铺同步运行产物：

```bash
bash "${REMOTE_ROOT}/source/tools/moonep/sync_combine_v2_perf_runtime.sh" \
  --hostfile "${REMOTE_ROOT}/hostfile" \
  --install-dir "${REMOTE_ROOT}/install" \
  --ssh-user root
```

每台节点都应输出 `SHA256 verified`。源码不需要同步到 worker。

节点缺少 `/etc/hccl_rootinfo.json` 时，在该节点生成并安装：

```bash
tmp=$(mktemp /tmp/hccl_rootinfo.XXXXXX)
mindcluster-tools rootinfo --output "${tmp}"
install -m 0644 "${tmp}" /etc/hccl_rootinfo.json
rm -f "${tmp}"
```

启动前用 `test -f /etc/hccl_rootinfo.json && test -s /etc/hccl_rootinfo.json`
确认它是非空普通文件。若该路径被误建为空目录，C++ 文件读取会抛出
`basic_filebuf::underflow ... Is a directory` 并使 rank abort；确认目录为空后删除并
重新生成文件。

工具缺失时安装 `unofficial-ascend-tools==0.0.7rc2`。覆盖已有 rootinfo 前先保存
原文件，避免破坏共享环境配置。

## 5. 识别共享环境空闲状态

在柜 8 CPU1 上可用维护脚本快速查看 15/9/4/7 号柜：

```bash
bash "${REMOTE_ROOT}/source/scripts/watch_cab15_9_4_7_npus.sh" --once
```

每柜显示 8 台服务器、每台 8 个字符：`0` 为空闲，`1` 为有进程，`?` 为探测
失败。`Alarm` 状态本身不表示卡不可用。

watcher 只判断是否有进程，不能区分进程类型。选择候选柜后，以 `npu-smi info`
进程表或 launcher 预检作最终判定：

- 无 NPU 进程：可测试。
- 只有 `tilexr_*` 进程：允许共用，可直接测试。
- `python` 或其他通信测试进程：不可共用，每 15 秒复查。
- `?`、SSH 失败或 `npu-smi` 失败：不可判定为空闲。
- 等待 120 秒仍未满足条件：停止本轮，不启动 rank，等待下次唤起。

共享环境禁止 `pkill`，也不要清理其他任务的 PID、端口、日志或 Mutagen 会话。

## 6. 启动 128P 测试

环境可用后，在主节点执行 launcher。以下命令关闭 profiling、关闭 reduce，且不
预热：

```bash
LOG_FILE="${REMOTE_ROOT}/logs/combine_v2_128p_$(date +%Y%m%d_%H%M%S).log"

bash "${REMOTE_ROOT}/source/tools/moonep/run_combine_v2_perf_multihost.sh" \
  --hostfile "${REMOTE_ROOT}/hostfile" \
  --install-dir "${REMOTE_ROOT}/install" \
  --cann-path "${CANN_PATH}" \
  --ssh-user root \
  --bs 8192 \
  --warmup 0 \
  --iterations 80 \
  --experts 256 \
  --hidden-size 3584 \
  --comm-domain 141 \
  --comm-id "${PRIMARY_HOST}:10067" \
  --wait-seconds 120 \
  --retry-seconds 15 \
  --timeout 600 \
  --log-file "${LOG_FILE}"
```

并发任务必须使用不同的 bootstrap 端口；launcher 还会使用由该端口推导出的
barrier 端口。不要添加 `--skip-npu-preflight`，除非本轮刚刚人工检查过 hostfile
中的全部节点。

128P launcher 会为每台节点预建一条 SSH ControlMaster，并让本机 8 个 rank 复用；
不要移除此逻辑，否则并发建连可能触发共享节点的 `MaxStartups`，表现为固定 rank
以 SSH `status 255` 退出。

## 7. 结果判定

正常聚合行如下：

```text
COMBINE_V2_PERF ... max_ms=<slowest-rank-mean> max_alg_bw_GBps=<bandwidth> ...
```

128P、80 轮、单 BS 的完整数据必须包含：

```text
COMBINE_V2_SAMPLE:    128 * 80 = 10240 条
COMBINE_V2_RANK_PERF: 128 条
```

快速计数：

```bash
awk '
  /^COMBINE_V2_SAMPLE / { samples++ }
  /^COMBINE_V2_RANK_PERF / { ranks++ }
  END { print "samples=" samples + 0, "rank_perf=" ranks + 0 }
' "${LOG_FILE}.ranks"/rank_*.log
```

正确性状态与性能采集相互独立。临时优化版本即使输出 `self_only_failed` 或
`failed`，只要上述记录完整，性能数据仍有效并应输出；报告中保留实际
`correctness` 状态即可。

旧测试产物如果出现 `barrier failed after all benchmark cases`，但已有
`10240/10240` 样本和 `128/128` rank 汇总，则本轮性能数据完整。原因是复用同一
端口时，旧 listener 在逐个 release 之后才关闭，快 rank 可能误连上一轮；当前
实现会先关闭本轮 listener，再 release 全部客户端。需要从旧 rank 日志重新聚合
时使用：

```bash
awk -v expected_ranks=128 -v expected_iterations=80 -v bs=8192 \
  -f "${REMOTE_ROOT}/source/tools/moonep/configs/aggregate_combine_v2_rank_samples.awk" \
  "${LOG_FILE}.ranks"/rank_*.log
```

零样本时不报告性能：SSH `status 255` 表示 rank 启动失败；`TsdProcessOpen failed:
31` 或 `UDMA init failed: -4` 表示资源尚未释放。两种情况都等待环境恢复后再试，
不终止共享任务。
