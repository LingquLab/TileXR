# MoonEP Combine V2 多机性能测试流程

本文只记录可复用的部署、编译、空闲检测、测试和结果判定流程。profiling 构建与
Chrome Trace JSON 导出见
[COMBINE_V2_PROFILING_TRACE.md](COMBINE_V2_PROFILING_TRACE.md)。

## 0. 0/2 柜固定快速流程

当前 0/2 柜 128P 测试使用固定目录
`/home/h00580772/tilexr_combine_v2_cab0_2_fast` 和 Mutagen 会话
`tilexr-combine-v2-cab0-2-fast`。会话建立后，每次只依次执行以下四条命令：

```bash
mutagen sync flush tilexr-combine-v2-cab0-2-fast
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/compile.sh'
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/sync_runtime.sh'
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/run.sh'
```

三个远端脚本只服务当前工程和 0/2 柜，固定使用 CPU1 `141.61.53.150`、CANN
`/home/pkg/b131/cann-9.1.0`，并执行 128P、BS8192、K16、H3584、Exp256、BF16、
warmup20、80 轮、no-reduce、关闭 profiling 的测试。环境空闲状态仍在执行四条命令前
通过外部 watcher 判断，不在 launcher 内重复检测。

### R1 固定 16 核和可重复 barrier

R1 在保留 bidirectional Ring 和 Legacy Grant 的前提下，将 Runtime blockDim 固定为 16，
并要求所有 16 个 block 以相同顺序进入 `AscendC::SyncAll<true>()`。2P-8P 的非工作核不能
在 Kernel 入口提前返回，只跳过 peer、QP、payload、Done 和 Reduce 工作。`SyncAll` 只提供
rendezvous，不提供状态归约；局部失败必须先写入 magic-tagged collective status，再由
16 核同步读取并统一继续或退出。

leading/round global barrier 使用两个 generation slot 循环复用。signal 除 generation 外还
必须携带并校验完整 `boundaryId`，否则 boundary `n` 的陈旧信号可能被 `n+2` 误认。R1
暂不启用 phase barrier，但预留其 boundary 编号，避免后续 R4 改变既有 round 编号。

2026-08-16 在 CANN 9.1、Ascend950、0/2 柜 128P 上完成 target compile 和实机验证：
BS8192、K16、H3584、Exp256、BF16、no-reduce、20 次 warmup + 80 次迭代，128/128 rank
正确性通过；平均 `4.563616 ms`，最大 `4.573830 ms`。日志为
`combine_v2_128p_noprofile_20260816_231819.log`。该结果不覆盖 32P/64P、Reduce 或 R2 之后
的 Server-Pair/64-Grant 行为。

### R2 停用 32P+ Legacy Grant

R2 在 32P/64P/128P 停止发布和等待 Legacy Ring Grant，2P-16P 保留旧路径。每个 round
的数据 CQ drain 后仍执行 global barrier，Done ordering、Ring peer 和公共 ABI 不变。

global barrier 的稳定 `boundaryId` 和两槽 ping-pong `generation` 必须解耦。R1-R3 为后续
phase barrier 预留编号，但尚不执行该 barrier；128P 的 boundary 因而从 4 直接跳到 6。
若使用 `boundaryId % 2`，这两个连续执行的 barrier 会复用同一个槽，较快 rank 可用
boundary 6 覆盖较慢 rank 尚未读取的 boundary 4 signal。停用 Legacy Grant 后该竞态表现为
一柜已等待 boundary 6、另一柜仍等待 boundary 4，最终约 10 秒超时。正确做法是按实际
执行序号选择 generation：leading 为 0，R1-R3 的 round `s` 使用 `s + 1`；signal guard
仍校验完整 `boundaryId`，用于拒绝陈旧或被覆盖的信号。

2026-08-17 在 CANN 9.1、Ascend950、0/2 柜 128P 上验证 R2：最小 BS128、1 次迭代、
no-reduce 用例由超时恢复为 `correctness=passed`；随后 BS8192、20 次 warmup + 80 次迭代
的 128/128 rank 均通过，平均 `4.466763 ms`，最大 `4.480840 ms`。完整日志为
`combine_v2_128p_noprofile_20260817_002247.log`。该证据覆盖 128P Ring、barrier-only、
no-reduce 路径，不覆盖 32P/64P、Reduce 或 Server-Pair/64-Grant。

## 1. 基准口径

常用 128P case：

```text
world_size=128, BS=8192, K=16, H=3584, ExpNum=256, BF16
warmup=0, iterations=80, reduce=disabled
```

`K=16` 和 `BF16` 由 benchmark 固定。省略 `--reduce-hidden` 即测试 no-reduce。

性能口径是在一次 rank 对齐后连续提交 80 轮，用整批 ACL Event 耗时除以 80 得到
每个 rank 的 `avg_ms`，再取 128 个 rank 平均耗时中的最大值作为整组延迟
`max_ms`。计时轮次之间没有 barrier、同步、D2H、profiling 解析或日志输出。单 rank
等效算法数据量和带宽为：

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

watcher 只判断是否有进程，不能区分进程类型。选择候选柜后，以外部
`npu-smi info` 进程表作最终判定：

- 无 NPU 进程：可测试。
- 只有 `tilexr_*` 进程：允许共用，可直接测试。
- `python` 或其他通信测试进程：不可共用，每 15 秒复查。
- `?`、SSH 失败或 `npu-smi` 失败：不可判定为空闲。
- 等待 120 秒仍未满足条件：停止本轮，不启动 rank，等待下次唤起。

launcher 不重复执行 NPU 进程预检，也不负责等待环境空闲。调用方必须在启动前通过
watcher 或等价的外部检查确认环境可用。

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
  --timeout 600 \
  --log-file "${LOG_FILE}"
```

并发任务必须使用不同的 bootstrap 端口；launcher 还会使用由该端口推导出的
barrier 端口。

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
COMBINE_V2_RANK_PERF: 128 条
```

关闭 profiling 时不输出 `COMBINE_V2_SAMPLE`。开启 profiling 时每个 rank 只为最后
一轮输出一条 `timing_source=kernel_profile` 的 sample，供最快/P50/最慢卡选择使用。

快速计数：

```bash
awk '
  /^COMBINE_V2_RANK_PERF / { ranks++ }
  END { print "rank_perf=" ranks + 0 }
' "${LOG_FILE}.ranks"/rank_*.log
```

正确性状态与性能采集相互独立。临时优化版本即使输出 `self_only_failed` 或
`failed`，只要 rank 性能记录完整，性能数据仍有效并应输出；报告中保留实际
`correctness` 状态即可。

旧测试产物如果出现 `barrier failed after all benchmark cases`，但已有完整的逐轮
样本和 `128/128` rank 汇总，则本轮性能数据完整。原因是复用同一
端口时，旧 listener 在逐个 release 之后才关闭，快 rank 可能误连上一轮；当前
实现会先关闭本轮 listener，再 release 全部客户端。需要从旧 rank 日志重新聚合
时使用：

```bash
awk -v expected_ranks=128 -v expected_iterations=80 -v bs=8192 \
  -f "${REMOTE_ROOT}/source/tools/moonep/configs/aggregate_combine_v2_rank_samples.awk" \
  "${LOG_FILE}.ranks"/rank_*.log
```

新格式中零条逐轮 sample 是正常现象，必须以 128 条 `COMBINE_V2_RANK_PERF` 判断
数据是否完整。SSH `status 255` 表示 rank 启动失败；`TsdProcessOpen failed: 31`
或 `UDMA init failed: -4` 表示资源尚未释放。两种情况都等待环境恢复后再试，不终止
共享任务。
