# UDMA Grouped AllToAll 修改与测试指南

本文供新的 Codex 会话接手 `grouped AllToAll` 修改、部署、物理 `2x8` 测试和
trace 分析时使用。目标是让一次实验只改变一个变量，并保留可复现的提交、bundle、
日志和 trace。

## 1. 新会话先做什么

在仓库根目录开始，依次读取：

```text
AGENTS.md
CLAUDE.md
docs/udma-grouped-alltoall-modify-test-guide.md
```

然后确认工作区和基线：

```powershell
git status --short --branch
git log -5 --oneline
git rev-parse HEAD
```

不要清理用户已有的未跟踪文件，不要覆盖远端未提交修改。修改前记录：

- 本地分支和完整 commit ID；
- CANN 路径；
- 两台机器和 rank 分配；
- 输入大小、chunk、warmup、repeat；
- route policy、QP 数、copyout worker 数；
- combined 或 staged 模式。

## 2. 当前测试拓扑

| 项目 | 值 |
|---|---|
| rank 0..7 | `141.61.50.31` |
| rank 8..15 | `141.61.49.223` |
| 构建机 | `141.61.49.223` |
| 远端目录 | `/home/h30059441/tilexr_grouped_alltoall_b101` |
| CANN | `/home/pkg/b101/cann` |
| rank size | 16，物理 `2x8` |
| 每 peer payload | 8 MiB |
| 每 rank input/output | 128 MiB |
| 正式测试 | warmup 5，repeat 50 |
| 单个 rank 超时 | `timeout 60s` |

不要把 SSH 密码写入仓库、脚本或本文。优先使用 SSH key；否则由操作者在会话中提供。

## 3. 算法和内存约束

### 3.1 数据流

每个 peer 的基本流程是：

```text
sender input slice
  -> UDMA payload PUT + ready signal
  -> same-peer/same-QP Quiet

receiver wait ready signal
  -> registered receive buffer
  -> local MTE copy
  -> output slice
```

第一版协议没有 ACK，也没有 device `SyncAll`。一次 kernel invocation 执行一次
AllToAll；registered receive buffer 使用两个完整 payload plane 做 ping-pong。

必须保持以下正确性约束：

1. `rankSize` 当前只支持 `N * 8`，范围为 8..128。
2. 每个 source rank 在接收区有独立 slot，不循环复用 peer slot。
3. payload 和 ready 必须落在同一目标 rank 的 registered GM。
4. 单 QP 模式下，payload、ready 和 Quiet 必须使用同一个 QP。
5. ready 可见时，对应 payload 必须已经对接收端 GM 可见。
6. ping-pong slot 由 `invocationId & 1` 选择；复用前必须保证上次 invocation 完成。
7. 修改 route 过滤时，send 和 receive 必须使用完全对称的 peer 集合。
8. 不要为了性能删除超时、Quiet 状态检查或最终输出正确性检查。

### 3.2 核、peer 和 Jetty

发送侧固定使用 core 0..15。每个 group lane 计算一个 peer。对于 `rankSize=16`，
只有一个 group，其中 15 个 lane 有效，重复的对径 peer lane 无效。

以 rank0 为例：

```text
core0..7  -> peer1..8
core8     -> peer15
core9     -> peer14
...
core14    -> peer9
core15    -> invalid
```

Jetty/WQ 由 `(peer, qpIdx)` 唯一确定：

```text
queue slot = peer * qpNum + qpIdx
```

因此两个 core 即使都记录为 QP0，只要 peer 不同，就不是同一个 Jetty。当前
`TILEXR_UDMA_QP_NUM=4`、两条跨节点 route 时，每个跨节点 peer 创建 8 个 Jetty：

```text
primary route:   QP0..3
secondary route: QP4..7
```

当前 kernel 对一个 peer 只选一个 Jetty：primary 使用 QP0，secondary 使用 QP4。
每张卡对远端节点的 8 个 peer 按 6:2 分配，即 6 个 peer-specific QP0 Jetty 和
2 个 peer-specific QP4 Jetty。

### 3.3 route stage

默认 `TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES=0`，完整流程一次执行，产生一个
combined trace。

诊断模式设为 1 后依次运行四个独立 stage：

| stage | 内容 |
|---|---|
| `local-send` | 节点内 UDMA send + Quiet，不执行 copyout |
| `local-copy` | 节点内 registered buffer 到 output，不等待 signal |
| `primary` | 跨节点 primary/6口 send、wait、copyout |
| `secondary` | 跨节点 secondary/2口 send、wait、copyout |

staged 模式用于定位瓶颈。它的 stage 耗时之和不是 route 同时工作时的端到端带宽。

## 4. 关键代码

| 文件 | 作用 |
|---|---|
| `tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp` | 32核 grouped kernel、send/wait/copyout、trace 打点 |
| `tests/udma/demo/tilexr_udma_alltoall_group_layout.h` | group、peer、ping-pong 和 registered memory 布局 |
| `tests/udma/demo/tilexr_udma_alltoall_group_route.h` | local/primary/secondary 分类与 QP 选择 |
| `tests/udma/demo/tilexr_udma_alltoall_group_trace.h` | 8 MiB trace 布局 |
| `tests/udma/demo/tilexr_udma_demo.cpp` | Host 分配、注册、warmup/repeat、staged 调度、校验 |
| `src/include/tilexr_udma.h` | device UDMA PUT、PUT+signal、Quiet API |
| `src/comm/udma/tilexr_udma_transport.cpp` | EID、route、Jetty/QP 创建和 device image |
| `tests/udma/demo/tilexr_udma_alltoall_group_trace_to_chrome.py` | raw trace 转 Chrome trace |

优先扩展现有 helper，不要在 kernel 和 Host 各复制一份 route 或 peer 规则。

## 5. 修改和本地验证

一次实验只改变一个因素，例如：

- `copyoutWorkers=8/16`；
- combined 与 staged；
- 单 peer 单 Jetty与多 Jetty；
- local/primary/secondary 的调度方式；
- chunk 或 pass 数。

修改后至少运行：

```powershell
git diff --check

# 如果已有本地构建产物
.\tmp\test_group_stage_iteration.exe

python -m unittest `
  tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
```

Linux 构建机还要运行：

```bash
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_group_layout
./tests/udma/install_b101/bin/test_tilexr_udma_alltoall_layout
./tests/udma/install_b101/bin/test_tilexr_udma_transport_layout
python3 -m unittest \
  tests.udma.unit.test_tilexr_udma_alltoall_group_trace_to_chrome -v
```

若修改 Jetty/QP 映射，增加聚焦单测，至少覆盖：

- 同一个 peer 的 QP 集合；
- 不同 peer 不共享 queue slot；
- primary 与 secondary 不串 route；
- 末尾不足16个 peer 的 group；
- 对径 peer 只出现一次；
- send/receive stage 过滤对称。

## 6. 提交、bundle 和部署

用户的既定流程是：先提交，再生成完整 bundle，再部署验证。

```powershell
git status --short
git diff --check
git add <本次修改文件>
git commit -m "perf(udma): <本次单变量实验>"

$bundle = "tmp/udma-grouped-alltoall-$(git rev-parse --short HEAD).bundle"
git bundle create $bundle HEAD
git bundle verify $bundle
Get-FileHash $bundle -Algorithm SHA256

scp $bundle root@141.61.50.31:/tmp/
scp $bundle root@141.61.49.223:/tmp/
```

两台机器更新前先检查远端工作区。若有未提交修改，不要覆盖：

```bash
cd /home/h30059441/tilexr_grouped_alltoall_b101
git status --short
```

工作区干净时，在两台机器分别执行：

```bash
bundle=/tmp/udma-grouped-alltoall-<short-head>.bundle
cd /home/h30059441/tilexr_grouped_alltoall_b101
git fetch "$bundle" HEAD
git checkout --detach FETCH_HEAD
git rev-parse HEAD
```

两台机器的 HEAD 必须与本地完整 commit ID 一致。

## 7. b101 构建

`141.61.50.31` 当前不负责构建。在 `141.61.49.223` 构建：

```bash
cd /home/h30059441/tilexr_grouped_alltoall_b101
source scripts/common_env.sh
source /home/pkg/b101/cann/set_env.sh

export ASCEND_HOME_PATH=/home/pkg/b101/cann
export ASCEND_HOME_DIR=/home/pkg/b101/cann
export ASCEND_TOOLKIT_HOME=/home/pkg/b101/cann
export PATH=/home/pkg/b101/cann/bin:$PATH
export LD_LIBRARY_PATH=/home/pkg/b101/cann/aarch64-linux/lib64:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$PWD/install
cmake --build build -j8
cmake --install build

cmake -S tests/udma -B tests/udma/build_b101 \
  -DCMAKE_INSTALL_PREFIX=$PWD/tests/udma/install_b101 \
  -DTILEXR_UDMA_DEMO_SOC_TYPE=Ascend950
cmake --build tests/udma/build_b101 -j8
cmake --install tests/udma/build_b101
```

不要让运行时加载 CANN `devlib` 中的 stub `libascend_hal.so`：

```bash
readelf -d build/src/comm/libtile-comm.so | grep -E 'RPATH|RUNPATH' || true
ldd tests/udma/install_b101/bin/tilexr_udma_demo | \
  grep -E 'lib(tile-comm|ascendcl|runtime|ascend_hal)'
```

将相同产物复制到 `141.61.50.31` 对应路径：

```text
install/lib64/libtile-comm.so
tests/udma/install_b101/lib/libtilexr_udma_demo_kernel.so
tests/udma/install_b101/bin/tilexr_udma_demo
```

复制后在两台机器执行 `sha256sum`，三个文件必须逐个一致。

## 8. 物理 2x8 测试

### 8.1 公共环境

两台机器使用相同变量。每轮必须更换 `TILEXR_COMM_ID` 端口和结果目录：

```bash
cd /home/h30059441/tilexr_grouped_alltoall_b101
source /home/pkg/b101/cann/set_env.sh

export ASCEND_HOME_PATH=/home/pkg/b101/cann
export PATH=/home/pkg/b101/cann/bin:$PATH
export LD_LIBRARY_PATH=$PWD/tests/udma/install_b101/lib:$PWD/install/lib64:\
/home/pkg/b101/cann/aarch64-linux/lib64:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64

export TILEXR_COMM_ID=141.61.50.31:<new-port>
export TILEXR_DEMO_BARRIER_HOST=141.61.50.31
export TILEXR_IPC_PID_MODE=pid
export TILEXR_UDMA_ROUTE_POLICY=all
export TILEXR_UDMA_QP_NUM=4

export TILEXR_DEMO_ALLTOALL_WARMUP=5
export TILEXR_DEMO_ALLTOALL_REPEAT=50
export TILEXR_DEMO_ALLTOALL_GROUP_CHUNK_ELEMENTS=2097152
export TILEXR_DEMO_ALLTOALL_GROUP_COPYOUT_WORKERS=16

export TILEXR_UDMA_GROUP_TRACE=1
export TILEXR_UDMA_GROUP_TRACE_DIR=/home/h30059441/<unique-result-dir>
mkdir -p "$TILEXR_UDMA_GROUP_TRACE_DIR"
```

这里 `2097152 int32 = 8 MiB/peer`。命令中的 `16 * 8 MiB` 对应每 rank
128 MiB input 和 128 MiB output。

### 8.2 combined 基线

```bash
export TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES=0
```

在 `.50.31` 并发启动 rank 0..7，在 `.49.223` 并发启动 rank 8..15：

```bash
for rank in $(seq <first-rank> <last-rank>); do
  timeout 60s ./tests/udma/install_b101/bin/tilexr_udma_demo \
    16 "$rank" 8 2097152 8 0 \
    >"$TILEXR_UDMA_GROUP_TRACE_DIR/rank_${rank}.log" 2>&1 &
done
wait
```

两台机器必须几乎同时启动。不要先等待一台完成再启动另一台。

### 8.3 staged 定位

```bash
export TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES=1
```

使用新的端口和目录，按相同方式启动16个 rank。当前会产生四套 trace：

```text
tilexr_group_trace_local-send_rank_<rank>.bin
tilexr_group_trace_local-copy_rank_<rank>.bin
tilexr_group_trace_primary_rank_<rank>.bin
tilexr_group_trace_secondary_rank_<rank>.bin
```

staged 测试仍使用 `timeout 60s`。如果60秒不足，先定位 barrier、残留进程或环境
问题，不要直接无限增加超时。

## 9. 结果验收

先检查所有 rank 的退出和正确性，再分析性能：

```bash
grep -H 'TileXR grouped alltoall demo success' \
  "$TILEXR_UDMA_GROUP_TRACE_DIR"/rank_*.log
grep -H -E 'ERROR|failed|MISMATCH|CQ incomplete|0x302|timeout' \
  "$TILEXR_UDMA_GROUP_TRACE_DIR"/rank_*.log || true
find "$TILEXR_UDMA_GROUP_TRACE_DIR" -name '*.bin' \
  -printf '%f %s\n' | sort -V
```

通过标准：

1. 16/16 rank 正常退出并打印 success。
2. 没有 quiet error、wait timeout、mismatch、`0x302`。
3. 每个 raw trace 严格为 8,388,608 bytes。
4. combined 模式有16个 trace；当前 staged 模式有64个 trace。
5. 日志确认 `groups=1`，payload 为 134,217,728 bytes/rank。
6. 正确性通过后才能使用本轮性能数字。

正式对比至少报告：

```text
commit
CANN
环境变量差异
Host perIter mean/min/max
loop49 kernel envelope mean/min/max
各 stage mean
send-put-signal 和 send-quiet
QP/peer/Jetty 分布
```

## 10. 下载并生成 loop49 trace

将 rank 0..7 从 `.50.31`、rank 8..15 从 `.49.223` 下载到本地同一目录。

combined 转换：

```powershell
$inputs = Get-ChildItem tmp/<run>/tilexr_group_trace_rank_*.bin |
  Sort-Object { [int]($_.BaseName -replace '.*_', '') } |
  ForEach-Object FullName

python tests/udma/demo/tilexr_udma_alltoall_group_trace_to_chrome.py `
  @inputs --output tmp/<run>/all_loops.json

python tmp/extract_xy_trace_iteration.py `
  tmp/<run>/all_loops.json `
  tmp/<run>/loop49.json 49
```

staged 模式对 `local-send`、`local-copy`、`primary`、`secondary` 分别执行同样的
转换和 iteration 49 提取。文件名中的 stage 必须与16个输入 trace 一致。

Chrome Tracing/Perfetto 中重点检查：

- 每个 rank 是否有15个唯一 send peer；
- 同一个 `(rank, peer)` 是否只由一个 core 负责；
- local stage 是否全部为 QP0；
- 跨节点是否为 primary QP0 96个 rank pair、secondary QP4 32个 rank pair；
- `send-put-signal` 后是否紧跟同 peer、同 QP 的 `send-quiet`；
- receive wait 是否在对应 signal 后结束；
- 是否存在空 span、重复 peer 或缺失 peer。

不同 rank 的 device cycle 没有全局统一时钟。转换器按 rank/iteration 归一化，适合
观察单 rank 内相对流水；不要把不同 rank 的绝对 `ts` 当成严格同步时间。

## 11. 多 Jetty 实验指导

当前“每个 peer 创建多个 Jetty，但 payload 只走一个 Jetty”。验证单 rank-pair
Jetty 是否限制带宽时，不能只让不同 peer 轮换 QP0..3：不同 peer 本来已经对应
不同 Jetty，这种修改没有增加同一 rank pair 的并行度。

正确的单变量实验矩阵是：

```text
1 Jetty/peer: 当前基线
2 Jetty/peer: 同一 peer 的 chunk 分成2段并发
4 Jetty/peer: 同一 peer 的 chunk 分成4段并发
```

实现时必须解决跨 QP 完成顺序：

1. 按同 route 的 QP 集合切分 payload，不能把 primary 数据切到 secondary route。
2. 所有 payload slice 都提交后，分别 Quiet 对应 QP。
3. 只有全部 payload QP 完成后，才能发布该 `(group, pass, peer)` 的 ready。
4. ready 使用确定的 QP，并再次 Quiet；不得让多个 QP竞争写同一个 signal token。
5. 接收端仍只等待一个 ready，然后 copyout 完整 chunk。
6. trace 增加 slice/QP 信息，能够区分 payload Quiet 和 ready Quiet。

不要直接把当前 `UDMAPutSignalNbiOnQp` 复制到四个 QP。单 QP 上的
payload-before-signal 顺序不能证明跨 QP 的其他 payload 已完成。

先只运行 `local-send` stage 比较 1/2/4 Jetty，减少 copyout 干扰；确认提升后再跑
combined，验证整体收益和正确性。每档都使用同一 commit、数据规模、warmup/repeat
和空闲环境。

## 12. 常见故障

### 初始化失败或部分 rank 卡住

- 检查16个 rank 是否同时启动；
- 更换 `TILEXR_COMM_ID` 端口；
- 检查两台机器是否存在残留 `tilexr_udma_demo`；
- 检查两台机器的 HEAD、二进制 hash 和环境变量是否一致；
- 只终止本轮明确识别的进程，不要杀其他用户作业。

### `0x302`、CQ error 或方向性失败

- 检查 payload、ready、Quiet 是否使用同一目标 peer 和预期 QP；
- 检查 `/etc/hccl_rootinfo.json` 的 EID/端口映射；
- 检查 `TILEXR_UDMA_ROUTE_POLICY=all` 和 `TILEXR_UDMA_QP_NUM=4`；
- 检查运行时是否误加载 CANN `devlib` stub；
- 多 Jetty修改时优先检查跨 QP ready 是否提前发布。

### trace 正确但性能异常

- 先确认没有其他 NPU/UDMA 进程；
- 区分 Host wall、kernel envelope 和纯 `send-quiet`；
- combined 与 staged 不能直接当成同一个吞吐指标；
- 只比较相同 payload 字节数和相同方向；
- 记录 `.50.31 -> .49.223` 与反方向，不能只看双向平均。

## 13. 新 Codex 会话提示词

可以在新会话直接使用：

```text
先读取 AGENTS.md、CLAUDE.md 和
docs/udma-grouped-alltoall-modify-test-guide.md。

目标：对 grouped AllToAll 做一次单变量修改并完成物理2x8验证。
保持每 rank 128 MiB、b101 CANN、warmup5/repeat50、timeout 60s。
先检查当前分支和已有改动，不要清理无关文件。修改后运行聚焦测试，提交，
生成并验证完整 bundle，部署到 141.61.50.31 和 141.61.49.223，确认 HEAD 和
产物 hash 一致，再跑16 rank。先检查全部正确性，再提取 loop49 trace 和分析性能。
任何时候只改变一个实验变量。

本轮要验证的变量：<在这里填写，例如 1/2/4 Jetty per peer>。
```
