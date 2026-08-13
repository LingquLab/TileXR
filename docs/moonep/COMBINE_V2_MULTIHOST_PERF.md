# MoonEP Combine V2 多机通信性能测试指导

本文说明如何在多台 Ascend 服务器上构建、同步和运行 MoonEP Combine V2
单算子通信性能测试。流程不依赖 MPI，由主服务器按 hostfile 顺序通过 SSH
并发拉起所有 rank。

## 1. 当前测试范围

当前 benchmark 固定以下算子参数：

| 参数 | 当前值 | 是否可免编译修改 |
| --- | --- | --- |
| `BS` | 默认 `128` | 是，使用 `--bs` 或 `--bs-list` |
| `K` | `16` | 否，当前写在 benchmark C++ 中 |
| `H` | `3584` | 否，当前写在 benchmark C++ 中 |
| 专家总数 | 默认 `64` | 是，使用 `--experts` 或 `-Experts` |
| dtype | `BF16` | 否，当前写在 benchmark C++ 中 |
| 每台服务器 rank 数 | `1-8` | 是，由 hostfile 配置 |

`--bs-list` 会在一次进程任务中覆盖多个 BS 点位。TileXR/UDMA 只初始化一次，
workspace 按列表中的最大 BS 申请并注册一次，所有点位复用该 workspace。

### 1.1 性能结果展示约束

正式性能报告展示以下两组结果：

- `avg_ms`：先对每个 rank 除去 warmup 后的全部计时 iteration 求平均，得到每卡
  耗时，再对所有卡的耗时求算术平均。
- `max_ms`：所有卡耗时中的最大值，即最慢卡在全部计时 iteration 上的平均耗时。
- `avg_alg_bw_GBps`：完整逻辑通信数据量除以 `avg_ms`。
- `max_alg_bw_GBps`：完整逻辑通信数据量除以 `max_ms`。

不统计或展示单轮样本的 `p50/min/max`。这里的 `max_ms` 是每卡平均耗时中的最大值，
不是 80 轮单次耗时的最大值。

BF16 Combine V2 的单 rank 逻辑通信数据量和算法带宽统一按以下公式计算：

```text
data_bytes = BS * K * H * 2
avg_alg_bw_GBps = data_bytes / avg_ms / 1e6
max_alg_bw_GBps = data_bytes / max_ms / 1e6
```

同 rank、同服务器跨卡和跨服务器的数据均计入 `data_bytes`，不再分别统计卡内、
跨卡或跨机比例。算法带宽使用单 rank 完整逻辑数据量，不额外乘 world size。
`GB/s` 使用十进制单位 `1 GB = 10^9 bytes`。

## 2. 脚本职责

| 文件 | 执行位置 | 职责 |
| --- | --- | --- |
| `tools/moonep/run_combine_v2_perf_cluster.sh` | 主服务器 | 完整入口：远端编译、平铺同步、启动测试 |
| `tools/moonep/build_combine_v2_perf.sh` | 主服务器 | 编译 benchmark，整理 `bin/` 和 `lib64/`，检查 RPATH 和 MPI 依赖 |
| `tools/moonep/sync_combine_v2_perf_runtime.sh` | 主服务器 | 主服务器直接 rsync 到 hostfile 中的每台服务器并校验 SHA-256 |
| `tools/moonep/run_combine_v2_perf_multihost.sh` | 主服务器 | NPU 预检、rank 映射、SSH 拉起、日志校验和性能聚合 |
| `tests/moonep_combine_v2/demo/tilexr_moonep_combine_v2_hardware_probe.cpp` | 编译产物 | 单 rank benchmark、TCP barrier、正确性检查和 ACL Event 计时 |

同步是平铺模式：主服务器直接同步每个目标节点，不允许计算节点继续向其他
节点分层转发。

## 3. 远端目录

建议在 `/home/h00580772` 下为任务建立独立根目录：

```text
/home/h00580772/tilexr_combine_v2/
|-- source/    # Mutagen 只同步到主服务器
|-- build/     # 只在主服务器编译
|-- install/   # 主服务器向全部测试节点平铺同步
|-- logs/      # 控制日志、逐 rank 日志、NPU 快照
`-- hostfile
```

本文中的已验证 16P 环境使用：

```text
/home/h00580772/tilexr_combine_v2_16p_cdf2b01_20260811
```

## 4. 环境前提

开始前确认：

1. 本地已安装 `mutagen`、`scp` 和 `ssh`。
2. 本地能够免密 SSH 登录主服务器，并可用 `scp` 上传临时 bash 入口到测试目录。
3. 主服务器能够免密 SSH 登录 hostfile 中的每一台服务器。
4. 所有服务器存在兼容的 CANN 和驱动；CANN 路径按第 4.2 节的环境规则选择，
   不要求所有环境统一使用 B150。
5. 主服务器和计算节点安装了 `bash`、`rsync`、`sha256sum`、`timeout`、
   `ss` 和 `npu-smi`。
6. 主服务器使用 Bash 4.3 或更高版本，以支持 `wait -n`。

计算节点不需要安装 MPI。

### 4.1 环境信息来源与名称解析

测试节点必须从以下两份本地环境清单选择，二者是互相独立的环境：

| 环境 | 权威信息来源 | 节点选择规则 |
| --- | --- | --- |
| 512P 环境 | `D:\3_codex\512P环境信息.txt` | 用户指定“`xx柜`”或“`xx柜 CPUy`”时，按该文件中的柜号和 CPU 编号解析 IP |
| A10-64P 环境 | `D:\3_codex\A10-64P环境.txt` | 文件按行给出独立服务器 IP；该环境没有“柜号/CPU 编号”语义 |

例如，`0号柜 CPU1` 必须从 512P 环境解析为 `141.61.53.150`。A10-64P
环境中的 `141.61.49.226`、`141.61.49.223` 等服务器不属于任何 `xx柜`。
不得根据 A10-64P 文件的行号或排列顺序推导柜号、CPU 编号，也不得把两个环境的
节点混入同一个 hostfile，除非用户明确要求跨环境测试。

柜号不保证连续。新增或调整测试规模时应重新读取对应源文件，不在脚本或文档中
凭历史结果猜测 IP。

### 4.2 CANN 版本选择约束

CANN 版本约束按测试环境生效，不是全局约束：

| 测试环境 | 编译与运行 CANN 约束 |
| --- | --- |
| 512P 环境 9 号柜、15 号柜 | 必须使用 `/home/pkg/910_B150/cann-9.1.0` 编译，并在 launcher 中通过 `--cann-path` 使用同一路径运行 |
| A10-64P 环境，包括 `141.61.49.223` | 不要求使用 B150；使用目标服务器实际安装且与编译产物兼容的 CANN，例如 223 当前可使用 `/home/pkg/b131/cann-9.1.0` |
| 其他 512P 柜号 | 不自动套用 9/15 号柜的 B150 约束；测试前检查目标柜的实际安装路径，并显式传入 `-CannPath` 或 `--cann-path` |

不得因为某次任务要求 9/15 号柜使用 B150，就把该要求扩展到
`141.61.49.223`、`141.61.49.226` 等 A10-64P 节点。反过来，也不得在 9/15
号柜沿用 A10-64P 的 b131/b061 路径。主服务器编译时使用的 CANN 必须与
hostfile 中运行节点可用的运行时和 ABI 匹配；若不匹配，应重新编译并重新平铺
同步产物，不能直接复用旧 `install/`。

## 5. Hostfile 与 rank 映射

hostfile 每行格式为 `IP:slots`，空行和以 `#` 开头的行会被忽略：

```text
141.61.49.226:8
141.61.49.223:8
```

规则如下：

- 第一台服务器承担 rank 0、bootstrap、编译和任务控制。
- hostfile 顺序决定 global rank 顺序，不根据 hostname 重新排序。
- `global_rank = 前面节点的 slots 总和 + local_rank`。
- `device = local_rank`。
- 每个 host 只能出现一次，单机 slots 必须为 `1-8`。
- world size 必须为 `2-8`、`16`、`32`、`64` 或 `128`；这是 Combine V2
  调度支持的 rank 集合。
- 专家总数必须能被 world size 整除。`ExpNum=256, world size=128` 表示每个
  rank 对应 2 个专家。
- 每个 BS 必须能被 world size 整除。

每台服务器使用 8 个 rank 时，当前直接支持：

| 节点数 | world size | 是否支持 |
| --- | --- | --- |
| 1 | 8P | 是 |
| 2 | 16P | 是 |
| 3 | 24P | 否，24 不能整除 64 |
| 4 | 32P | 是 |
| 8 | 64P | 是 |
| 16 | 128P | 是 |

扩容时只需要在 hostfile 中按期望 rank 顺序列出对应服务器。64P 使用 8 台，
128P 使用 16 台；每台均为 8 个 rank：

```text
PRIMARY_IP:8
WORKER_1_IP:8
WORKER_2_IP:8
WORKER_3_IP:8
WORKER_4_IP:8
WORKER_5_IP:8
WORKER_6_IP:8
WORKER_7_IP:8
```

## 6. 创建 Mutagen 会话

每个主服务器和远端根目录使用独立会话，不要重定向已有的无关会话：

```bash
mutagen sync create \
  --name tilexr-combine-v2-64p \
  --mode one-way-replica \
  --ignore-vcs \
  --ignore build \
  --ignore "build_*" \
  --ignore install \
  --compression zstandard \
  "D:\3_codex\TileXR-PR-DEBUG\TileXR" \
  "root@PRIMARY_IP:/home/h00580772/tilexr_combine_v2/source"
```

运行前由本地手动 flush 源码会话：

```bash
mutagen sync flush tilexr-combine-v2-64p
```

源码只通过 Mutagen 同步到主服务器，不同步到计算节点。测试入口不再封装
Mutagen；需要先确认会话目标是 `<ssh-user>@<primary-host>:<remote-root>/source`。

## 7. 完整流程

### 7.1 16P 单 BS

先在本地仓库根目录将服务器端入口上传到主服务器测试目录，再通过 SSH 在主服务器上执行：

```bash
scp tools/moonep/run_combine_v2_perf_cluster.sh \
  root@141.61.49.226:/home/h00580772/tilexr_combine_v2_16p_cdf2b01_20260811/run_combine_v2_perf_cluster.sh

ssh root@141.61.49.226 \
  "bash /home/h00580772/tilexr_combine_v2_16p_cdf2b01_20260811/run_combine_v2_perf_cluster.sh \
    --remote-root /home/h00580772/tilexr_combine_v2_16p_cdf2b01_20260811 \
    --bs 128"
```

该命令依次执行：

1. 在主服务器配置和增量编译 benchmark。
2. 检查可执行文件没有 MPI 依赖和 build 目录 RPATH。
3. 主服务器将 `install/` 直接同步到全部节点。
4. 每台服务器校验相同的 SHA-256 manifest。
5. 对全部服务器执行 NPU 占用预检。
6. 主服务器通过 SSH 并发拉起所有 rank。
7. 收集逐 rank 结果并输出全局聚合性能。

### 7.2 多 BS 批量测试

```bash
scp tools/moonep/run_combine_v2_perf_cluster.sh \
  root@PRIMARY_IP:/home/h00580772/tilexr_combine_v2/run_combine_v2_perf_cluster.sh

ssh root@PRIMARY_IP \
  "bash /home/h00580772/tilexr_combine_v2/run_combine_v2_perf_cluster.sh \
    --remote-root /home/h00580772/tilexr_combine_v2 \
    --bs-list 128,256,512,1024,8192 \
    --experts 64 \
    --warmup 20 \
    --iterations 80"
```

`--bs` 与 `--bs-list` 互斥。没有指定时默认测试 `BS=128`。

当 `PRIMARY_IP` 和 hostfile 属于 9 号柜或 15 号柜时，上述命令必须额外传入：

```bash
--cann-path /home/pkg/910_B150/cann-9.1.0
```

在 `141.61.49.223` 等 A10-64P 节点上不要机械添加该参数，应按第 4.2 节使用
该节点实际可用且与产物匹配的 CANN。

### 7.3 常用 Bash 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--remote-root` | 必填 | 远端任务根目录 |
| `--hostfile` | `<remote-root>/hostfile` | 远端 hostfile 绝对路径 |
| `--source-dir` | `<remote-root>/source` | 远端源码目录 |
| `--build-dir` | `<remote-root>/build` | 远端构建目录 |
| `--install-dir` | `<remote-root>/install` | 远端运行产物目录 |
| `--cann-path` | 当前 `ASCEND_HOME_PATH` 或 toolkit latest | CANN 根目录；9/15 号柜必须显式改为 `/home/pkg/910_B150/cann-9.1.0`，223 不强制 B150 |
| `--bs` | `128` | 单个 BS 点位 |
| `--bs-list` | 空 | 逗号分隔的多个 BS 点位 |
| `--warmup` | `20` | 每个 BS 的预热次数 |
| `--iterations` | `80` | 每个 BS 的计时次数，也称 loop 数 |
| `--experts` | `64` | 专家总数，必须能被 world size 整除 |
| `--comm-domain` | `141` | Shared-QP 通信域 |
| `--comm-port` | `10067` | rank 0 bootstrap 端口 |
| `--wait-seconds` | `120` | NPU 最大等待时间 |
| `--retry-seconds` | `15` | NPU 占用重试间隔 |
| `--rank-timeout` | `600` | 每个 rank 的任务超时 |
| `--build-jobs` | `nproc` | 编译并发度 |
| `--log-file` | 自动生成 | 主服务器控制日志绝对路径 |

## 8. 免编译快速测试

如果只修改 BS、warmup、iterations 或通信端口，并且 `install/` 已经同步到
hostfile 中的全部服务器，可以跳过 Mutagen、编译和产物同步，直接在主服务器
执行 launcher：

```bash
bash /home/h00580772/tilexr_combine_v2_16p_cdf2b01_20260811/source/tools/moonep/run_combine_v2_perf_multihost.sh \
  --hostfile /home/h00580772/tilexr_combine_v2_16p_cdf2b01_20260811/hostfile \
  --install-dir /home/h00580772/tilexr_combine_v2_16p_cdf2b01_20260811/install \
  --cann-path /home/pkg/b061/cann-9.1.T560 \
  --ssh-user root \
  --bs-list 128,256,512,8192 \
  --experts 64 \
  --warmup 20 \
  --iterations 80 \
  --comm-id 141.61.49.226:10067 \
  --wait-seconds 120 \
  --retry-seconds 15 \
  --timeout 600
```

服务器端完整入口默认会执行一次增量 build；源码未变化时通常很快。若要严格跳过
编译，可以向完整入口传入 `--skip-build`，或直接使用上述服务器端 launcher。

上述免编译示例针对 A10-64P 的 226/223 环境，因此使用 b061。若目标是 9/15
号柜，`--cann-path` 必须改为 `/home/pkg/910_B150/cann-9.1.0`，并确认现有
`install/` 本身也是用该 B150 路径编译；否则不能跳过编译和产物同步。

以下变化不能直接复用旧产物：

- 修改 C++ benchmark、算子 Host、Kernel、CMake 或头文件。
- 修改当前硬编码的 `H/K/dtype`。
- 更换不兼容的 CANN、驱动、SoC 或 ABI。

## 9. 手工分步执行

需要排查某个阶段时，可以按以下顺序执行。

### 9.1 本地同步源码

```bash
mutagen sync flush tilexr-combine-v2-16p-cdf2b01-226
```

源码只通过 Mutagen 同步到主服务器，不同步到计算节点。

### 9.2 主服务器编译

```bash
bash ${REMOTE_ROOT}/source/tools/moonep/build_combine_v2_perf.sh \
  --source-dir ${REMOTE_ROOT}/source \
  --build-dir ${REMOTE_ROOT}/build \
  --install-dir ${REMOTE_ROOT}/install \
  --cann-path /home/pkg/b061/cann-9.1.T560 \
  --jobs 16
```

上例的 b061 路径适用于已有兼容环境，不是全局强制值。在 9/15 号柜执行时，必须
将编译参数改为：

```bash
--cann-path /home/pkg/910_B150/cann-9.1.0
```

在 `141.61.49.223` 执行时不要求 B150，应传入该服务器实际可用且与运行产物
匹配的路径，例如：

```bash
--cann-path /home/pkg/b131/cann-9.1.0
```

构建成功后，运行产物位于：

```text
${REMOTE_ROOT}/install/bin/tilexr_moonep_combine_v2_perf
${REMOTE_ROOT}/install/lib64/
```

### 9.3 主服务器平铺同步

```bash
bash ${REMOTE_ROOT}/source/tools/moonep/sync_combine_v2_perf_runtime.sh \
  --hostfile ${REMOTE_ROOT}/hostfile \
  --install-dir ${REMOTE_ROOT}/install \
  --ssh-user root
```

成功标志是 hostfile 中每台服务器均输出：

```text
<host>: SHA256 verified
```

### 9.4 主服务器启动测试

使用第 8 节的 launcher 命令。launcher 不调用 `mpirun`，也不检查任何 MPI
安装目录。

## 10. 新增服务器或更换主服务器

### 10.1 新增计算节点

1. 在主服务器的 hostfile 中按期望 rank 顺序增加 `IP:slots`。
2. 配置主服务器到新节点的免密 SSH。
3. 确认新节点的 CANN、驱动和工具路径满足第 4 节要求。
4. 重新运行平铺同步或服务器端完整入口。
5. launcher 会自动把新节点加入 NPU 预检和 rank 映射。

只增加节点且代码不变时不要求重新编译，但必须把当前 `install/` 同步到新节点。

### 10.2 更换主服务器

1. 创建指向新主服务器 `<RemoteRoot>/source` 的新 Mutagen 会话。
2. 将 `-PrimaryHost` 改为 hostfile 第一项。
3. 确认新主服务器存在 hostfile，并能免密 SSH 到全部节点。
4. flush 新的 Mutagen 会话，并通过 `scp` 上传入口后用 `ssh` 运行服务器端完整入口。

## 11. NPU 占用规则

正式启动前，每轮都会对 hostfile 中所有服务器执行 `npu-smi info`，原始输出
保存到：

```text
<log-file>.npu_preflight/attempt_NN/<host>.log
```

判定规则：

- 没有 NPU 进程：允许启动。
- 只有进程名以 `tilexr_` 开头：允许多任务共用 NPU。
- Python 或其他通信测试进程：阻塞启动。
- 阻塞后默认每 15 秒重新检查。
- 120 秒仍未空闲：退出码为 `75`，不启动任何 rank。
- SSH、`npu-smi` 或进程表解析失败：立即停止。
- 不终止任何无关任务。

每个 rank 使用本任务专属 PID 文件和 `job_id`。中断、失败或超时时只清理本次
任务，不使用全局 `pkill`。

## 12. 通信端口

默认 bootstrap 地址为：

```text
TILEXR_COMM_ID=<hostfile 第一台服务器>:10067
```

TCP barrier 默认使用 bootstrap 端口加 97，即默认 `10164`。启动前会检查两个
端口是否已监听；任一端口被占用都会停止任务。并发运行多个 TileXR 测试时，
必须为每个任务设置不同的 `-CommPort` 或 `--comm-id`。

## 13. 日志与结果口径

默认日志位于主服务器 `<RemoteRoot>/logs/`。每次测试包含：

```text
<log-file>                         # 控制日志和全部 rank 输出
<log-file>.ranks/rank_0000.log     # rank 0 原始日志
<log-file>.ranks/rank_0001.log     # rank 1 原始日志
<log-file>.ranks/rank_averages.tsv # 每个 rank 的计时样本平均值
<log-file>.npu_preflight/          # 每轮 NPU 原始快照
```

rank 映射示例：

```text
RANK_MAP rank=8 host=141.61.49.223 local_rank=0 device=0
```

单 rank 样本：

```text
COMBINE_V2_SAMPLE bs=128 iteration=0 rank=8 elapsed_ms=0.412345
```

脚本最终聚合结果只包含正式指标：

```text
COMBINE_V2_PERF bs=128 k=16 h=3584 experts=64 dtype=bf16 ranks=16 iterations=80 avg_ms=0.410000 avg_alg_bw_GBps=35.805034 max_ms=0.450000 max_alg_bw_GBps=32.622364 correctness=passed
```

聚合方法：

1. 每个 rank 使用 ACL Event 测量单次算子时间，warmup 不进入样本。
2. 每个 rank 对自己的全部计时 iteration 求平均，得到本卡耗时。
3. 对全部卡的耗时求算术平均得到 `avg_ms`，取最大值得到 `max_ms`。
4. 用 `BS * K * H * dtype_bytes` 分别除以 `avg_ms` 和 `max_ms` 计算两项等效
   算法带宽，所有路由数据统一计为通信数据量。
5. 每个 BS 必须具备 `world size` 条 `COMBINE_V2_RANK_PERF` 通过记录，并且
   每个 rank 必须具备完整的计时样本，否则脚本返回失败。

正确性检查读取 `activeOutputOffset` 指向的当前 scratch epoch，验证 BF16 路由行
来自预期 source rank。该地址表示 Combine V2 原始通信输出，不是 TopK reduce
后的最终输出张量。

## 14. 已验证结果

### 14.1 512P 环境单机 8P

2026-08-11 在 512P 环境 2 号柜第一台服务器完成单机 8P 验证。节点按
`D:\3_codex\512P环境信息.txt` 解析为：

```text
CPU1 141.61.52.35
```

测试参数为 `BS=128, K=16, H=3584, experts=64, BF16, warmup=20,
iterations=80`。结果按“每卡 80 轮平均，再对所有卡取平均值和最大值”的正式
口径统计：

| BS | `avg_ms` | `avg_alg_bw_GBps` | `max_ms` | `max_alg_bw_GBps` |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.187879 | 78.135892 | 0.195413 | 75.123272 |

8 个 rank 正确性全部通过，共包含 640 个计时样本。正式日志：

```text
/home/h00580772/tilexr_combine_v2_8p_cab2_cpu1_logs/combine_v2_8p_bs128_w20_i80_20260811.log
```

### 14.2 A10-64P 环境历史 16P 结果

2026-08-11 在以下两台服务器上完成 16P、多 BS 验证：

```text
141.61.49.226:8
141.61.49.223:8
```

该记录生成于默认次数调整之前，测试参数为
`K=16, H=3584, experts=64, BF16, warmup=5, iterations=20`。当前正式测试
默认值已经调整为 `warmup=20, iterations(loop)=80`。下表采用旧的“逐轮最慢
rank 再求平均”口径，仅保留为历史记录，不能与 14.1 的当前正式口径直接比较。

| BS | `avg_ms` | `alg_bw_GBps` |
| ---: | ---: | ---: |
| 128 | 0.532375 | 27.574668 |
| 256 | 0.723498 | 40.580800 |
| 512 | 0.882888 | 66.509292 |
| 1024 | 2.202714 | 53.316278 |
| 2048 | 3.892452 | 60.342690 |
| 4096 | 7.777167 | 60.402721 |
| 8192 | 15.758240 | 59.621131 |

正式日志：

```text
/home/h00580772/tilexr_combine_v2_16p_cdf2b01_20260811/logs/combine_v2_16p_bs128_8192_batch_20260811.log
```

该结果包含 112 条 rank 正确性通过记录和 2240 条计时样本。测试基于当前工作树，
其中 Combine V2 kernel 的 `kEnableSafetyChecks=false`。

### 14.3 512P 环境 0/2 号柜 128P 结果

2026-08-11 在 512P 环境 0 号柜和 2 号柜共 16 台服务器上完成 128P 单点测试，
每台服务器使用 8 张卡。测试参数为
`BS=8192, K=16, H=3584, experts=256, BF16, warmup=20, iterations=80`。

单 rank 的完整逻辑通信数据量为：

```text
8192 * 16 * 3584 * 2 = 939,524,096 bytes
```

结果按“每卡 80 轮平均，再对 128 张卡取平均值和最大值”的正式口径统计：

| BS | `avg_ms` | `avg_alg_bw_GBps` | `max_ms` | `max_alg_bw_GBps` |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 20.588683 | 45.633035 | 20.930525 | 44.887747 |

128 个 rank 的正确性全部通过，共包含 10240 个计时样本；每个 rank 均具备完整的
80 个计时样本。最慢卡为 rank 1，其 80 轮平均耗时为 `20.930525 ms`。

全部性能测量和正确性检查完成后，最终 TCP barrier 有 36 个 rank 失败，因此
launcher 返回码为 1，未输出正常的 `COMBINE_V2_PERF` 聚合行。上表由原始
`COMBINE_V2_SAMPLE` 日志按第 13 节口径重新聚合，可用于本次性能结果；该异常仅
表示测试结束阶段的 barrier/清理未正常完成，不表示计时样本或正确性检查失败。

正式日志：

```text
/home/h00580772/tilexr_combine_v2_128p_cab0_2_logs/combine_v2_128p_bs8192_exp256_w20_i80_20260811.log
```

## 15. 常见失败

| 现象 | 检查项 |
| --- | --- |
| `remote runtime validation failed` | 新节点的 install、CANN、`timeout` 或 `ss` 是否存在 |
| `unsupported Combine V2 world size` | 调整 hostfile，使 rank 总数为 `2-8/16/32/64/128` |
| `expert count ... must be divisible by world size` | 调整 `--experts` 或 hostfile rank 总数 |
| `batch size ... must be divisible by world size` | 调整 BS 或 rank 总数 |
| `bootstrap or barrier port is already listening` | 更换 `-CommPort` |
| `NPU preflight timed out` | 查看 `.npu_preflight/`，等待非 TileXR 任务结束 |
| `a rank launcher failed` | 查看 `.ranks/rank_NNNN.log` |
| `rank logs do not contain...` | 检查缺失的 rank 正确性或 iteration 样本 |
| SHA-256 校验失败 | 重新从主服务器执行平铺同步，不要直接在 worker 修改 install |
