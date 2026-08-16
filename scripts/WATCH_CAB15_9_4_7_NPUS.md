# 15/9/4/7 柜 NPU 占用监控使用说明

## 文件位置

本地 PC 脚本：

```text
D:\3_codex\tileXR\scripts\watch_cab15_9_4_7_npus.sh
```

8 柜 CPU1 上的可执行副本：

```text
/home/h00580772/tilexr_npu_watch_15_9_4_7/scripts/watch_cab15_9_4_7_npus.sh
```

## 监控范围

| 机框编号 | 柜号 | NPU 数量 |
| --- | ---: | ---: |
| C04-B04 | 15 | 64 |
| C02-B10 | 9 | 64 |
| C02-A09 | 4 | 64 |
| C02-B08 | 7 | 64 |

脚本共检测 32 台服务器、256 张 NPU。每个柜按 CPU1 到 CPU8 排列，每台
服务器的 8 位状态按 NPU0 到 NPU7 排列。

## 持续监控

在本地 PC 打开 Git Bash，然后执行：

```bash
ssh root@141.61.55.42
cd /home/h00580772/tilexr_npu_watch_15_9_4_7
bash scripts/watch_cab15_9_4_7_npus.sh
```

脚本每 2 秒刷新一次。按 `Ctrl+C` 停止。

也可以在本地 Git Bash 中用一条命令启动：

```bash
ssh -t root@141.61.55.42 \
  'cd /home/h00580772/tilexr_npu_watch_15_9_4_7 && bash scripts/watch_cab15_9_4_7_npus.sh'
```

## 单次检查

登录 8 柜 CPU1 后执行：

```bash
cd /home/h00580772/tilexr_npu_watch_15_9_4_7
bash scripts/watch_cab15_9_4_7_npus.sh --once
```

## 输出含义

```text
Cabinet 15: 00000000 ... freeNum=64/64
Cabinet 9 : 00000000 ... freeNum=64/64
Cabinet 4 : 00000000 ... freeNum=64/64
Cabinet 7 : 00000000 ... freeNum=64/64
```

- `0`：NPU 空闲。
- `1`：NPU 有运行进程，判定为占用。
- `?`：SSH、`npu-smi` 或采样超时，不能判定为空闲。
- `freeNum`：当前柜中状态为 `0` 的 NPU 数量，探测失败的 NPU 不计入空闲数。

脚本只检查进程占用，不修改、停止或清理任何 NPU 任务，也不代表完整的设备健康检查。
