## Context

TileXR 项目有 17 个 shell 脚本，其中多个脚本之间存在重复逻辑：
- `cann_download_install.sh` 和 `cann_local_install.sh` 各自包含一份相同的目录权限修复循环
- `cann_download_install.sh` 的安装步骤与 `cann_local_install.sh` 几乎完全重复
- `hcomm_build_install.sh` 和 `hcomm_clean_build_install.sh` 仅 `--noclean` flag 不同，但安装行重复了两遍；`hcomm_local_install.sh` 又单独存一份安装逻辑
- `ops_build_run.sh` 的 run 段与 `ops_only_run.sh` 完全重复

重构目标：通过提取共用函数 + 组合式调用，消除重复，同时保持每个脚本独立可运行。

## Goals / Non-Goals

**Goals:**
- `fix_permissions()` 提取到 `common_util.sh`
- `cann_download_install.sh` 下载后调用 `bash cann_local_install.sh`（组合式）
- curl 后台进程的 PID 持久化到文件，重跑时通过 `/proc/$pid/comm` 验证进程身份决定是否重新 fork
- `hcomm_build_install.sh` 和 `hcomm_clean_build_install.sh` 共享 `_hcomm_build()` 函数（定义在 `common_util.sh` 中），安装步骤统一由 `hcomm_local_install.sh` 承担
- `ops_build_run.sh` build+install 后调用 `bash ops_only_run.sh`
- 每个脚本对外行为（入参、产出）不变

**Non-Goals:**
- 不修改 `common_env.sh`（环境变量部分已足够清晰）
- 不改动 `run_ops_test.sh` 中的硬编码配置（另议）
- 不合并 `hcomm_build_install.sh` 和 `hcomm_clean_build_install.sh` 为单一入口

## Decisions

### 决策 1：`fix_permissions()` 放在 `common_util.sh` 而非单独文件

所有脚本均已 source `common_env.sh`，而 `common_env.sh` 已 source `common_util.sh`。将 `fix_permissions()` 加入 `common_util.sh` 无需改变任何 source 链，调用方只需把循环替换为函数调用。

**替代方案**：提取到 `cann_common.sh` 专用文件。增加了文件数量，收益不大，否定。

### 决策 2：组合调用使用 `bash <script>`，而非 `source`

- `bash` 调用：每个脚本在独立子 shell 中运行，互不污染；每个脚本可单独执行，也可被上层调用
- `source` 调用：共享当前 shell 环境，但 `exit` 会退出调用方；且被调用脚本中的变量会泄漏到调用方

选择 `bash`。由于每个脚本都会自行 `source common_env.sh`，环境变量重复加载无副作用。

### 决策 3：curl PID 追踪机制

目标：脚本重跑时，若上次的 curl 下载仍在进行，直接接管等待，不重复 fork。

```
PID 文件路径：${TILEXR_TEMP_HOME}/cann_toolkit.pid
              ${TILEXR_TEMP_HOME}/cann_ops.pid

重跑逻辑：
  if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file")
      if kill -0 "$pid" 2>/dev/null; then
          comm=$(cat /proc/$pid/comm 2>/dev/null)
          if [ "$comm" = "curl" ]; then
              # 接管：直接进入等待循环
          else
              # PID 被复用，清除文件，重新 fork
          fi
      else
          # 进程已退出，清除文件，重新 fork
      fi
  else
      # 首次运行，直接 fork
  fi
```

curl 本身已有 `-C -`（断点续传），重新 fork 也不会从头下载。

### 决策 4：`_hcomm_build()` 定义位置

`_hcomm_build()` 接受一个可选参数（`--noclean` 或空），包含 build 命令逻辑。定义在 `common_util.sh` 中，`hcomm_build_install.sh` 和 `hcomm_clean_build_install.sh` source 进来后直接调用，安装步骤则用 `bash hcomm_local_install.sh` 完成。

## Risks / Trade-offs

- **`/proc/$pid/comm` 可移植性** → 在 Linux 上可靠；macOS 不支持（但项目明确 Ubuntu 20.04，无风险）
- **bash 子 shell 调用开销** → 每次调用多一个 bash 进程，可忽略不计
- **`hcomm_local_install.sh` 被直接调用时的幂等性** → 该脚本直接运行已有 .run 包，无副作用，重复执行安全
- **PID 文件残留** → 若 curl 被 kill 但 PID 文件未清除，下次运行会因 `kill -0` 失败自动清除并重新 fork，无需手动干预
