## Why

各脚本之间存在大量重复代码（权限修复循环、安装逻辑、运行逻辑），且同类操作（如 hcomm 的 build+install）被拆散到多个文件中各自为政，导致维护成本高、改动容易遗漏。重构统一管理，便于后续扩展和维护。

## What Changes

- **`common_util.sh`**：新增 `fix_permissions()` 函数，消除两个 cann 脚本中的重复权限修复代码
- **`cann_local_install.sh`**：成为纯原子安装脚本，调用 `fix_permissions()`
- **`cann_download_install.sh`**：下载完成后调用 `cann_local_install.sh`；curl 后台进程通过 PID 文件追踪，重跑时检查 `/proc/$pid/comm` 判断是否需要重新 fork
- **`hcomm_local_install.sh`**：成为纯原子安装脚本（只做 install）
- **`hcomm_build_install.sh`**：build（带 `--noclean`）后调用 `hcomm_local_install.sh`
- **`hcomm_clean_build_install.sh`**：build（不带 `--noclean`）后调用 `hcomm_local_install.sh`
- **`ops_only_run.sh`**：成为纯原子运行脚本
- **`ops_build_run.sh`**：build + install 后调用 `ops_only_run.sh`，消除重复的 run 逻辑
- **`common_env.sh`**：基本不动

## Capabilities

### New Capabilities

- `fix-permissions`: 目录权限修复函数，提取自 cann 脚本，集中到 common_util.sh
- `curl-pid-tracking`: curl 后台下载的 PID 文件追踪与进程验证机制，支持重跑时接管正在进行的下载
- `compositional-scripts`: 脚本组合调用模式——原子脚本独立可用，上层脚本通过 `bash` 调用组合

### Modified Capabilities

## Impact

- 涉及文件：`common_util.sh`、`cann_download_install.sh`、`cann_local_install.sh`、`hcomm_build_install.sh`、`hcomm_clean_build_install.sh`、`hcomm_local_install.sh`、`ops_build_run.sh`、`ops_only_run.sh`
- 不影响 `common_env.sh`、`opbase_build_install.sh`、`test_build.sh`、`test_allreduce.sh`、`run_ops_test.sh`、`plog_grep.sh`、`device_connect.sh`、`driver_fix.sh`、`watch.sh`
- 对外行为不变：每个脚本的入口参数和执行效果保持一致
