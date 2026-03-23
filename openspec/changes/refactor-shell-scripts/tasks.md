## 1. common_util.sh — 提取公共函数

- [x] 1.1 在 `common_util.sh` 中添加 `fix_permissions <path>` 函数（将原两个 cann 脚本中的权限修复循环提取为函数）
- [x] 1.2 在 `common_util.sh` 中添加 `_hcomm_build [--noclean]` 函数（封装 hcomm build 命令，接受可选 `--noclean` 参数）

## 2. cann_local_install.sh — 原子安装脚本

- [x] 2.1 删除内联的权限修复循环，改为调用 `fix_permissions ${TILEXR_CANN_HOME}`
- [x] 2.2 验证脚本可独立执行：`bash cann_local_install.sh`（需 TEMP_HOME 中有对应 .run 文件）

## 3. cann_download_install.sh — 下载 + 组合调用

- [x] 3.1 删除内联的权限修复循环（由 `cann_local_install.sh` 负责）
- [x] 3.2 为两个 curl 进程实现 PID 文件追踪（写入 `${TILEXR_TEMP_HOME}/cann_toolkit.pid` 和 `cann_ops.pid`）
- [x] 3.3 在每个 curl 启动前实现重跑检查逻辑：读取 PID 文件 → `kill -0` 检查存活 → `/proc/$pid/comm` 验证为 `curl` → 决定接管或重新 fork
- [x] 3.4 curl 进程正常退出后删除对应 PID 文件
- [x] 3.5 删除内联的安装逻辑，在两个 curl 均完成后改为调用 `bash ${script_path}/cann_local_install.sh`

## 4. hcomm_build_install.sh 和 hcomm_clean_build_install.sh — 共享 build 函数

- [x] 4.1 修改 `hcomm_build_install.sh`：将 build 命令替换为 `_hcomm_build --noclean`，安装步骤替换为 `bash ${script_path}/hcomm_local_install.sh`
- [x] 4.2 修改 `hcomm_clean_build_install.sh`：将 build 命令替换为 `_hcomm_build`（无参数），安装步骤替换为 `bash ${script_path}/hcomm_local_install.sh`
- [x] 4.3 验证 `hcomm_local_install.sh` 可独立执行（自行 source common_env.sh，无多余空行）

## 5. ops_build_run.sh — 调用 ops_only_run.sh

- [x] 5.1 修改 `ops_build_run.sh`：删除末尾内联的 run 逻辑，改为 `bash ${script_path}/ops_only_run.sh ${ops}`
- [x] 5.2 确认 `ops_only_run.sh` 接受 `$1` 参数作为 ops 名称（与 `ops_build_run.sh` 的 `${ops}` 变量对应）

## 6. 验证

- [x] 6.1 逐个确认修改后的脚本 `source common_env.sh` 后无报错（shellcheck 或手动检查）
- [x] 6.2 确认 `hcomm_local_install.sh` 中无多余空行（清理历史遗留格式）
- [x] 6.3 确认 `ops_only_run.sh` 中无多余空行（清理历史遗留格式）
