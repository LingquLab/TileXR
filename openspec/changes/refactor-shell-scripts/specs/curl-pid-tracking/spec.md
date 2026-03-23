## ADDED Requirements

### Requirement: curl PID 持久化到文件
`cann_download_install.sh` 启动后台 curl 进程后，SHALL 将其 PID 写入 `${TILEXR_TEMP_HOME}/cann_toolkit.pid`（toolkit）和 `${TILEXR_TEMP_HOME}/cann_ops.pid`（ops）。

#### Scenario: 首次运行写入 PID 文件
- **WHEN** 脚本首次运行且 PID 文件不存在
- **THEN** 启动 `curl ... &`，将 `$!` 写入对应 PID 文件

### Requirement: 重跑时检查 PID 文件并决定是否重新 fork
`cann_download_install.sh` 重跑时 SHALL 读取 PID 文件，按以下逻辑决定处理方式：
1. PID 文件不存在 → 正常 fork 新 curl 进程
2. PID 文件存在但进程已退出（`kill -0` 失败）→ 删除 PID 文件，重新 fork
3. PID 文件存在且进程存活，但 `/proc/$pid/comm` 不为 `curl` → PID 被复用，删除 PID 文件，重新 fork
4. PID 文件存在且进程存活且为 curl → 接管该 PID，直接进入等待循环，不重新 fork

#### Scenario: curl 仍在运行时重跑脚本
- **WHEN** 脚本重跑且 PID 文件中的进程存活且 comm 为 `curl`
- **THEN** 不启动新 curl，直接等待该进程完成

#### Scenario: PID 被其他进程复用
- **WHEN** 脚本重跑且 PID 文件中的 PID 对应进程 comm 不为 `curl`
- **THEN** 删除旧 PID 文件，重新 fork curl（curl `-C -` 自动断点续传）

#### Scenario: curl 已完成后重跑
- **WHEN** 脚本重跑且 PID 文件中的进程已退出
- **THEN** 删除旧 PID 文件，重新 fork curl

### Requirement: 下载完成后清理 PID 文件
curl 进程正常退出后，SHALL 删除对应的 PID 文件。

#### Scenario: 下载成功后清理
- **WHEN** curl 进程正常退出（exit code 0）
- **THEN** 对应 PID 文件被删除
