## ADDED Requirements

### Requirement: cann_download_install.sh 调用 cann_local_install.sh
`cann_download_install.sh` 的安装阶段 SHALL 通过 `bash ${script_path}/cann_local_install.sh` 完成，不再内联重复安装逻辑。

#### Scenario: 下载完成后调用 local install
- **WHEN** toolkit 和 ops 的 curl 下载均完成
- **THEN** 执行 `bash cann_local_install.sh`，由其完成 fix_permissions 和两个 .run 包的安装

### Requirement: hcomm 脚本共享 _hcomm_build 函数
`common_util.sh` 中 SHALL 定义 `_hcomm_build [--noclean]` 函数，包含 hcomm build 命令逻辑。

#### Scenario: hcomm_build_install.sh 使用 --noclean
- **WHEN** 执行 `hcomm_build_install.sh`
- **THEN** 调用 `_hcomm_build --noclean`，完成后执行 `bash hcomm_local_install.sh`

#### Scenario: hcomm_clean_build_install.sh 使用全量构建
- **WHEN** 执行 `hcomm_clean_build_install.sh`
- **THEN** 调用 `_hcomm_build`（不带 --noclean），完成后执行 `bash hcomm_local_install.sh`

### Requirement: ops_build_run.sh 调用 ops_only_run.sh
`ops_build_run.sh` 的运行阶段 SHALL 通过 `bash ${script_path}/ops_only_run.sh` 完成，不再内联重复 run 逻辑。

#### Scenario: build 完成后调用 ops_only_run.sh
- **WHEN** ops-transformer build 和 install 均成功
- **THEN** 执行 `bash ops_only_run.sh [ops]`，将 ops 参数透传

### Requirement: 原子脚本独立可运行
所有原子脚本（`cann_local_install.sh`、`hcomm_local_install.sh`、`ops_only_run.sh`）SHALL 在不依赖调用方的情况下可直接执行，自行 source `common_env.sh`。

#### Scenario: 单独执行原子脚本
- **WHEN** 直接运行 `bash cann_local_install.sh`（或其他原子脚本）
- **THEN** 脚本正常执行，不因缺少调用方环境变量而失败
