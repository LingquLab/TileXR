## ADDED Requirements

### Requirement: fix_permissions 函数集中定义
`common_util.sh` 中 SHALL 定义 `fix_permissions <path>` 函数，将指定路径及其所有祖先目录（直至 `/` 或 `/home`）的权限修复为 755。

#### Scenario: 路径权限不足时自动修复
- **WHEN** 调用 `fix_permissions /some/deep/path`
- **THEN** 从该路径向上遍历，对每个权限不为 755 的目录执行 `chmod 755` 并打印 warn 日志，直到到达 `/` 或 `/home` 为止

#### Scenario: 路径权限已正确时跳过
- **WHEN** 调用 `fix_permissions` 且所有祖先目录权限已为 755
- **THEN** 不执行任何 chmod，不打印 warn 日志

### Requirement: cann 脚本使用 fix_permissions 函数
`cann_local_install.sh` 和 `cann_download_install.sh` 中 SHALL 删除内联的权限修复循环，改为调用 `fix_permissions ${TILEXR_CANN_HOME}`。

#### Scenario: cann 安装前权限修复
- **WHEN** 执行任意 cann 安装脚本
- **THEN** 通过 `fix_permissions` 函数完成权限修复，行为与原内联循环完全一致
