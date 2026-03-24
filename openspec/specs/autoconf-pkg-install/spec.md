## ADDED Requirements

### Requirement: _install_autoconf_pkg 函数定义
`common_env.sh` 中 SHALL 定义 `_install_autoconf_pkg <pkg_name> <tarball> [configure_args...]` 函数，封装标准 autoconf 安装流程：创建目录、解压 tarball、configure、make、make install，并报告成功或失败。安装前缀固定为 `${TILEXR_UTIL_HOME}/<pkg_name>/`，日志追加到 `${TILEXR_TEMP_HOME}/3rd.log`。

#### Scenario: 正常安装
- **WHEN** 调用 `_install_autoconf_pkg sshpass sshpass-1.06.tar.gz`
- **THEN** 解压到 `${TILEXR_TEMP_HOME}/sshpass/`，configure 安装到 `${TILEXR_UTIL_HOME}/sshpass/`，make install 成功后打印 success 日志

#### Scenario: 携带额外 configure 参数
- **WHEN** 调用 `_install_autoconf_pkg mpich mpich-4.3.1.tar.gz --disable-fortran`
- **THEN** configure 命令追加 `--disable-fortran` 参数，其余流程不变

#### Scenario: make install 失败
- **WHEN** make install 返回非零退出码
- **THEN** 打印 error 日志，函数返回非零退出码

### Requirement: prepare.sh 使用 _install_autoconf_pkg
`prepare.sh` 中 `time`、`patch`、`sshpass`、`mpich` 的安装块 SHALL 替换为 `_install_autoconf_pkg` 调用，消除重复的内联安装代码。

#### Scenario: time 安装
- **WHEN** `${TILEXR_UTIL_HOME}/time/bin/time` 不存在
- **THEN** 调用 `_install_autoconf_pkg time time-1.9.tar.gz`，行为与原内联代码一致

#### Scenario: mpich 安装（带额外参数）
- **WHEN** `${MPI_HOME}/bin/mpirun` 不存在
- **THEN** 调用 `_install_autoconf_pkg mpich mpich-4.3.1.tar.gz --disable-fortran`

### Requirement: prepare.sh 顶层执行，无函数包装
`prepare.sh` SHALL 以顶层脚本形式直接执行，不将主体包裹在 `prepare_3rd()` 函数中。

#### Scenario: 直接执行
- **WHEN** 运行 `bash prepare.sh`
- **THEN** 脚本顶层逐步执行各安装步骤，无需调用任何包装函数

### Requirement: prepare.sh 以 TILEXR_HOME 结尾
`prepare.sh` 末尾 SHALL 执行 `cd ${TILEXR_HOME}` 回到项目根目录，替换原有的 `cd $PWD`。

#### Scenario: 执行完成后目录正确
- **WHEN** prepare.sh 执行完毕
- **THEN** 当前目录为 `${TILEXR_HOME}`，而非某个临时构建目录
