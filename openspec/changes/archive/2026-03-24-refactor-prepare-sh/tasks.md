## 1. common_env.sh — 添加 _install_autoconf_pkg 函数

- [x] 1.1 在 `common_env.sh` 末尾（`_hcomm_build` 之后、`if [[ "${BASH_SOURCE[0]}" == "${0}" ]]` 之前）添加 `_install_autoconf_pkg <pkg_name> <tarball> [configure_args...]` 函数

## 2. prepare.sh — 重构主体

- [x] 2.1 去掉 `prepare_3rd()` 函数包装，将函数体内容提升为顶层代码
- [x] 2.2 将 `time` 安装块替换为 `_install_autoconf_pkg time time-1.9.tar.gz`
- [x] 2.3 将 `patch` 安装块替换为 `_install_autoconf_pkg patch patch-2.8.tar.gz`
- [x] 2.4 将 `sshpass` 安装块替换为 `_install_autoconf_pkg sshpass sshpass-1.06.tar.gz`
- [x] 2.5 将 `mpich` 安装块替换为 `_install_autoconf_pkg mpich mpich-4.3.1.tar.gz --disable-fortran`
- [x] 2.6 将末尾 `cd $PWD` 修复为 `cd ${TILEXR_HOME}`

## 3. 验证

- [x] 3.1 `bash -n common_env.sh && bash -n prepare.sh` 语法检查通过
- [x] 3.2 确认 `common_util.sh` 中无 `TILEXR_` 变量引用（依赖方向保持正确）
