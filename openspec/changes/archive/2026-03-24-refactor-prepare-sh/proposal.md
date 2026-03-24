## Why

`prepare.sh` 中 8 个工具包的安装逻辑大量重复（autoconf 模式出现 4 次），且存在 `cd $PWD` bug 和函数包装冗余，与昨天确立的"提取公共函数、消除重复"重构原则不一致。

## What Changes

- **`common_env.sh`**：新增 `_install_autoconf_pkg <pkg_name> <tarball> [configure_args...]` 函数，封装 tar 解压 + configure + make + make install 的标准流程，安装到 `${TILEXR_UTIL_HOME}/<pkg_name>/`
- **`prepare.sh`**：
  - 去掉 `prepare_3rd()` 函数包装，改为顶层直接执行（与其他脚本风格统一）
  - `time`、`patch`、`sshpass`、`mpich` 四个包的安装块替换为 `_install_autoconf_pkg` 调用
  - 修复 `cd $PWD` → `cd ${TILEXR_HOME}`
  - 补齐遗漏的 `colorful_time`（`sshpass` tar、`mpich` configure）

## Capabilities

### New Capabilities

- `autoconf-pkg-install`: 标准 autoconf 工具包安装函数，集中到 `common_env.sh`

### Modified Capabilities

## Impact

- 涉及文件：`common_env.sh`、`prepare.sh`
- 不影响安装结果和安装路径，对外行为不变
- `_install_autoconf_pkg` 函数定义在 `common_env.sh`（环境层），与 `_hcomm_build` 同层，保持依赖方向正确
