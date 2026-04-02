# 模块：environment（环境与构建脚本）

## 1. 模块作用

管理 TileXR 的开发环境配置、依赖安装、构建流程和测试执行。所有脚本均依赖 `common_env.sh` 提供的环境变量基础设施。

---

## 2. 目录结构

```
/root/code/TileXR/（根目录脚本）
├── common_env.sh              # 环境变量基础设施（必须首先 source）
├── common_util.sh             # 通用工具函数
├── cann_download_install.sh   # CANN 工具链安装
├── cann_local_install.sh      # CANN 本地安装（离线）
├── hcomm_build_install.sh     # hcomm 子模块编译安装
├── hcomm_clean_build_install.sh # 清理重建 hcomm
├── hcomm_local_install.sh     # hcomm 本地安装
├── opbase_build_install.sh    # OpBase 子模块安装
├── ops_build_run.sh           # ops-transformer 编译 + 运行算子
├── ops_only_run.sh            # 仅运行算子（不重新编译）
├── test_build.sh              # 构建 HCCL 测试套件
├── test_allreduce.sh          # 运行 AllReduce 多卡测试（mpirun）
├── plog_grep.sh               # 过滤设备日志（如 plog_grep.sh ERROR）
├── prepare.sh                 # 首次环境准备
└── device_connect.sh          # 设备连接脚本
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `common_env.sh` | 所有脚本的基础，设置 `TILEXR_HOME`、`TILEXR_CANN_HOME`、`TILEXR_ASCEND_DEV_NUM`、`TILEXR_MAX_NPROC`、`TILEXR_HALF_NPROC`；检测 SOC 类型；source CANN 环境 |
| `ops_build_run.sh` | 编译 ops-transformer 并运行算子，是 mc2 算子的主要构建入口 |
| `test_allreduce.sh` | 通过 `mpirun` 启动多 rank 进程运行 AllReduce 测试 |
| `plog_grep.sh` | 过滤 `/var/log/npu/slog/` 下的设备日志，支持按级别（ERROR/WARNING/INFO）过滤 |

---

## 4. 关键环境变量（由 `common_env.sh` 设置）

```bash
TILEXR_HOME=/root/code/TileXR         # 项目根目录
TILEXR_CANN_HOME=<cann 安装目录>       # CANN 工具链路径
TILEXR_CANN_VER=9.0.0-beta.1          # CANN 版本
TILEXR_ASCEND_DEV_NUM=<设备数>         # NPU 设备数量（npu-smi 探测）
TILEXR_MAX_NPROC=<CPU核数>             # 最大编译并发数
TILEXR_HALF_NPROC=<CPU核数/2>         # 半数并发（保守编译）
SOC_VERSION=<ascend910b|310p3|...>    # 当前 SOC 类型
```

---

## 5. 标准构建流程

```bash
# 首次安装（按顺序）
bash cann_download_install.sh       # 1. 安装 CANN 工具链
source common_env.sh                # 2. 设置环境
bash hcomm_build_install.sh         # 3. 编译安装 hcomm
bash opbase_build_install.sh        # 4. 安装 opbase
bash ops_build_run.sh               # 5. 编译运行 ops-transformer

# 核心库构建
source common_env.sh
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j${TILEXR_MAX_NPROC} && make install

# 测试运行
bash test_build.sh
bash test_allreduce.sh
```

---

## 6. 关键业务逻辑

### SOC 类型自动检测
`common_env.sh` 调用 `npu-smi info` 解析当前设备的芯片型号，自动设置 `SOC_VERSION`，供 CMake 和 ops 构建脚本使用。

### 并发数控制
`TILEXR_MAX_NPROC` 和 `TILEXR_HALF_NPROC` 通过 `nproc` 探测 CPU 核数，避免编译时过度占用资源。已知这两个变量是由用户修改 `common_env.sh` 后的设计选择，而非自动限速。

### CANN 多版本支持
`common_env.sh` 内部通过版本匹配（8.5.0 / 9.0.0-beta.1）选择不同的 OBS 下载目录和安装路径，切换版本时只需修改 `TILEXR_CANN_VER`。

---

## 7. 开发注意事项

- 每个新 shell 会话都必须执行 `source common_env.sh`，不能用 `bash common_env.sh`（后者不影响当前 shell 的环境变量）。
- `plog_grep.sh` 需要 root 权限读取 `/var/log/npu/slog/`。
- `test_allreduce.sh` 依赖 `mpirun`（OpenMPI），需提前安装并配置 hostfile。
- 修改 `common_env.sh` 中的路径变量后，需重新 `source` 才生效。

---

## 8. 未来可扩展点

- **Docker 化**：将 `common_env.sh` 中的依赖封装为 Dockerfile，提供可复现的开发环境。
- **CI 集成**：`test_allreduce.sh` 可参数化 rank 数和测试类型，接入自动化测试流水线。
- **版本管理**：增加 `common_env.sh` 的版本检查，若 CANN 版本不匹配则给出明确错误提示。
