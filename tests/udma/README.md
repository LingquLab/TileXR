# TileXR UDMA 测试指南

本文档提供 TileXR UDMA 功能的完整测试指南，包括单元测试和集成测试。

---

## 📋 目录

1. [测试概述](#测试概述)
2. [环境准备](#环境准备)
3. [构建测试](#构建测试)
4. [运行测试](#运行测试)
5. [测试说明](#测试说明)
6. [故障排查](#故障排查)
7. [预期结果](#预期结果)

---

## 测试概述

### 测试目标

验证 TileXR 的 UDMA 集成功能，包括：
- shmem API (`aclshmemx_get_udma_info`) 的正确性
- TileXR 初始化流程中的 UDMA 集成
- 多进程/多卡环境下的 UDMA 功能
- 内存管理和资源清理

### 测试层次

```
┌─────────────────────────────────────┐
│  集成测试 (test_tilexr_udma)       │
│  - TileXR 完整初始化流程            │
│  - 多 rank 协调                     │
│  - 共享内存缓冲区                   │
│  - 压力测试                         │
└─────────────────────────────────────┘
              ↓ 依赖
┌─────────────────────────────────────┐
│  单元测试 (test_shmem_api)         │
│  - API 参数验证                     │
│  - 初始化状态检查                   │
│  - UDMA 信息获取                    │
│  - 多次调用一致性                   │
└─────────────────────────────────────┘
              ↓ 依赖
┌─────────────────────────────────────┐
│  shmem 库 (libshmem.so)            │
│  - aclshmemx_get_udma_info() API   │
└─────────────────────────────────────┘
```

### 测试文件

```
tests/udma/
├── CMakeLists.txt                    # 构建配置
├── build.sh                          # 构建脚本
├── run_tests.sh                      # 测试运行脚本
├── README.md                         # 本文档
├── unit/
│   └── test_shmem_api.cpp           # shmem API 单元测试
└── integration/
    └── test_tilexr_udma.cpp         # TileXR 集成测试
```

---

## 环境准备

### 硬件要求

- **最低配置**: 1 张昇腾 NPU (910B/910A5/310P3)
- **推荐配置**: 2 张或更多昇腾 NPU（用于多卡测试）

### 软件要求

- **操作系统**: Ubuntu 20.04 LTS
- **CANN**: 9.1.0 或更高版本
- **驱动**: NPU driver ≥ 25.5.0
- **编译器**: GCC 13.3.0 或兼容版本
- **CMake**: 3.16 或更高版本
- **MPI**: MPICH 或 OpenMPI（可选，用于多进程测试）

### 检查环境

```bash
# 检查 NPU 驱动
npu-smi info

# 检查 NPU 数量
lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | wc -l

# 检查 CANN 版本
cat ${ASCEND_HOME_PATH}/version.info

# 检查 MPI（可选）
which mpirun
mpirun --version
```

### 构建依赖

确保以下库已构建：

1. **shmem 库** (tilexr-udma-integration 分支)
   ```bash
   cd /home/TileXR/3rdparty/shmem
   git checkout tilexr-udma-integration
   bash scripts/build.sh -soc_type Ascend950
   # 输出: install/shmem/lib/libshmem.so
   ```

2. **TileXR 库** (feature/udma-integration 分支)
   ```bash
   cd /home/TileXR
   git checkout feature/udma-integration
   source common_env.sh
   mkdir -p build && cd build
   cmake -DCMAKE_INSTALL_PREFIX=../install ..
   make -j$(nproc) && make install
   # 输出: install/lib/libtile-comm.so
   ```

---

## 构建测试

### 快速构建

```bash
cd /home/TileXR/tests/udma
bash build.sh
```

### 手动构建

```bash
cd /home/TileXR/tests/udma

# 加载环境
source ../../common_env.sh

# 创建构建目录
mkdir -p build && cd build

# 配置
cmake -DCMAKE_INSTALL_PREFIX=../install ..

# 构建
make -j$(nproc)

# 安装
make install
```

### 验证构建

```bash
ls -lh install/bin/
# 应该看到:
# test_shmem_api
# test_tilexr_udma

# 检查依赖
ldd install/bin/test_shmem_api
ldd install/bin/test_tilexr_udma
```

---

## 运行测试

### 快速运行（推荐）

```bash
cd /home/TileXR/tests/udma
bash run_tests.sh
```

这个脚本会自动运行所有测试并输出结果。

### 手动运行

#### 1. 设置环境变量

```bash
cd /home/TileXR/tests/udma
source ../../common_env.sh

# 添加库路径
export LD_LIBRARY_PATH=/home/TileXR/3rdparty/shmem/install/shmem/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/TileXR/install/lib:${LD_LIBRARY_PATH}
```

#### 2. 运行单元测试

```bash
# shmem API 单元测试（单进程）
./install/bin/test_shmem_api
```

**预期输出**:
```
========================================
  shmem UDMA API Unit Tests
========================================

=== Test Case: API Parameter Validation ===
[PASS] Should return INVALID_PARAM when udma_info_ptr is NULL
[PASS] Should return INVALID_PARAM when udma_info_size is NULL
[PASS] Should return INVALID_PARAM when both parameters are NULL

=== Test Case: Uninitialized State ===
[PASS] Should return INNER_ERROR when shmem not initialized

=== Test Case: Full Initialization Flow ===
[PASS] aclrtSetDevice should succeed
[PASS] aclshmemx_get_uniqueid should succeed
[PASS] aclshmemx_set_attr_uniqueid_args should succeed
[PASS] aclshmemx_init_attr should succeed
[PASS] aclshmemx_get_udma_info should succeed
[PASS] UDMA info pointer should not be NULL
[PASS] UDMA info size should be greater than 0
UDMA Info Pointer: 0x7f8a40000000
UDMA Info Size: 12345 bytes
[PASS] UDMA info should be in device memory
Memory Type: 1 (0=HOST, 1=DEVICE)
[PASS] aclshmem_finalize should succeed

=== Test Case: Multiple Calls Consistency ===
[PASS] All calls should succeed
[PASS] All calls should return the same pointer
[PASS] All calls should return the same size
Pointer consistency: 0x7f8a40000000 == 0x7f8a40000000 == 0x7f8a40000000
Size consistency: 12345 == 12345 == 12345

========================================
  Test Summary
========================================
Total:  16
Passed: 16
Failed: 0
========================================
```

#### 3. 运行集成测试（单进程）

```bash
# TileXR 集成测试（单卡）
export RANK=0
export RANK_SIZE=1
./install/bin/test_tilexr_udma
```

**预期输出**:
```
========================================
  TileXR UDMA Integration Tests
========================================
Environment:
  RANK: 0
  RANK_SIZE: 1
  PID: 12345
Using device: 0

=== Test Case: TileXR Basic Initialization ===
Rank: 0/1
[PASS] TileXRInit should succeed
[PASS] TileXRSync should succeed

=== Test Case: UDMA Initialization ===
[PASS] TileXRInit should succeed
[PASS] CommArgs pointer should not be NULL
CommArgs pointer: 0x7f8a50000000

=== Test Case: Multi-Rank Initialization ===
[SKIP] This test requires at least 2 ranks

=== Test Case: Shared Memory Buffers ===
[PASS] TileXRInit should succeed
[PASS] Send buffer should not be NULL
Send buffer: 0x7f8a60000000
[PASS] Recv buffer should not be NULL
Recv buffer: 0x7f8a70000000
[PASS] Send and recv buffers should be different

=== Test Case: Stress Test - Multiple Init/Finalize ===
Iteration 1/5
Iteration 2/5
Iteration 3/5
Iteration 4/5
Iteration 5/5
[PASS] All iterations should succeed (5/5)

========================================
  Test Summary (Rank 0)
========================================
Total:  10
Passed: 10
Failed: 0
========================================
```

#### 4. 运行多进程测试（需要 MPI）

```bash
# 2 进程测试
mpirun -n 2 ./install/bin/test_tilexr_udma

# 4 进程测试（如果有 4 张卡）
mpirun -n 4 ./install/bin/test_tilexr_udma

# 8 进程测试（如果有 8 张卡）
mpirun -n 8 ./install/bin/test_tilexr_udma
```

**预期输出** (2 进程):
```
========================================
  TileXR UDMA Integration Tests
========================================
Environment:
  RANK: 0
  RANK_SIZE: 2
  PID: 12345
Using device: 0

[... Rank 0 测试输出 ...]

========================================
  TileXR UDMA Integration Tests
========================================
Environment:
  RANK: 1
  RANK_SIZE: 2
  PID: 12346
Using device: 1

[... Rank 1 测试输出 ...]

=== Test Case: Multi-Rank Initialization ===
Rank 0/2 starting...
Rank 1/2 starting...
[PASS] TileXRInit should succeed on rank 0
[PASS] TileXRInit should succeed on rank 1
[PASS] First sync should succeed on rank 0
Rank 0 passed first sync
[PASS] First sync should succeed on rank 1
Rank 1 passed first sync
[PASS] Second sync should succeed on rank 0
Rank 0 passed second sync
[PASS] Second sync should succeed on rank 1
Rank 1 passed second sync
Rank 0 finalized
Rank 1 finalized
```

---

## 测试说明

### 单元测试 (test_shmem_api)

#### Test 1: API Parameter Validation
**目的**: 验证 API 参数检查  
**测试点**:
- NULL 指针参数应返回 `ACLSHMEM_INVALID_PARAM`
- 验证所有参数组合

#### Test 2: Uninitialized State
**目的**: 验证未初始化状态处理  
**测试点**:
- 在 shmem 初始化前调用 API 应返回 `ACLSHMEM_INNER_ERROR`

#### Test 3: Full Initialization Flow
**目的**: 验证完整的初始化流程  
**测试点**:
- ACL 设备设置
- shmem unique ID 获取
- shmem 属性配置
- UDMA 引擎启用
- shmem 初始化
- UDMA 信息获取
- 设备内存验证
- 资源清理

#### Test 4: Multiple Calls Consistency
**目的**: 验证多次调用的一致性  
**测试点**:
- 多次调用返回相同的指针
- 多次调用返回相同的大小
- 验证幂等性

### 集成测试 (test_tilexr_udma)

#### Test 1: TileXR Basic Initialization
**目的**: 验证 TileXR 基本初始化  
**测试点**:
- `TileXRInit()` 成功
- `TileXRSync()` 成功
- 资源正确清理

#### Test 2: UDMA Initialization
**目的**: 验证 UDMA 集成  
**测试点**:
- CommArgs 指针有效
- UDMA 信息已设置（需要进一步验证）

#### Test 3: Multi-Rank Initialization
**目的**: 验证多进程协调  
**测试点**:
- 所有 rank 成功初始化
- 同步点正常工作
- 进程间协调正确

#### Test 4: Shared Memory Buffers
**目的**: 验证共享内存缓冲区  
**测试点**:
- 发送缓冲区有效
- 接收缓冲区有效
- 缓冲区地址不同

#### Test 5: Stress Test
**目的**: 压力测试  
**测试点**:
- 多次初始化/清理循环
- 资源泄漏检测
- 稳定性验证

---

## 故障排查

### 常见问题

#### 1. 找不到 libshmem.so

**错误**:
```
error while loading shared libraries: libshmem.so: cannot open shared object file
```

**解决方案**:
```bash
export LD_LIBRARY_PATH=/home/TileXR/3rdparty/shmem/install/shmem/lib:${LD_LIBRARY_PATH}
```

#### 2. 找不到 libtile-comm.so

**错误**:
```
error while loading shared libraries: libtile-comm.so: cannot open shared object file
```

**解决方案**:
```bash
export LD_LIBRARY_PATH=/home/TileXR/install/lib:${LD_LIBRARY_PATH}
```

#### 3. aclshmemx_get_udma_info 返回 INNER_ERROR

**可能原因**:
- shmem 未正确初始化
- UDMA 未启用（需要设置 `data_op_engine_type = ACLSHMEM_DATA_OP_UDMA`）
- 硬件不支持 UDMA（需要 Ascend 950+）
- shmem 构建时未启用 UDMA 支持

**调试步骤**:
```bash
# 1. 检查 shmem 构建配置
cd /home/TileXR/3rdparty/shmem
cat build/CMakeCache.txt | grep UDMA

# 2. 检查 shmem 初始化状态
# 在测试代码中添加:
int status = aclshmemx_init_status();
printf("shmem init status: %d\n", status);

# 3. 检查硬件支持
npu-smi info | grep "Chip Name"
```

#### 4. 多进程测试失败

**可能原因**:
- MPI 环境配置问题
- 设备绑定冲突
- 进程间通信失败

**调试步骤**:
```bash
# 1. 测试 MPI 基本功能
mpirun -n 2 hostname

# 2. 检查设备可见性
mpirun -n 2 npu-smi info

# 3. 使用详细日志
export ASCEND_GLOBAL_LOG_LEVEL=0
mpirun -n 2 ./install/bin/test_tilexr_udma
```

#### 5. 设备内存不足

**错误**:
```
aclrtMalloc failed: 500002
```

**解决方案**:
```bash
# 1. 检查设备内存使用
npu-smi info

# 2. 清理其他进程
pkill -9 test_

# 3. 重置设备
npu-smi -i 0 -r
```

### 日志分析

#### 启用详细日志

```bash
# 设置日志级别（0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR）
export ASCEND_GLOBAL_LOG_LEVEL=0

# 设置日志路径
export ASCEND_PROCESS_LOG_PATH=/home/TileXR/run/plog/$(date +%y%m%d%H%M)

# 运行测试
./install/bin/test_shmem_api
```

#### 查看日志

```bash
# 查看最新日志
ls -lt /home/TileXR/run/plog/ | head -5

# 搜索错误
grep -r "ERROR" /home/TileXR/run/plog/$(date +%y%m%d%H%M)/

# 搜索 UDMA 相关日志
grep -r "UDMA\|udma" /home/TileXR/run/plog/$(date +%y%m%d%H%M)/
```

---

## 预期结果

### 成功标准

#### 单元测试
- ✅ 所有测试用例通过 (16/16)
- ✅ UDMA 信息指针非空
- ✅ UDMA 信息大小 > 0
- ✅ 指针指向设备内存
- ✅ 多次调用返回一致结果

#### 集成测试（单进程）
- ✅ 所有测试用例通过 (10/10)
- ✅ TileXR 初始化成功
- ✅ CommArgs 指针有效
- ✅ 共享内存缓冲区有效
- ✅ 压力测试全部通过

#### 集成测试（多进程）
- ✅ 所有 rank 测试通过
- ✅ 进程间同步正常
- ✅ 无死锁或超时
- ✅ 资源正确清理

### 性能指标

虽然这些测试主要关注功能正确性，但也可以观察以下性能指标：

- **初始化时间**: < 1 秒
- **同步延迟**: < 100 毫秒
- **内存占用**: 每个 rank 约 200 MB
- **多次初始化**: 无明显性能下降

---

## 下一步

测试通过后，可以进行：

1. **性能基准测试**: 对比 UDMA vs MTE vs RDMA 的性能
2. **通信功能测试**: 测试实际的点对点和集合通信
3. **DeepEP 集成**: 基于 TileXR 实现 DeepEP 昇腾版本
4. **长时间稳定性测试**: 运行数小时验证稳定性

---

## 附录

### 测试环境信息收集

运行测试前，建议收集以下信息：

```bash
#!/bin/bash
echo "=== System Information ==="
uname -a
cat /etc/os-release | grep PRETTY_NAME

echo -e "\n=== NPU Information ==="
npu-smi info

echo -e "\n=== CANN Version ==="
cat ${ASCEND_HOME_PATH}/version.info

echo -e "\n=== Driver Version ==="
cat /usr/local/Ascend/driver/version.info

echo -e "\n=== Library Versions ==="
ls -lh /home/TileXR/3rdparty/shmem/install/shmem/lib/libshmem.so
ls -lh /home/TileXR/install/lib/libtile-comm.so

echo -e "\n=== Environment Variables ==="
env | grep -E "ASCEND|TILEXR|LD_LIBRARY_PATH"
```

保存输出以便问题排查。

---

**文档版本**: 1.0  
**最后更新**: 2026-05-25  
**维护者**: TileXR Team
