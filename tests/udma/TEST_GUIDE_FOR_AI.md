# TileXR UDMA 测试代码说明

**创建时间**: 2026-05-25  
**目标**: 为另一个 AI 提供完整的测试执行指南

---

## 📦 测试代码概览

### 文件结构

```
tests/udma/
├── CMakeLists.txt              # CMake 构建配置
├── build.sh                    # 自动构建脚本
├── run_tests.sh                # 自动测试脚本
├── README.md                   # 详细测试文档（1500+ 行）
├── QUICKSTART.md               # AI 快速指南（200+ 行）
├── unit/
│   └── test_shmem_api.cpp     # shmem API 单元测试（250+ 行）
└── integration/
    └── test_tilexr_udma.cpp   # TileXR 集成测试（300+ 行）
```

### 测试覆盖

#### 单元测试 (test_shmem_api.cpp)

测试 `aclshmemx_get_udma_info()` API 的所有方面：

1. **参数验证** (3 个测试)
   - NULL 指针检测
   - 参数组合验证
   - 错误码正确性

2. **状态检查** (1 个测试)
   - 未初始化状态处理
   - 错误返回验证

3. **完整流程** (9 个测试)
   - ACL 初始化
   - shmem 初始化
   - UDMA 启用
   - 信息获取
   - 设备内存验证
   - 资源清理

4. **一致性验证** (3 个测试)
   - 多次调用返回相同指针
   - 多次调用返回相同大小
   - 幂等性验证

**总计**: 16 个断言

#### 集成测试 (test_tilexr_udma.cpp)

测试 TileXR 的 UDMA 集成：

1. **基本初始化** (2 个测试)
   - TileXRInit 成功
   - TileXRSync 成功

2. **UDMA 集成** (2 个测试)
   - CommArgs 指针有效
   - UDMA 信息已设置

3. **多进程协调** (4+ 个测试)
   - 所有 rank 初始化成功
   - 同步点正常工作
   - 进程间协调正确
   - 资源正确清理

4. **共享内存** (4 个测试)
   - 发送缓冲区有效
   - 接收缓冲区有效
   - 缓冲区地址不同
   - 缓冲区可访问

5. **压力测试** (1 个测试)
   - 5 次初始化/清理循环
   - 无资源泄漏
   - 稳定性验证

**总计**: 10+ 个断言（单进程），更多（多进程）

---

## 🚀 执行指南

### 给 AI 的指令

**任务**: 在昇腾环境上执行 TileXR UDMA 测试

**步骤**:

1. **阅读快速指南**
   ```bash
   cat /home/TileXR/tests/udma/QUICKSTART.md
   ```

2. **执行测试**
   ```bash
   cd /home/TileXR/tests/udma
   bash build.sh
   bash run_tests.sh
   ```

3. **收集结果**
   - 保存完整输出
   - 记录所有 PASS/FAIL
   - 注意任何警告或异常

4. **报告结果**
   - 使用 QUICKSTART.md 中的报告模板
   - 包含环境信息
   - 包含完整日志

### 预期执行时间

- **构建**: 2-5 分钟
- **单元测试**: 30-60 秒
- **集成测试（单进程）**: 30-60 秒
- **集成测试（多进程）**: 1-2 分钟
- **总计**: 5-10 分钟

### 成功标准

```bash
# 构建成功
$ bash build.sh
# 退出码: 0

# 测试成功
$ bash run_tests.sh
Test 1 (shmem API):        PASS
Test 2 (TileXR Single):    PASS
Test 3 (TileXR Multi):     PASS
# 退出码: 0
```

---

## 🔍 测试重点

### 关键验证点

1. **UDMA 信息获取**
   ```cpp
   void* udmaInfoPtr = nullptr;
   size_t udmaInfoSize = 0;
   int ret = aclshmemx_get_udma_info(&udmaInfoPtr, &udmaInfoSize);
   
   // 验证点:
   // - ret == ACLSHMEM_SUCCESS
   // - udmaInfoPtr != nullptr
   // - udmaInfoSize > 0
   // - 指针指向设备内存
   ```

2. **TileXR 初始化**
   ```cpp
   int ret = TileXRInit(rank, rankSize);
   
   // 验证点:
   // - ret == TILEXR_SUCCESS
   // - CommArgs 指针有效
   // - UDMA 信息已设置
   ```

3. **多进程同步**
   ```cpp
   ret = TileXRSync();
   
   // 验证点:
   // - 所有 rank 都成功
   // - 无死锁
   // - 无超时
   ```

### 预期输出示例

#### 单元测试成功输出

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

========================================
  Test Summary
========================================
Total:  16
Passed: 16
Failed: 0
========================================
```

#### 集成测试成功输出

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
[PASS] Recv buffer should not be NULL
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

---

## 🐛 故障排查

### 常见问题及解决方案

#### 1. 构建失败

**症状**: `build.sh` 失败，找不到库或头文件

**检查**:
```bash
# 检查 shmem 库
ls -lh /home/TileXR/3rdparty/shmem/install/shmem/lib/libshmem.so

# 检查 TileXR 库
ls -lh /home/TileXR/install/lib/libtile-comm.so

# 检查环境变量
echo $ASCEND_HOME_PATH
echo $ARCH
```

**解决**:
```bash
# 重新构建依赖
cd /home/TileXR/3rdparty/shmem
bash scripts/build.sh -soc_type Ascend950

cd /home/TileXR
source common_env.sh
rm -rf build && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc) && make install
```

#### 2. 运行时库找不到

**症状**: `error while loading shared libraries: libshmem.so`

**解决**:
```bash
export LD_LIBRARY_PATH=/home/TileXR/3rdparty/shmem/install/shmem/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/TileXR/install/lib:${LD_LIBRARY_PATH}
```

#### 3. UDMA 初始化失败

**症状**: `aclshmemx_get_udma_info` 返回 `INNER_ERROR`

**可能原因**:
- shmem 未启用 UDMA 支持
- 硬件不支持 UDMA
- 初始化顺序错误

**检查**:
```bash
# 检查 shmem 构建配置
cd /home/TileXR/3rdparty/shmem
cat build/CMakeCache.txt | grep UDMA

# 检查硬件
npu-smi info | grep "Chip Name"
# 应该看到 Ascend 950 或更高
```

#### 4. 多进程测试失败

**症状**: MPI 测试超时或死锁

**检查**:
```bash
# 测试 MPI 基本功能
mpirun -n 2 hostname

# 检查设备数量
lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | wc -l
```

**解决**:
- 确保有足够的 NPU 设备
- 检查 MPI 配置
- 查看详细日志

---

## 📊 测试报告模板

执行测试后，请提供以下信息：

```markdown
## 测试执行报告

### 环境信息
- 操作系统: [uname -a 输出]
- NPU 型号: [npu-smi info | grep "Chip Name" 输出]
- NPU 数量: [数量]
- CANN 版本: [版本号]
- 驱动版本: [版本号]

### 构建结果
- 状态: [成功/失败]
- 时间: [X 秒]
- 错误: [如果失败，粘贴错误信息]

### 测试结果
- Test 1 (shmem API): [PASS/FAIL]
  - Total: 16
  - Passed: [数量]
  - Failed: [数量]
  
- Test 2 (TileXR Single): [PASS/FAIL]
  - Total: 10
  - Passed: [数量]
  - Failed: [数量]
  
- Test 3 (TileXR Multi): [PASS/SKIP/FAIL]
  - Ranks: [数量]
  - 状态: [描述]

### 关键输出
```
[粘贴 UDMA Info Pointer, Size, Memory Type 等关键信息]
```

### 完整日志
```
[粘贴 run_tests.sh 的完整输出]
```

### 问题和观察
[记录任何异常、警告或值得注意的现象]

### 结论
[总体评估：成功/部分成功/失败]
```

---

## 📚 文档索引

- **QUICKSTART.md**: AI 快速执行指南（推荐首先阅读）
- **README.md**: 完整测试文档（详细说明和故障排查）
- **本文档**: 测试代码说明和执行总结

---

## 🎯 测试目标

这些测试旨在验证：

1. ✅ shmem 的 `aclshmemx_get_udma_info()` API 正确实现
2. ✅ TileXR 正确集成 shmem UDMA 功能
3. ✅ UDMA 设备内存指针正确传递
4. ✅ 多进程环境下的 UDMA 初始化和同步
5. ✅ 资源管理和清理正确
6. ✅ 稳定性和可靠性

---

## 💡 给 AI 的提示

1. **按顺序执行**: 先构建，再测试
2. **保存日志**: 使用 `tee` 保存完整输出
3. **注意细节**: 关注 PASS/FAIL 状态和关键数值
4. **完整报告**: 即使测试失败，也要提供完整信息
5. **环境检查**: 测试前确认环境正确配置

---

**祝测试顺利！如有问题，请提供完整的测试报告和日志。** 🚀
