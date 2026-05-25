# TileXR UDMA 测试 - AI 快速指南

**目标读者**: 在昇腾环境上执行测试的 AI Agent  
**预计时间**: 15-30 分钟  
**前置条件**: 已完成 TileXR 和 shmem 的构建

---

## 🚀 快速开始（3 步）

### Step 1: 进入测试目录

```bash
cd /home/TileXR/tests/udma
```

### Step 2: 构建测试

```bash
bash build.sh
```

**预期输出**:
```
==========================================
  Building UDMA Tests
==========================================
[构建日志...]
==========================================
  Build Complete
==========================================
Test binaries installed to: /home/TileXR/tests/udma/install/bin
```

### Step 3: 运行测试

```bash
bash run_tests.sh
```

**预期输出**:
```
==========================================
  Running UDMA Tests
==========================================
[测试日志...]
==========================================
  Test Results Summary
==========================================
Test 1 (shmem API):        PASS
Test 2 (TileXR Single):    PASS
Test 3 (TileXR Multi):     PASS
==========================================
```

---

## ✅ 成功标准

测试成功的标志：

1. **构建成功**: `build.sh` 退出码为 0
2. **所有测试通过**: `run_tests.sh` 输出 "PASS"
3. **无错误日志**: 没有 "ERROR" 或 "FAIL" 输出
4. **退出码为 0**: `echo $?` 返回 0

---

## ❌ 失败处理

### 如果构建失败

```bash
# 检查依赖库
ls -lh /home/TileXR/3rdparty/shmem/install/shmem/lib/libshmem.so
ls -lh /home/TileXR/install/lib/libtile-comm.so

# 如果库不存在，重新构建
cd /home/TileXR/3rdparty/shmem
bash scripts/build.sh -soc_type Ascend950

cd /home/TileXR
source common_env.sh
rm -rf build && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc) && make install
```

### 如果测试失败

```bash
# 1. 检查环境
source /home/TileXR/common_env.sh
npu-smi info

# 2. 查看详细日志
export ASCEND_GLOBAL_LOG_LEVEL=0
bash run_tests.sh 2>&1 | tee test_output.log

# 3. 手动运行单个测试
export LD_LIBRARY_PATH=/home/TileXR/3rdparty/shmem/install/shmem/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/TileXR/install/lib:${LD_LIBRARY_PATH}
./install/bin/test_shmem_api
```

---

## 📊 测试报告模板

测试完成后，请提供以下信息：

```
## 测试环境
- 操作系统: [uname -a]
- NPU 型号: [npu-smi info | grep "Chip Name"]
- NPU 数量: [lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | wc -l]
- CANN 版本: [cat ${ASCEND_HOME_PATH}/version.info]
- 驱动版本: [cat /usr/local/Ascend/driver/version.info]

## 构建结果
- 构建状态: [成功/失败]
- 构建时间: [X 秒]
- 构建日志: [如果失败，粘贴关键错误]

## 测试结果
- Test 1 (shmem API): [PASS/FAIL]
- Test 2 (TileXR Single): [PASS/FAIL]
- Test 3 (TileXR Multi): [PASS/SKIP/FAIL]

## 详细输出
[粘贴 run_tests.sh 的完整输出]

## 问题和观察
[记录任何异常、警告或值得注意的现象]
```

---

## 🔍 关键检查点

在运行测试时，请特别注意：

### 1. UDMA 信息获取
```
UDMA Info Pointer: 0x7f8a40000000  ← 应该是非零地址
UDMA Info Size: 12345 bytes        ← 应该 > 0
Memory Type: 1                     ← 应该是 1 (DEVICE)
```

### 2. 多进程同步
```
Rank 0 passed first sync           ← 所有 rank 都应该通过
Rank 1 passed first sync
Rank 0 passed second sync
Rank 1 passed second sync
```

### 3. 测试统计
```
Total:  16                         ← 单元测试应该有 16 个断言
Passed: 16                         ← 应该全部通过
Failed: 0                          ← 应该为 0
```

---

## 🐛 常见问题速查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `libshmem.so: cannot open` | 库路径未设置 | `export LD_LIBRARY_PATH=...` |
| `INNER_ERROR` | UDMA 未启用 | 检查 shmem 构建配置 |
| `Device not found` | NPU 不可用 | `npu-smi info` 检查设备 |
| MPI 测试 SKIP | 只有 1 张卡 | 正常，单卡环境会跳过 |
| 测试超时 | 进程死锁 | Ctrl+C 后查看日志 |

---

## 📝 执行清单

测试前请确认：

- [ ] 已切换到正确的分支 (`feature/udma-integration`)
- [ ] 已构建 shmem 库 (tilexr-udma-integration 分支)
- [ ] 已构建 TileXR 库
- [ ] 已加载环境变量 (`source common_env.sh`)
- [ ] NPU 设备可用 (`npu-smi info`)

测试后请确认：

- [ ] 所有测试通过或记录失败原因
- [ ] 收集了完整的测试输出
- [ ] 检查了日志文件（如果有错误）
- [ ] 填写了测试报告

---

## 💡 提示

1. **首次运行**: 如果是第一次运行测试，建议先手动执行每个步骤，观察输出
2. **日志保存**: 使用 `bash run_tests.sh 2>&1 | tee test_output.log` 保存完整日志
3. **环境隔离**: 每次测试前建议重启终端或重新 source 环境
4. **设备重置**: 如果测试失败，可以尝试 `npu-smi -i 0 -r` 重置设备

---

## 📞 获取帮助

如果遇到问题：

1. 查看详细文档: `cat README.md`
2. 检查构建日志: `cat build/CMakeFiles/CMakeError.log`
3. 查看运行日志: `ls -lt /home/TileXR/run/plog/`
4. 提供完整的测试报告和日志

---

**祝测试顺利！** 🎉
