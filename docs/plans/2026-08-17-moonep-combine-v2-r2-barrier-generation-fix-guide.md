# MoonEP Combine V2 R2 Barrier Generation 复用问题修复指导

状态：根因已确认，待按本文修复和验证

日期：2026-08-17

关联方案：

- `docs/plans/2026-08-16-moonep-combine-v2-server-pair-mapping.md`
- `docs/specs/2026-08-14-moonep-combine-v2-full-sync-design.md`
- `docs/moonep/MINDSPEED_DEBUGGING_EXPERIENCE.md`

## 1. 目标与范围

修复 R2 在 32P、64P 和 128P 上停用 Legacy Grant 后暴露的 full-sync generation
提前复用问题，使 leading/round global barrier 的两个 generation slot 按实际执行顺序
安全交替。

必须保持：

- R2 继续使用 bidirectional Ring；
- 32P+ 继续停用 Legacy Grant；
- 2P-16P 继续保留 Legacy Grant；
- phase barrier 仍留到 R4，不在 R2 提前启用；
- boundary ID、完整 boundary guard、公共 API 和 21 参数 Kernel ABI 不变；
- `WaitStepCqs()` 的 `cqTail == cqTarget`、`tail == submittedHead`、
  `head == tail` 终态检查不变；
- 不修改 `tools/moonep/cab0_2_128p/compile.sh`、`sync_runtime.sh` 和 `run.sh`。

非目标：

- 不切换 R3 Server-Pair；
- 不接入 R6-R8 64-Grant；
- 不通过增加 timeout、跳过 CQ/Done 或关闭错误检查规避失败；
- 不扩大 full-sync workspace，除非推荐方案经验证不可行并重新评审。

## 2. 已确认的失败证据

R2 源码与主服务器源码 SHA-256 一致，现有 schedule、layout、host、launch、public ABI、
source guard 六个 Host 测试均通过，但 128P 实机正确性失败。

关键日志：

```text
/home/h00580772/tilexr_combine_v2_cab0_2_fast/logs/
combine_v2_128p_noprofile_20260816_234049.log

/home/h00580772/tilexr_combine_v2_cab0_2_fast/logs/
r2_small_diag_clean.log
```

production case 完成 80 次迭代但 `correctness=failed`。small case 中所有 rank 约在 10 秒
边界退出。rank0 的输出具有确定性断点：

```text
source 1..20: 正确
source 0: Self 缺失
source 21: 第一个远端缺失
```

128P bidirectional receive 映射为：

```text
source distance 1..4   -> step 0
source distance 5..8   -> step 1
source distance 9..12  -> step 2
source distance 13..16 -> step 3
source distance 17..20 -> step 4
source distance 21     -> step 5
Self                   -> step 7
```

因此数据 step0-step4 已完成，执行在 step5 前停止。该断点与 step4 后 boundary6 的
generation 复用竞态一致。

## 3. 根因

当前 boundary 编号为未来 R4 phase barrier 保留一个空位：

```cpp
return round < phaseStepCount ? 1U + round : 2U + round;
```

generation 却直接按 boundary ID 取模：

```cpp
generation = boundaryId % 2;
```

R2 不执行预留的 phase barrier，因此相邻的实际 barrier 会跳过一个 ID，并立即复用同一
generation slot：

| Rank 数 | 实际 boundary 序列 | generation 序列 | 非法复用 |
| ---: | --- | --- | --- |
| 32P | `0, 1, 3` | `0, 1, 1` | `1 -> 3` |
| 64P | `0, 1, 2, 4, 5` | `0, 1, 0, 0, 1` | `2 -> 4` |
| 128P | `0, 1, 2, 3, 4, 6, 7, 8, 9` | `0, 1, 0, 1, 0, 0, 1, 0, 1` | `4 -> 6` |

一次 distributed barrier 只能证明本 rank 已收到当前 boundary 的全部信号，不能证明
所有其他 rank 都已经消费了本 rank 发布的信号。竞态顺序如下：

```text
fast rank 完成 boundary4
-> fast rank 完成 step4
-> fast rank 把 gen0 source slot 改写为 boundary6
-> slow rank 仍在等待同一 slot 的 boundary4
-> slow rank 看到 boundary6，完整 guard 正确拒绝
-> boundary4 已被覆盖，slow rank 最终 timeout
-> 其他 rank 在 boundary6 等不到 slow rank，随后统一失败
```

完整 boundary guard 能拒绝 stale 或未来信号，但不能恢复已被覆盖的旧信号。因此问题不在
guard 字段，而在 slot 生命周期。

## 4. 必须建立的不变量

修复后必须同时满足：

1. `boundaryId` 是协议身份，继续写入 signal 并参与完整 guard；
2. `generation` 是存储槽身份，只由实际执行 barrier 的 ordinal 决定；
3. 相邻的实际 barrier 必须使用不同 generation；
4. 同一 generation 再次使用前，中间必须完整执行另一个 distributed barrier；
5. 不通过清零 signal 解决复用，继续使用 magic、boundary 和 guard 拒绝 stale 数据；
6. 两槽安全证明必须基于实际执行顺序，不能基于预留但未执行的 boundary ID。

两槽成立的原因是：rank A 若要完成 ordinal `n+1`，必须收到 rank B 发布的 `n+1` 信号；
rank B 只有完成 ordinal `n` 后才能发布该信号。因此 A 不可能在 B 尚未完成 `n` 时进入
ordinal `n+2` 并复用 ordinal `n` 的槽。

## 5. 推荐修复设计

将 boundary identity 与 execution ordinal 显式分离。

R1-R3 的实际执行 ordinal：

```text
leading barrier: ordinal = 0
round r:         ordinal = r + 1
generation:      ordinal % 2
```

推荐让 `RunGlobalBarrier` 同时接收 `boundaryId` 和 `barrierOrdinal`，内部只使用：

```cpp
generation = MoonEpCombineV2BarrierGeneration(barrierOrdinal);
```

用途必须分离：

```text
boundaryId:
  signal.boundaryId
  FullSyncGuard
  failure step/stage
  collective status stageId

barrierOrdinal:
  generation slot selection only
```

调用关系：

```cpp
RunGlobalBarrier(/* boundaryId */ 0U,
    /* barrierOrdinal */ 0U, ...);

const uint32_t boundaryId = MoonEpCombineV2RoundBoundary(step, rankSize_);
RunGlobalBarrier(boundaryId,
    /* barrierOrdinal */ step + 1U, ...);
```

不得把 `boundaryId` 改成连续值来掩盖问题。保留当前 boundary 编号可避免 R4 启用 phase
barrier 时改变 profile、trace、failure record 和 signal guard 的协议身份。

R4 增加真实 phase barrier 时，必须重新按实际顺序分配 ordinal：

```text
leading
-> phase0 rounds
-> phase barrier
-> phase1 rounds
```

此时 boundary ID 可以保持现有 `0..9` 语义，generation 仍只消费实际 ordinal。

### 备选方案

为每个 boundary 分配独立 slot 可以消除复用竞态，但会扩大 workspace、修改 Host layout
和更多测试，R2 不需要承担该改动。除非两槽序列方案被硬件证据推翻，否则不采用。

提前执行预留 phase barrier 会把 R4 行为并入 R2，违反阶段隔离，也不采用。

## 6. 逐文件修改指导

### 6.1 Schedule helper

文件：`src/moonep/common/moonep_combine_schedule.h`

- 保留 `MoonEpCombineV2RoundBoundary()` 的 boundary ID 结果；
- 增加或重命名 helper，明确输入是 execution ordinal；
- 禁止调用方再用 `MoonEpCombineV2FullSyncGeneration(boundaryId)`；
- helper 保持 Host/AICore 共用、C++14 可编译、无 Host-only STL。

### 6.2 Kernel barrier

文件：`src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h`

- 扩展内部 `RunGlobalBarrier` 参数，分开传递 boundary ID 和 ordinal；
- leading barrier 传 `(0, 0)`；
- R2 round `step` 传 `(MoonEpCombineV2RoundBoundary(step, rankSize_), step + 1)`；
- signal、guard、receive matching 和 collective status 继续使用 boundary ID；
- full-sync source/receive index只使用由 ordinal 得到的 generation；
- 不改变 16 核 `SyncAll<true>()` 数量和顺序；
- 不改变 Legacy Grant、Done、CQ/SQ 或数据路由代码。

### 6.3 Schedule unit test

文件：`tests/moonep_combine_v2/unit/test_combine_v2_schedule.cpp`

新增对实际 barrier 序列的组合测试，而不是分别测试 boundary 和 `% 2`：

```text
32P  generations: 0,1,0
64P  generations: 0,1,0,1,0
128P generations: 0,1,0,1,0,1,0,1,0
```

必须断言：

- 相邻实际 barrier 的 generation 不同；
- boundary4 和 boundary6 guard 不同；
- boundary4 和 boundary6 使用不同 generation；
- boundary ID 仍保留 phase 空位；
- 非法 ordinal 不产生越界 generation。

### 6.4 Source guard

文件：`tests/moonep_combine_v2/unit/test_combine_v2_source_guard.cpp`

- 要求 `RunGlobalBarrier` 显式消费独立 ordinal；
- 拒绝从 `boundaryId` 直接计算 generation；
- 保留所有 16 核 barrier 对称、UB WQE、MTE3、`st_dev` 和 CQ 终态检查；
- 保留 R2 对 Legacy Grant publish/wait 的 rank-size gate。

## 7. 实施顺序与验收

### 任务 1：先增加失败测试

目标：用 Host oracle 精确复现 `boundary4 -> boundary6` 同槽问题。

Acceptance：新测试在当前代码失败，并打印 rank size、round、boundary、ordinal 和
generation；原有 schedule 映射测试仍通过。

### 任务 2：修复 generation plumbing

目标：只分离 boundary ID 和 execution ordinal，不改数据路径。

Acceptance：32P/64P/128P 的实际 generation 序列严格交替；Kernel 中完整 boundary guard
仍使用原 boundary ID。

### 任务 3：Host 与 target compile

在 Git Bash 中执行：

```bash
cmake -S tests/moonep_combine_v2 -B build-combine-v2-host
cmake --build build-combine-v2-host -j"$(nproc)"
ctest --test-dir build-combine-v2-host --output-on-failure
```

若远端 CMake 包没有 `ctest`，直接运行 schedule、layout、host、launch、public ABI 和
source guard 六个二进制。

Acceptance：Host 测试全部通过；CANN 9.1 AICore target compile 通过；Kernel ABI 仍为
21 个 64-bit 参数。

### 任务 4：硬件验证

先复测能快速证明根因的 128P small case：

```text
BS128, H3584, K16, BF16, no-reduce, warmup0, iteration1
```

必须确认 source21 和 Self 均恢复，且无 boundary timeout、CQ/SQ outstanding 或
failure record。

随后按 R2 阶梯执行：

```text
32P small/production
-> 64P small/production
-> 128P small/production
-> epoch0 -> epoch1 -> epoch0
-> 20 warmup + 80 iterations stress
```

正确性以每 rank 输出为准，不以 SSH 或 controller 退出码单独判断。

## 8. 标准远端闭环

Mutagen session 必须保持 `one-way-safe`。三个远端脚本保持现状，只执行：

```bash
mutagen sync flush tilexr-combine-v2-cab0-2-fast
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/compile.sh'
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/sync_runtime.sh'
ssh root@141.61.53.150 'bash /home/h00580772/tilexr_combine_v2_cab0_2_fast/tools/moonep/cab0_2_128p/run.sh'
```

运行前若 NPU 只有其他 `tilexr_*` 任务可直接共用；若存在占用 NPU 的 Python 或其他通信
测试，每 15 秒复查，最多等待 120 秒，不终止其他用户任务。

## 9. 完成标准

只有以下全部成立才能恢复 R2 acceptance：

1. boundary ID 与 generation ordinal 已在 API 和命名上分离；
2. 32P/64P/128P 实际 barrier 序列不存在相邻同 generation；
3. 完整 boundary guard 和 stale rejection 保持；
4. 32P+ 不读写 Legacy Grant receive/source；
5. 2P-16P Legacy Grant 行为不变；
6. 128P source21、Self 和所有其他 source 正确；
7. 32P/64P/128P 每 rank correctness 通过；
8. CQ/SQ、Done、boundary 和 epoch 复用无 timeout 或 outstanding；
9. CANN 9.1 target compile 和 128P 80 iterations stress 通过；
10. 未提前引入 R3/R4/R6-R8 行为，未修改三个远端脚本。

若修复后首个失败边界发生变化，应保存新日志并重新按 CQ、Done、boundary、Reduce 顺序
定位，不继续叠加 speculative patch。
