# All-to-All UDMA 多核实现 — 验证成功记录

> 记录 TileXR UDMA all-to-all kernel 从单核标量拷贝改造为 `rankSize` 核并行版本的设计、验证流程与性能结果。
> 验证环境:8 × Ascend950DT(A5 / 3510 架构),CANN 25.1,bisheng 编译器。
> 验证日期:2026-06-25。

---

## 1. 改造背景

原始 `tilexr_udma_all_to_all_kernel` 存在两个问题:

1. **self-copy 用标量逐元素拷贝** — 本 rank 自己那一块 `for (i) selfDst[i] = selfSrc[i];`,效率极低,违反 Ascend C API 最佳实践(黑名单 `SetValue`/标量 GM 写)。
2. **单核串行** — host 侧 `launch_tilexr_udma_all_to_all(1, ...)`,blockDim=1,所有 peer 的 UDMA PUT + self-copy 在 1 个 AI Core 上串行,未利用多核。

改造目标:

- self-copy 改为 `DataCopyPad` 批量搬运(经 UB 中转,MTE2/MTE3 同步)。
- 整个 kernel 改为 `rankSize` 核并行,每个 core 负责一个 peer 的搬运,压力分摊到 1/N。

---

## 2. 算法流程

### 2.1 整体语义

all-to-all:每个 rank 把自己 input 中**属于各个 peer 的那一块**送到**对方 rank 的 output 区域**。即 `output[peer][rank] = input[rank][peer]`。

- input 按 `[dstRank]` 切分:本 rank 持有所有 peer 的数据
- output 按 `[srcRank]` 切分:本 rank 的 output 接收来自各 rank 的数据

### 2.2 多核分片策略

host 启动 `blockDim = rankSize` 个 block。kernel 内用 `AscendC::GetBlockIdx()` 取当前 block id,通过 stride 循环分配 peer:

```
for (peer = blockIdx; peer < rankSize; peer += blockNum)
```

当 `blockNum == rankSize` 时,block b 处理 peer b:

| block id (b) | peer == rank? | 执行路径 |
|---|---|---|
| b == rank | 是 | **本地 self-copy**(DataCopyPad,无网络) |
| b != rank | 否 | **UDMA PUT 到 peer b**(硬件 DMA) |

- 每个 core 只处理 1 个 peer,N 次搬运分摊到 N 个 core。
- 不同 core 操作不同 peer 的发送队列(SQ),无 `headAddr`/`wqeCnt` 竞争。
- 降级兼容:`blockNum < rankSize` 时 stride 循环让每个 core 处理多个 peer,逻辑仍正确。

### 2.3 阶段详解

**阶段 0:初始化与守卫**
- 解析 `CommArgs`,取 `rank`/`rankSize`/UDMA registry 指针。
- `UDMARegistryEnabled(args)` 检查使能位 + registry 非空;未使能则 block 0 写完 debug 后 return。
- **debug header 只由 block 0 写**(`blockIdx == 0`),避免多核并发写 `debug[0..5]` 竞争。

**阶段 1:self-copy 分支(peer == rank)**
- 本 rank 自己那一块,不经网络,纯本地搬移。
- `TPipe` + `TBuf<VECCALC>`(64KB UB)中转,64KB 分块:
  - CopyIn:GM → UB(`DataCopyPad`,MTE2 异步)→ `SetFlag/WaitFlag<MTE2_MTE3>` 同步
  - CopyOut:UB → GM(`DataCopyPad`,MTE3 异步)→ `SetFlag/WaitFlag<MTE3_MTE2>` 同步
- `DataCopyPad` 自动处理非对齐尾部,整 `bytes` 一次搬运,无标量补尾。
- 末尾 `PipeBarrier<PIPE_ALL>()` 确保本地拷贝完成。

**阶段 2:UDMA PUT 分支(peer != rank)**
- `localSrc = input + peer*elementsPerPeer + inputElementOffset`(本 rank 持有、属于该 peer 的数据)
- `remoteOffset = outputByteOffset + rank*payloadBytes`(对方 output 中本 rank 的位置)
- 取该 peer 的 WQ 上下文 `UDMAGetWQCtx(udmaInfo, peer, 0)`、远端 mem info。
- `UDMARegisteredRangeValid` 校验 `remoteOffset + bytes` 落在 host 注册给该 peer 的 region 内。
- **投递 WQE**:`UDMAPutNbi` → `UDMAWrite` → `UDMAPostSend`:
  - 在 SQ 环 `bufAddr + wqeSize*(curHead % depth)` 处填 `UDMASqeCtx`(远端地址/tpn/opcode=WRITE)+ `UDMASgeCtx`(本地源地址 + 长度)
  - `UDMACleanCacheLines` 刷 WQE cacheline,`curHead += wqeBbCnt`,写 doorbell,递增 `wqeCnt`
  - non-blocking:WQE 进 SQ 后硬件异步发起跨卡 DMA 写,GM→GM 直达
- **Quiet 同步**:`UDMAQuietStatus` → `UDMAPollCQ` 轮询该 peer 完成队列(CQ),直到本次 WQE 的 CQE 产生,确保数据已落盘到远端 GM。

### 2.4 同步与退出保证

- `SetFlag/WaitFlag` 成对使用,`EVENT_ID0`,与仓库内 fused IPC kernel 同范式(已验证可用)。
- self-copy 分支:event 配对完整(CopyIn 的 MTE2_MTE3 + CopyOut 的 MTE3_MTE2),循环内每轮自平衡,末尾 `PipeBarrier<PIPE_ALL>`。
- UDMA 分支:无 UB event,仅 `UDMAQuietStatus` 轮询 CQ,CQE 必然产生(硬件完成保证),不会无限阻塞。
- **每个 core 处理完自己的 peer 后循环结束、kernel 返回,正常退出。** msprof 实测 task 在 ~189μs 完成,无死锁。

---

## 3. 关键代码片段

### 3.1 kernel 主体(`tests/udma/demo/tilexr_udma_demo_kernel.cpp:122-229`)

```cpp
extern "C" __global__ __aicore__ void tilexr_udma_all_to_all_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR debugGM,
    int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset, int32_t chunkElements)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto input = reinterpret_cast<__gm__ int32_t*>(inputGM);
    auto output = reinterpret_cast<__gm__ int32_t*>(outputGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    int32_t rank = args->rank;
    int32_t rankSize = args->rankSize;
    bool enabled = TileXR::UDMARegistryEnabled(args);

    // Multi-core: one block per peer. Block b handles peer b:
    //   - peer == rank -> local self-copy via DataCopyPad
    //   - peer != rank -> UDMA PUT to that peer
    // Block 0 writes the shared debug header; per-peer slots are written by
    // their owning block only, so no cross-block debug races.
    const int32_t blockIdx = AscendC::GetBlockIdx();
    const int32_t blockNum = AscendC::GetBlockNum();

    if (blockIdx == 0 && debug != nullptr) {
        debug[0] = TILEXR_UDMA_DEMO_MAGIC;
        debug[1] = rank;
        debug[2] = rankSize;
        debug[3] = enabled ? 1 : 0;
        debug[4] = elementsPerPeer;
        debug[5] = static_cast<int32_t>(outputByteOffset);
    }
    if (!enabled) {
        return;
    }

    const int32_t effectiveChunkElements = chunkElements > 0 ? chunkElements : elementsPerPeer;
    const uint64_t payloadBytes = AllToAllPayloadBytes(effectiveChunkElements);
    const uint32_t bytes = static_cast<uint32_t>(payloadBytes);

    // This block's assigned peer. When host launches rankSize blocks, block b
    // handles peer b. If fewer blocks are launched, peers are round-robined
    // and each block may handle more than one peer (still correct, just less
    // parallel); stride == blockNum keeps peer slots disjoint across blocks.
    for (int32_t peer = blockIdx; peer < rankSize; peer += blockNum) {
        if (peer == rank) {
            // Self-copy: local DataCopyPad, no network.
            auto selfSrc = input + static_cast<uint64_t>(rank) * elementsPerPeer + inputElementOffset;
            auto selfDst = output + static_cast<uint64_t>(rank) * effectiveChunkElements;
            constexpr uint32_t SELF_COPY_UB_BYTES = 64 * 1024;
            AscendC::TPipe pipe;
            AscendC::TBuf<AscendC::QuePosition::VECCALC> selfCopyTBuf;
            pipe.InitBuffer(selfCopyTBuf, SELF_COPY_UB_BYTES);
            AscendC::LocalTensor<uint8_t> selfCopyLocal = selfCopyTBuf.Get<uint8_t>();

            auto selfSrcBytes = reinterpret_cast<__gm__ uint8_t*>(selfSrc);
            auto selfDstBytes = reinterpret_cast<__gm__ uint8_t*>(selfDst);
            for (uint32_t offset = 0; offset < bytes; offset += SELF_COPY_UB_BYTES) {
                uint32_t copyBytes = (bytes - offset < SELF_COPY_UB_BYTES)
                    ? (bytes - offset) : SELF_COPY_UB_BYTES;

                AscendC::GlobalTensor<uint8_t> srcGlobal;
                srcGlobal.SetGlobalBuffer(selfSrcBytes + offset);
                AscendC::DataCopyPadExtParams<uint8_t> padIn {false, 0U, 0U, 0};
                AscendC::DataCopyExtParams copyIn {1U, copyBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(selfCopyLocal, srcGlobal, copyIn, padIn);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                AscendC::GlobalTensor<uint8_t> dstGlobal;
                dstGlobal.SetGlobalBuffer(selfDstBytes + offset);
                AscendC::DataCopyExtParams copyOut {1U, copyBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(dstGlobal, selfCopyLocal, copyOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            continue;
        }

        // UDMA PUT to peer.
        auto localSrc = input + static_cast<uint64_t>(peer) * elementsPerPeer + inputElementOffset;
        uint64_t remoteOffset = outputByteOffset +
            static_cast<uint64_t>(rank) * payloadBytes;
        auto registry = TileXR::GetUDMARegistry(args);
        auto udmaInfo = TileXR::GetUDMAInfo(args);
        auto wqCtx = TileXR::UDMAGetWQCtx(udmaInfo, peer, 0);
        auto remoteMemInfo = TileXR::UDMAGetRemoteMemInfo(udmaInfo, peer);
        bool rangeValid = TileXR::UDMARegisteredRangeValid(registry, peer, remoteOffset, bytes);
        uint32_t wqeBefore = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_RANGE_VALID_BASE + peer] = rangeValid ? 1 : 0;
            debug[TILEXR_UDMA_DEMO_DEBUG_WQE_BEFORE_BASE + peer] = static_cast<int32_t>(wqeBefore);
            debug[TILEXR_UDMA_DEMO_DEBUG_LOCAL_TOKEN_BASE + peer] = static_cast<int32_t>(wqCtx->localTokenId);
            debug[TILEXR_UDMA_DEMO_DEBUG_REMOTE_BASE_LOW_BASE + peer] =
                static_cast<int32_t>(reinterpret_cast<uint64_t>(registry->regions[peer].base) & 0xFFFFFFFFU);
            debug[TILEXR_UDMA_DEMO_DEBUG_MEM_ADDR_LOW_BASE + peer] =
                static_cast<int32_t>(remoteMemInfo->addr & 0xFFFFFFFFU);
            debug[TILEXR_UDMA_DEMO_DEBUG_TPN_BASE + peer] = static_cast<int32_t>(remoteMemInfo->tpn);
        }
        TileXR::UDMAPutNbi<int32_t>(args, peer, localSrc, remoteOffset, bytes);
        uint32_t wqeAfter = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wqCtx->wqeCntAddr), 0);
        if (debug != nullptr && peer < 16) {
            debug[TILEXR_UDMA_DEMO_DEBUG_WQE_AFTER_BASE + peer] = static_cast<int32_t>(wqeAfter);
        }
        uint32_t status = TileXR::UDMAQuietStatus(args, peer);
        if (debug != nullptr) {
            debug[TILEXR_UDMA_DEMO_DEBUG_UDMA_STATUS_BASE + peer] = static_cast<int32_t>(status);
        }
    }
}
```

### 3.2 host 侧 launch(blockDim = rankSize)

`tests/udma/demo/tilexr_udma_demo.cpp:807` 与 `:850` 两处调用,均从 `1` 改为 `static_cast<uint32_t>(rankSize)`:

```cpp
// chunked strict 路径(:807)
launch_tilexr_udma_all_to_all(
    static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
    reinterpret_cast<GM_ADDR>(debug), chunkElements, static_cast<uint64_t>(outputOffset),
    0, chunkElements);

// 单 pass 路径(:850)
launch_tilexr_udma_all_to_all(
    static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
    reinterpret_cast<GM_ADDR>(debug), elementsPerRank, static_cast<uint64_t>(outputOffset), 0,
    elementsPerRank);
```

### 3.3 launch wrapper(透传 blockDim)

`tests/udma/demo/tilexr_udma_demo_kernel.cpp:731`:

```cpp
void launch_tilexr_udma_all_to_all(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements)
{
    tilexr_udma_all_to_all_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, input, output, debug, elementsPerPeer, outputByteOffset, inputElementOffset, chunkElements);
}
```

### 3.4 UDMA PUT 底层路径(`src/include/tilexr_udma.h`)

`UDMAPutNbi` → `UDMAWrite` → `UDMAPostSend` 核心逻辑:

```cpp
template <typename T>
__aicore__ inline void UDMAPutNbi(
    const __gm__ CommArgs* args, int targetRank, const __gm__ T* localSrc, uint64_t byteOffset, uint32_t byteCount)
{
    if (!UDMARegistryEnabled(args)) return;
    auto registry = GetUDMARegistry(args);
    if (!UDMARegisteredRangeValid(registry, targetRank, byteOffset, byteCount)) return;
    auto remoteAddr = UDMARegisteredRemoteAddr(registry, targetRank, byteOffset);
    UDMAWrite(args, remoteAddr, reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(localSrc)),
              targetRank, 0, byteCount);
}
```

`UDMAPostSend`:填 WQE(`UDMASqeCtx` 远端 + `UDMASgeCtx` 本地)→ 刷 cacheline → 写 doorbell → 递增 wqeCnt。`UDMAQuiet`:`UDMAPollCQ` 轮询 CQ tail 直到 CQE 产生。

---

## 4. 验证流程

### 4.1 环境

- 服务器:`root@141.62.19.144`(8 × Ascend950DT,CANN 25.1.rc1.b142)
- 仓库路径:`/home/tileXR-new/`
- 编译:`bash tests/udma/build.sh`(bisheng 交叉编译,产出 `install/bin/tilexr_udma_demo`)
- 强制 UDMA 路径:`TILEXR_DEMO_ALLTOALL_USE_UDMA=1`(默认 0 走 IPC data-as-flag fallback)

### 4.2 功能验证

```bash
# 2 rank, 1024 elements
TILEXR_DEMO_ALLTOALL_USE_UDMA=1 bash tests/udma/demo/run_tilexr_udma_demo.sh 2 2 1024

# 4 rank, 1024 elements
TILEXR_DEMO_ALLTOALL_USE_UDMA=1 bash tests/udma/demo/run_tilexr_udma_demo.sh 2 4 1024 4 0
```

结果:全部 rank `TileXR UDMA demo success`,peer debug `status=0`(CQ 完成)、`wqe=0->1`(计数正确),输出数据正确。

### 4.3 msprof 性能采集(4 rank, 8M 数据)

数据量 8M = 8*1024*1024 字节 = 2,097,152 个 int32。启动脚本 `run_fused_prof.sh`:

```bash
#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
TILEXR_ROOT=/home/tileXR-new
UDMA_DIR=${TILEXR_ROOT}/tests/udma
export LD_LIBRARY_PATH=${UDMA_DIR}/install/lib:${UDMA_DIR}/install/lib64:${TILEXR_ROOT}/install/lib:${TILEXR_ROOT}/install/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH}
export TILEXR_DEMO_ALLTOALL_USE_UDMA=1
export TILEXR_DEMO_ALLTOALL_REPEAT=1

RANK_SIZE=4
ELEM=2097152
BIN=${UDMA_DIR}/install/bin/tilexr_udma_demo

pids=()
for rank in $(seq 0 $((RANK_SIZE-1))); do
  RANK=${rank} RANK_SIZE=${RANK_SIZE} "${BIN}" "${RANK_SIZE}" "${rank}" 2 "${ELEM}" "${RANK_SIZE}" 0 \
    > /tmp/a2a_rank${rank}.log 2>&1 &
  pids+=("$!")
done
ret=0
for idx in "${!pids[@]}"; do
  wait "${pids[$idx]}" && echo "rank ${idx} ok" || { echo "rank ${idx} FAIL $?"; ret=1; }
done
echo "=== rank tails ==="
for rank in $(seq 0 $((RANK_SIZE-1))); do echo "--- rank ${rank} ---"; tail -6 /tmp/a2a_rank${rank}.log 2>/dev/null; done
exit ${ret}
```

msprof 采集命令:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.T560/aarch64-linux/lib64:/usr/local/Ascend/cann-9.1.T560/tools/profiler/lib64:$LD_LIBRARY_PATH
PROF_DIR=/home/tileXR-new/tests/udma/prof_out
rm -rf $PROF_DIR; mkdir -p $PROF_DIR
PATH=/usr/local/Ascend/cann-9.1.T560/tools/profiler/bin:$PATH \
msprof --output="$PROF_DIR" \
  --application=/home/tileXR-new/tests/udma/run_fused_prof.sh \
  --task-time=l0 \
  --ai-core=on \
  --aic-mode=sample-based \
  --aic-freq=100 \
  --aic-metrics=PipeUtilization \
  --aicpu=on \
  --runtime-api=on
```

关键点:
- `--application` 指向启动脚本,msprof 跟踪其 fork 的 4 个 rank 子进程,4 个 device 全部采到。
- `--aic-mode=sample-based` + `--aic-freq=100`:采样模式,对自定义 kernel(非 GE 模型)友好。
- `TILEXR_DEMO_ALLTOALL_USE_UDMA=1` 在脚本内 export,确保走 UDMA 路径而非 IPC fallback。

---

## 5. 性能结果

### 5.1 执行结果

4 个 rank 全部 `TileXR UDMA demo success`,数据正确,无卡死。

### 5.2 Kernel 任务时长(op_summary)

| Device | `tilexr_udma_all_to_all_kernel` 时长 (μs) |
|--------|------------------------------------------|
| dev0 | 192.397 |
| dev1 | 193.275 |
| dev2 | 185.162 |
| dev3 | 184.794 |

平均约 **189 μs**(8M 数据 / 4 rank)。

### 5.3 AI Vector Core 利用率

每个 device 实际活跃 4 个 vector core,与 `blockDim=4` 吻合:

| Core | 角色 | scalar | mte2 | mte3 |
|------|------|--------|------|------|
| Core36 | UDMA PUT | ~0.95 | 0 | ~0.003 |
| Core54 | self-copy | ~0.88-0.95 | ~0.32 | ~0.002 |
| Core72 | UDMA PUT | ~0.95 | 0 | ~0.003 |
| Core90 | UDMA PUT | ~0.95 | 0 | ~0.003 |

- 其余 60+ core 全 0
- Average:scalar 0.058,mte2 0.009,mte3 0.005

### 5.4 结论与瓶颈

1. **多核分片生效**:每 device 用 4 个 vector core,与 `blockDim=4` 完全吻合,`GetBlockIdx()` 分片正确,每个 core 正常退出(无死锁)。
2. **利用率低是设计预期**:UDMA PUT 把数据搬运下放给硬件 DMA 引擎(SQ/CQ),core 只做标量寄存器操作(投 WQE + 轮询 CQ),故 AI Core 利用率不是衡量此 kernel 的好指标。
3. **瓶颈在 UDMA 静默同步**:每个 PUT 分支 `PutNbi` 后立即 `UDMAQuietStatus` 轮询 CQ,是"发一个等一个"串行模式。4 个 peer 已多核并行,单 core 内的 quiet 轮询主导 ~189μs。
4. **self-copy(Core54)的 mte2=0.32**:DataCopyPad 经 UB 中转 8M,mte2 占比合理;但仍是 GM→UB→GM 两跳,而 UDMA PUT 是 GM→GM 一跳。后续可把 self-copy 也改成 UDMA 自投递(向自己 rank 注册区 PUT)统一路径。

### 5.5 采集产物

性能数据落在 `/home/tileXR-new/tests/udma/prof_out/` 下 4 个 `PROF_*/` 目录,每个含 `mindstudio_profiler_output/` 下的:
- `op_summary_*.csv` — kernel 任务时长
- `ai_core_utilization_*.csv` / `ai_vector_core_utilization_*.csv` — 每 core 利用率
- `task_time_*.csv` — task 时序
- `api_statistic_*.csv` — runtime API 统计

---

## 6. 关于"卡住"的澄清

改造过程中曾观察到运行卡死,排查结论:

- 默认 `TILEXR_DEMO_ALLTOALL_USE_UDMA=0` 时,demo 走 **IPC data-as-flag fallback 路径**(非本 kernel),卡死发生在该 fallback 的 TCP barrier / flag 轮询,与多核 UDMA 改动**无关**。
- 强制 `TILEXR_DEMO_ALLTOALL_USE_UDMA=1` 后,实际执行改后的多核 kernel,在 2/4 卡、1K~8M 数据量下全部稳定通过,无卡死。

---

## 7. 文件清单

| 文件 | 改动 |
|------|------|
| `tests/udma/demo/tilexr_udma_demo_kernel.cpp` | kernel 改多核分片 + self-copy 改 DataCopyPad |
| `tests/udma/demo/tilexr_udma_demo.cpp` | 两处 launch blockDim: 1 → rankSize |
| `tests/udma/demo/run_fused_prof.sh` | msprof 采集用启动脚本(新增) |