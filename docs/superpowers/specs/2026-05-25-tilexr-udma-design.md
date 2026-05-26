# TileXR UDMA 能力扩展设计

**日期**: 2026-05-25  
**目标**: 为 TileXR 添加 UDMA（URMA 直驱）通信能力，供新算子核（如 DeepEP dispatch/combine）在算子侧直接使用，不改动现有集合通信路径和内存初始化。

---

## 背景

TileXR 当前通过 `CommArgs.peerMems[]` 提供 IPC 共享内存，集合算子（AllReduce、AllGather 等）依赖 HCCS/MTE 引擎在节点内通信。跨节点场景需要 URMA/RDMA 能力，目前 `ExtraFlag::RDMA` 位已定义但未在传输层实际分发。

shmem（`3rdparty/shmem`，fork 自 `https://github.com/LingquLab/shmem`，branch `tilexr-udma-integration`）提供了完整的 UDMA 设备侧 API（`aclshmemx_udma_put_nbi` 等），以及 QP 初始化基础设施。

---

## 设计目标

- 在 TileXR 的 `CommArgs` 中携带 UDMA QP 上下文指针
- TileXR init 内部透明地完成 shmem UDMA 初始化，调用方无需感知 shmem
- 提供 `include/tilexr_udma.h`：设备侧薄封装，kernel 开发者只 include TileXR 头文件
- UDMA 不可用时（单节点 HCCS 环境、shmem init 失败）静默降级，不影响现有路径
- **不改动** TileXR 现有内存初始化（`peerMems[]`、IPC setup）

---

## 变更范围

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/include/comm_args.h` | 修改 | 加 `ExtraFlag::UDMA` 位 + `udmaInfoPtr` 字段 |
| `src/comm/tilexr_comm.h` | 修改 | 加 `InitUDMA()` 私有方法声明、`udmaInfoDev_` 成员 |
| `src/comm/tilexr_comm.cpp` | 修改 | `Init()` / `InitThread()` 追加 UDMA init 阶段；`LcclCommDestroy` 追加 finalize |
| `include/tilexr_udma.h` | 新增 | AICore kernel 侧 UDMA 薄封装 |
| `.gitmodules` | 修改 | 追加 `3rdparty/shmem` submodule 条目 |
| `CMakeLists.txt` | 修改 | 引入 shmem host 库和 include 路径 |
| CCE 编译脚本 (`build.sh` 等) | 修改 | 追加 `-I3rdparty/shmem/include` |

---

## 第一节：CommArgs 与 ExtraFlag

**文件**: `src/include/comm_args.h`

```cpp
struct ExtraFlag {
    // 现有位不变...
    static constexpr uint32_t UDMA = 1 << 10;  // bit 10（原为空闲）
};

struct CommArgs {
    // 现有字段不变...
    uint64_t fftsVal = 0;
    GM_ADDR  udmaInfoPtr = nullptr;  // device-side ACLSHMEMAIVUDMAInfo*
                                     // nullptr 表示 UDMA 不可用
};
```

`udmaInfoPtr` 指向设备 HBM 上一份 `ACLSHMEMAIVUDMAInfo` 的拷贝（48 字节，6 个 `uint64_t`）。该结构体中的 SQ/RQ/CQ 数组地址在 shmem init 时已分配在设备内存中，直接 `rtMemcpy` 即可。

---

## 第二节：Init 流程变更

### 进程模式（`TileXRComm::Init()`）

在现有 `InitCommMem()` 之后、`SyncCommArgs()` 之前插入 `InitUDMA()`：

```
GetDev() → EnablePeerAccess() → InitCommMem() → InitUDMA() → SyncCommArgs()
```

**`InitUDMA()` 逻辑**：

```
1. rank 0: aclshmemx_get_uniqueid(&shmem_uid)
   其余 rank: shmem_uid 置零

2. sockExchange_->AllGather(&shmem_uid, 1, all_uids)
   所有 rank 取 all_uids[0]（rank 0 的 UID）

3. aclshmemx_set_attr_uniqueid_args(
       rank_, rankSize_,
       TILEXR_SHMEM_MIN_MEM,          // = 1 * 1024 * 1024（1 MB），仅供 shmem 内部 sync
       &shmem_uid, &attr)
   attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA

4. ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr)
   if (ret != 0) { LOG_WARNING(...); return TILEXR_SUCCESS; }  // 静默降级

5. // shmem 在 init 后将 ACLSHMEMAIVUDMAInfo* 存入全局 state->qp_info（uint64_t）
   // 通过 aclshmemi_get_device_state() 获取 state 指针（shmem 内部 API，需确认可用性）
   udmaInfoHost = reinterpret_cast<ACLSHMEMAIVUDMAInfo*>(
       aclshmemi_get_device_state()->qp_info)
   rtMalloc(&udmaInfoDev_, sizeof(ACLSHMEMAIVUDMAInfo), RT_MEMORY_HBM)
   rtMemcpy(udmaInfoDev_, udmaInfoHost, sizeof(ACLSHMEMAIVUDMAInfo),
            RT_MEMCPY_HOST_TO_DEVICE)
   // 实现时需验证 aclshmemi_get_device_state() 的实际导出名称

6. commArgs_.udmaInfoPtr = reinterpret_cast<GM_ADDR>(udmaInfoDev_)
   commArgs_.extraFlag  |= ExtraFlag::UDMA
```

### 线程模式（`TileXRComm::InitThread()`）

与进程模式对称：通过 `g_localPeerMemMap[uid]` 全局 map 协调 shmem UID——rank 0 线程写入 UID，其他线程轮询直到可读，之后各自执行步骤 3–6。

### Destroy

`LcclCommDestroy` 中，若 `commArgs_.udmaInfoPtr != nullptr`：
```cpp
aclshmem_finalize();
rtFree(udmaInfoDev_);
```

---

## 第三节：`include/tilexr_udma.h`

AICore kernel 侧封装，仅依赖 UDMA 引擎头文件，不引入 shmem 内存接口。

```cpp
// include/tilexr_udma.h
// AICore kernel 侧使用；host 侧不 include 此文件。
// 需要 TileXRCommInit* 已完成且 UDMAEnabled() 返回 true。
#pragma once
#include "comm_args.h"
#include "shmem/include/device/gm2gm/engine/shmem_device_udma.h"

namespace TileXR {

// 可用性检查
__aicore__ inline bool UDMAEnabled(__gm__ const CommArgs *args) {
    return (args->extraFlag & ExtraFlag::UDMA) != 0;
}

// 非阻塞 Put：本地 src → remote pe 的 dst（对称内存地址）
// ub_stage: 调用方提供的 UB 暂存区，>= 64B
template <typename T>
__aicore__ inline void UDMAPutNbi(
    __gm__ T *dst, __gm__ T *src,
    __ubuf__ T *ub_stage, uint32_t elem_size, int pe)
{
    aclshmemx_udma_put_nbi(dst, src, ub_stage, elem_size, pe);
}

// 非阻塞 Get：remote pe 的 src → 本地 dst
template <typename T>
__aicore__ inline void UDMAGetNbi(
    __gm__ T *dst, __gm__ T *src,
    __ubuf__ T *ub_stage, uint32_t elem_size, int pe)
{
    aclshmemx_udma_get_nbi(dst, src, ub_stage, elem_size, pe);
}

// Put + 原子信号通知（dispatch 关键路径）
template <typename T>
__aicore__ inline void UDMAPutSignalNbi(
    __gm__ T *dst, __gm__ T *src, uint32_t elem_size,
    __gm__ uint64_t *remote_sig, uint64_t signal, int pe)
{
    aclshmemx_udma_put_signal_nbi(dst, src, elem_size, remote_sig, signal, pe);
}

// 等待到 pe 的所有未完成 UDMA 操作完成
__aicore__ inline void UDMAQuiet(int pe) {
    aclshmemx_udma_quiet(pe);
}

// 原子加
template <typename T>
__aicore__ inline void UDMAAtomicAdd(__gm__ T *dst, T val, int pe) {
    aclshmemx_udma_atomic_add(dst, val, pe);
}

// 原子 fetch-and-add
template <typename T>
__aicore__ inline T UDMAAtomicFetchAdd(__gm__ T *dst, T val, int pe) {
    return aclshmemx_udma_atomic_fetch_add(dst, val, pe);
}

// 原子 compare-and-swap
template <typename T>
__aicore__ inline T UDMAAtomicCompareSwap(__gm__ T *dst, T cond, T val, int pe) {
    return aclshmemx_udma_atomic_compare_swap(dst, cond, val, pe);
}

} // namespace TileXR
```

**调用约定**：
- `ub_stage` 最小 64 字节，通过 `TBuf<VECCALC>` 或 `TBuf<VECOUT>` 分配
- `UDMAQuiet(pe)` 仅等待到单个 pe，全员 quiet 循环调用或配合 `SyncCollectives::WaitAllRankOuterFlag`
- `dst`/`src` 必须是通过 `aclshmem_malloc` 分配的对称内存地址；`peerMems[]` 中的 IPC 地址不适用

---

## 第四节：构建变更

### `.gitmodules`

```ini
[submodule "3rdparty/shmem"]
    path   = 3rdparty/shmem
    url    = https://github.com/LingquLab/shmem.git
    branch = tilexr-udma-integration
```

根目录现有 `shmem/` 目录在 submodule 就位后删除。

### CMake（host 侧）

```cmake
set(SHMEM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/shmem")

find_library(SHMEM_HOST_LIB aclshmem
    HINTS "${SHMEM_ROOT}/lib" "${SHMEM_ROOT}/build/lib"
    REQUIRED)

target_include_directories(tile-comm PRIVATE "${SHMEM_ROOT}/include")
target_link_libraries(tile-comm PRIVATE ${SHMEM_HOST_LIB})
```

### CCE 编译（device 侧）

shmem 设备侧为纯头文件（`__aicore__` 内联），无需链接设备库，只需追加 include 路径：

```bash
# build.sh 或 CMakeLists 的 cce 编译选项
-I${TILEXR_HOME}/3rdparty/shmem/include
```

---

## 降级行为

| 场景 | 行为 |
|------|------|
| shmem init 失败（UDMA 硬件不可用） | `InitUDMA()` 记录 WARNING，返回 `TILEXR_SUCCESS`；`udmaInfoPtr` 保持 `nullptr`，不设 `ExtraFlag::UDMA` |
| kernel 侧 `UDMAEnabled()` 返回 false | 调用方自行决定 fallback（如走 MTE 路径或报错） |
| 单节点 HCCS 环境 | 同上，现有集合通信路径完全不受影响 |
