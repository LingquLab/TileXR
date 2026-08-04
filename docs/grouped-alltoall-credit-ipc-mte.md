# Grouped AllToAll Credit IPC 与 MTE Set/Wait 实现

本文整理 TileXR grouped all-to-all ingress credit 在当前版本 `b51dc2e` 中的完整实现，覆盖通信域初始化、IPC 内存布局、Host 到 Device 参数传递，以及 Device 侧通过 MTE 发布和轮询 credit 的流程。其它项目可以复用该设计，但必须先确认目标硬件支持对已映射 IPC GM 地址执行 MTE 访问。

## 1. 功能语义

Credit 用于限制同一张目标卡在相邻 group 中承受的并发 ingress。基本时序是：

```text
发送端 S                              接收端 R
group g: UDMA put + signal  --------> 等待 payload signal 成功
                                       MTE3 写 credit(g+1)
                                       到 S 拥有的 IPC credit buffer
group g+1: MTE2 轮询本地 credit <---- credit 对 S 可见
group g+1: UDMA put + signal  ------->
```

当前发布点位于接收端确认 UDMA payload signal 到达之后、receive-copy 之前。因此当前 credit 的准确含义是：

> 上一 group 的远端数据已经到达，接收端允许下一 group ingress。

它不表示上一批数据已被 receive-copy 或应用层消费。如果复用同一块 payload 存储要求“消费完成后才能覆盖”，必须把 credit 发布点后移到 copyout/消费完成之后。

Credit 通道只传控制 token，使用 IPC GM 加 MTE，不使用 UDMA QP。

## 2. 代码位置

| 模块 | 文件 | 职责 |
|---|---|---|
| 公共 ABI 和常量 | `D:/workspace/TileXR/.kilo/worktrees/serene-beak/src/include/comm_args.h` | IPC 大小、stride、`CommArgs::creditMems` |
| 通信域 Host 初始化 | `D:/workspace/TileXR/.kilo/worktrees/serene-beak/src/comm/tilexr_comm.cpp` | 申请、授权、交换、映射和释放 IPC memory |
| 布局与 Host 校验 | `D:/workspace/TileXR/.kilo/worktrees/serene-beak/tests/udma/demo/tilexr_udma_alltoall_group_layout.h` | ping-pong slot、offset、参数约束 |
| Demo Host 流程 | `D:/workspace/TileXR/.kilo/worktrees/serene-beak/tests/udma/demo/tilexr_udma_demo.cpp` | 环境变量、映射完整性和 single-pass 校验 |
| Kernel launch ABI | `D:/workspace/TileXR/.kilo/worktrees/serene-beak/tests/udma/demo/tilexr_udma_alltoall_group_launcher.cpp` | credit offset 和 ingressWindow 下发 |
| Device set/wait | `D:/workspace/TileXR/.kilo/worktrees/serene-beak/tests/udma/demo/tilexr_udma_alltoall_group_kernel.cpp` | MTE3 publish、MTE2 poll 和调度 |

## 3. IPC 内存参数与布局

公共常量定义如下：

```cpp
constexpr int TILEXR_MAX_RANK_SIZE = 1024;
constexpr int64_t CREDIT_IPC_STRIDE = 512;
constexpr int64_t CREDIT_IPC_SLOT_BYTES =
    TILEXR_MAX_RANK_SIZE * CREDIT_IPC_STRIDE;  // 512 KiB
constexpr int64_t CREDIT_IPC_BYTES =
    2 * CREDIT_IPC_SLOT_BYTES;                // 1 MiB
```

每个 rank 独立申请一块固定 1 MiB 的 credit IPC memory。它按 invocation 奇偶分成两个 slot：

```text
rank R 拥有的 credit buffer，共 1 MiB

slot 0: [0,       512 KiB)
slot 1: [512 KiB,   1 MiB)

slot 内：
rank index 0: [0 * 512, 1 * 512)
rank index 1: [1 * 512, 2 * 512)
...
rank index 1023
```

每个 credit 占位 512 B。token 只使用第一个 `uint64_t`，但 set 和 wait 都搬运完整 512 B，即 64 个 `uint64_t`。固定 stride 同时满足数据搬运对齐，并隔离不同 peer 的 cache line/传输单元，避免伪共享。

内存按最大 1024 rank 固定布局，而不是按当前 `rankSize` 缩小。因此不同进程和 Host/Kernel 无需协商动态 slot 大小。

## 4. 通信域 Host 初始化

### 4.1 开关和初始化顺序

通信域通过以下变量启用专用 credit IPC memory：

```bash
export TILEXR_ENABLE_CREDIT_IPC=1
```

普通多进程通信域中的相关初始化顺序为：

```text
GetDev
InitCommon
InitCommMem（TILEXR_ENABLE_IPC 开启时）
InitCreditCommMem（TILEXR_ENABLE_CREDIT_IPC=1 时）
InitUDMA
InitSDMA
SyncCommArgs
```

Credit IPC 必须在 `SyncCommArgs()` 前完成，因为映射后的 Device 地址要写入 `CommArgs`。

### 4.2 收集 IPC 授权信息

`InitCreditCommMem()` 首先收集所有进程的 PID 和 SDID：

```text
rtDeviceGetBareTgid(localPid)
socketExchange_->AllGather(localPid, allPids)

rtGetDeviceInfo(..., infoTypeSdid=26, localSdid)
socketExchange_->AllGather(localSdid, allSdids)
```

这里使用 bare TGID，目的是在容器环境中取得 Runtime IPC 授权需要的宿主侧进程标识。

IPC 授权模式由下列变量控制：

```bash
export TILEXR_IPC_PID_MODE=pid
# 或
export TILEXR_IPC_PID_MODE=sdid
```

当前代码对 `CHIP_910_9391 <= chip < CHIP_950` 默认选择 SDID，其它情况默认 PID；显式环境变量优先。授权由 `SetIpcPidSdid()` 对本 rank 的 IPC name 添加所有远端访问者。

对应 Runtime API 为：

```cpp
// PID 模式
rtSetIpcMemPid(ipcName, &peerPid, HCCL_IPC_PID_ARRAY_SIZE);

// SDID 模式；失败时当前实现回退到 PID 模式
rtSetIpcMemorySuperPodPid(
    ipcName, peerSdid, &peerPid, HCCL_IPC_PID_ARRAY_SIZE);
```

### 4.3 申请、命名和映射

每个 rank 的 Host 流程如下：

```cpp
// 1. 本 rank 申请 owner buffer。
aclrtMalloc(&creditIpcMem_[rank], CREDIT_IPC_BYTES,
            ACL_MEM_MALLOC_HUGE_FIRST);

// 310P3 使用 ACL_MEM_MALLOC_HUGE_FIRST_P2P。

// 2. 只初始化一次，后续依靠单调 token 复用。
aclrtMemset(creditIpcMem_[rank], CREDIT_IPC_BYTES, 0, CREDIT_IPC_BYTES);

// 3. 导出 IPC name，并授权所有 peer PID/SDID。
rtIpcSetMemoryName(creditIpcMem_[rank], CREDIT_IPC_BYTES,
                   localName, IPC_NAME_SIZE);
SetIpcPidSdid(localName, allPids, allSdids);

// 4. AllGather 每个 rank 的 IPC name。
GetName(localName, allNames);

// 5. 打开其它 rank 的 owner buffer。
rtIpcOpenMemory(&creditIpcMem_[peer], allNames[peer]);
```

完成后，在任意 rank `R` 的进程中：

```text
creditIpcMem_[R]    = R 自己申请的本地 owner 地址
creditIpcMem_[peer] = peer buffer 映射到 R 地址空间后的 Device GM 地址
```

`OpenCreditIpcMem()` 跳过 self，也沿用 `SkipUnusedChannel910B2C()` 对不使用的 910B2C 链路做过滤。使用 credit 的 Host 代码随后要求当前通信范围内每个 `creditMems[peer]` 都非空，因此迁移时必须让“跳过映射”和“实际参与 peer”保持一致。

### 4.4 Host 到 Device 参数传递

`CommArgs` 提供固定长度的映射表：

```cpp
struct CommArgs {
    // ...
    GM_ADDR creditMems[TILEXR_MAX_RANK_SIZE] = {};
};
```

`SyncCommArgs()` 将 Host 通信域中的地址逐项复制进去：

```cpp
for (int i = 0; i < rankSize_; ++i) {
    commArgs_.creditMems[i] = creditIpcMem_[i];
}

aclrtMalloc(&commArgsPtr_, sizeof(commArgs_), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(commArgsPtr_, sizeof(commArgs_),
            &commArgs_, sizeof(commArgs_),
            ACL_MEMCPY_HOST_TO_DEVICE);
```

Kernel 获得 `commArgsPtr_` 后，可以直接把 `args->creditMems[peer]` 作为 `__gm__` 地址用于 MTE。这里传递的是 Device 可访问的 owner/IPC mapping 地址，不是 Host 虚拟地址或 IPC name。

Kernel launch ABI 另外传递：

```cpp
uint64_t creditOffset0;
uint64_t creditOffset1;
uint32_t ingressWindow;
```

当前 credit kernel 参数结构为 152 B。Host 和 Kernel 的参数结构、字段顺序及对齐必须同步修改，不能只改一侧。

## 5. Ping-pong 和 token

当前 invocation 使用两个 slot 交替复用：

```cpp
slot = invocationId & 1U;
creditOffset[0] = 0;
creditOffset[1] = CREDIT_IPC_SLOT_BYTES;
```

Token 格式为：

```cpp
uint64_t token =
    (uint64_t(invocationId + 1) << 32) |
    (uint64_t(slot) << 31) |
    (uint64_t(group) << 16) |
    uint64_t(pass + 1);
```

Credit 当前只支持 single pass，因此 credit token 的 `pass` 固定为 0。消费者判断：

```cpp
observed >= expectedToken
```

Buffer 只在通信域初始化时清零一次。之后 token 随 invocation/group 单调推进，旧值无需再次清零。迁移时必须确保 invocation 不会在 buffer 生命周期内回绕，否则 `>=` 比较将不再安全。

## 6. Device 侧 MTE set

### 6.1 地址所有权

接收 rank `R` 完成 group `g` 的 payload signal wait 后，计算下一 group 同 lane 的发送者 `S`，然后写入：

```cpp
args->creditMems[S]
    + creditOffset[slot]
    + R * CREDIT_IPC_STRIDE
```

也就是：

```text
buffer owner = 下一 group 的发送者 S
slot index   = 当前 invocation 的 ping-pong slot
entry index  = 授权者/目标接收 rank R
```

这个方向非常关键。R 不是写自己的 buffer，而是通过 IPC mapping 写 S 拥有的 buffer；这样 S 在发送下一 group 前只需轮询自己的本地 owner buffer。

### 6.2 MTE3 发布代码

当前实现先在 UB 中构造 512 B credit block，再用 MTE3 写远端 IPC GM：

```cpp
auto creditLocal = relayLocal.ReinterpretCast<uint64_t>();
creditLocal.SetValue(0, creditToken);

SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

GlobalTensor<uint64_t> remoteCreditGlobal;
remoteCreditGlobal.SetGlobalBuffer(remoteCredit, 64);
DataCopy(remoteCreditGlobal, creditLocal, 64); // 64 * 8 B = 512 B

SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
```

两个同步方向分别解决：

- `S_MTE3`：保证 MTE3 读取 UB 前，scalar 对 `creditLocal` 的 token 写入已经可见。
- `MTE3_S`：保证 scalar 路径继续执行前，本次 UB→GM/IPC 写已经完成。

不要用 `GlobalTensor::SetValue()` 代替生产路径中的 MTE copy。当前实现专门经 UB 和 MTE3 写入远端 IPC memory，以获得明确的数据搬运与同步语义。

### 6.3 发布者唯一性

Receive-copy 可能有 32 个 worker，但 credit 只允许指定 owner 发布：

```text
worker < 16
```

这保证每个 lane/entry 只有一个 core 写 token，避免两个 core 并发写同一 512 B entry。当前代码还要求在最后一个 pass 发布；由于 ingress credit 只允许 single pass，实际就是 pass 0。

## 7. Device 侧 MTE wait

发送 rank `S` 在发送 group `g > 0` 到目标 `R` 前，轮询：

```cpp
args->creditMems[S]
    + creditOffset[slot]
    + R * CREDIT_IPC_STRIDE
```

从 S 的视角，`args->creditMems[S]` 是自己的本地 owner buffer。entry `R` 表示目标接收端 R 已经授予本次发送 credit。

每次 load 都将完整 512 B 从 GM 搬到 UB：

```cpp
GlobalTensor<uint64_t> creditGlobal;
creditGlobal.SetGlobalBuffer(credit, 64);
auto creditLocal = relayLocal.ReinterpretCast<uint64_t>();

DataCopy(creditLocal, creditGlobal, 64); // GM -> UB, 512 B
SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

uint64_t observed = creditLocal.GetValue(0);
```

`MTE2_S` 保证 scalar 读取 UB token 前，GM→UB 的 MTE2 搬运已经完成。

轮询逻辑为：

```cpp
const uint64_t begin = GetSystemCycle();
do {
    observed = LoadCreditMte(creditAddress, relayLocal);
    if (GetSystemCycle() - begin >= timeoutCycles) {
        return false;
    }
} while (observed < expectedToken);
```

当前 timeout 是 `10000000000` cycles。超时会记录 credit-wait stage、group、peer、expected 和 observed token，并提前退出该 kernel core。

Group 0 不等待，因为还没有前一 group 可以授予 credit。对当前 route 数据量为 0 的 peer，也跳过 wait 和对应 send。

## 8. Host 侧运行约束

Demo 中还需要启用 ingress window：

```bash
export TILEXR_ENABLE_CREDIT_IPC=1
export TILEXR_DEMO_ALLTOALL_GROUP_INGRESS_WINDOW=1
```

当前实现限制如下：

```text
ingressWindow = 0 或 1
credit 模式要求 groupWidth = 16
credit 模式要求 CHUNK_ELEMENTS = ELEMENTS，即 single pass
creditMems[0..rankSize-1] 必须全部非空
```

Host 用静态断言保护通信域和 demo 的布局常量：

```cpp
static_assert(kAllToAllGroupCreditStride == CREDIT_IPC_STRIDE);
static_assert(kAllToAllGroupCreditSlotBytes == CREDIT_IPC_SLOT_BYTES);
```

其它项目也应在 ABI 两侧增加同类断言，避免 stride 或最大 rank 数不一致造成静默越界。

## 9. 释放顺序

通信域销毁时必须区分远端 mapping 和本地 owner allocation：

```text
1. kernel/stream 已停止访问 CommArgs 和 credit memory
2. rtIpcCloseMemory() 关闭所有 peer 的远端 mapping
3. aclrtFree() 释放本 rank 的 owner credit allocation
4. aclrtFree() 释放 Device CommArgs
```

当前代码由 `CloseCreditIpcMem()` 关闭 `creditIpcMem_[peer]`，再通过 `FreePeerMem(creditIpcMem_[rank_])` 释放本地 allocation。不能对远端 mapping 调用 `aclrtFree()`，也不能在 peer 尚可能访问时提前释放 owner buffer。

## 10. 迁移到其它项目的最小步骤

1. 定义 Host/Device 共用的 `MAX_RANK_SIZE`、512 B stride、两个 slot 和总字节数。
2. 每 rank 申请一块独立的 P2P/IPC Device GM buffer，并清零一次。
3. 收集所有进程的 bare PID 和设备 SDID。
4. 导出本 rank IPC name/handle，并按目标芯片要求授权 PID 或 SDID。
5. AllGather IPC name/handle，在每个 rank 打开所有参与 peer 的映射。
6. 将 owner 地址和所有 IPC mapping 填入 Device 可见的 `creditMems[]`。
7. 把 `CommArgs` 复制到 Device，并在 kernel launch ABI 中传入 slot offsets 和 enable flag。
8. Producer 在 UB 写 token，经 `S_MTE3` 后用 MTE3 搬 512 B 到远端 IPC GM，再执行 `MTE3_S`。
9. Consumer 用 MTE2 从本地 owner GM 搬 512 B 到 UB，经 `MTE2_S` 后读取 token 并轮询。
10. 使用随 epoch/invocation 单调增加的 token，并用 ping-pong slot 隔离相邻 invocation。
11. 明确定义 credit 发布点是“数据到达”“copyout 完成”还是“应用消费完成”。
12. 销毁时先停止 kernel，再关闭远端 mapping，最后释放本地 owner buffer。

## 11. 常见错误与检查方法

| 问题 | 表现或风险 | 检查方法 |
|---|---|---|
| `creditMems[peer]` 为空 | Device 访问异常或 Host 拒绝启动 | 启动前逐 peer 校验映射 |
| owner/index 写反 | 永久 credit-wait 超时 | 确认 R 写 `creditMems[S] + R*stride`，S 读 `creditMems[S] + R*stride` |
| Host/Kernel stride 不一致 | 读到其它 peer token或越界 | 共享头文件并加 `static_assert` |
| 只搬 8 B 或不满足平台对齐 | DataCopy 行为不稳定或性能异常 | 保持 512 B 对齐和 512 B copy |
| 缺少 `S_MTE3` | MTE3 可能读到 UB 旧值 | scalar 写 UB 后建立 S→MTE3 依赖 |
| 缺少 `MTE3_S` | credit 未完成便复用 UB/继续调度 | publish 后建立 MTE3→S 依赖 |
| 缺少 `MTE2_S` | scalar 可能在 load 完成前读取 UB | load 后建立 MTE2→S 依赖 |
| 多个 core 发布同一 entry | 重复或竞争写 token | 为每个 lane/peer 指定唯一 publisher |
| token 不单调或 invocation 回绕 | 旧 token 被误判为新 credit | 评估 token 位宽和通信域最长生命周期 |
| 发布点过早 | payload 被覆盖而尚未消费 | 根据 buffer 复用语义移动 publish 点 |
| IPC 授权模式不匹配 | `rtIpcOpenMemory` 失败 | 核对 PID/SDID、容器 bare PID 和芯片规则 |
| 超出 MTE 可访问拓扑 | 远端 IPC 地址不能被 MTE 访问 | 先在目标超节点/P2P 范围做最小 set/wait 验证 |

调试超时时，至少记录：

```text
local rank
peer rank
buffer owner rank
slot / byte offset
group / pass / invocation
expected token
observed token
publisher core
waiter core
```

其中 expected/observed token 可以直接拆出 invocation、slot、group 和 pass，用于区分“对端没有发布”“读错 entry”以及“读到了上一轮 token”。

## 12. 设计边界

- 该 credit 方案依赖超节点或目标 P2P 拓扑内，MTE 能访问 `rtIpcOpenMemory` 返回的 GM mapping。它不是任意跨节点的通用控制通道。
- 512 B stride 是当前实现的 Host/Device ABI，不只是性能参数；修改时必须同步布局、申请大小、DataCopy 长度、地址计算和静态断言。
- 两个 ping-pong slot 解决相邻 invocation 干扰，但不能替代正确的 epoch/token 设计。
- Busy polling 会占用 AIV/MTE2 资源。若快慢卡严重，credit-wait 本身可能成为关键路径，需要结合调度顺序、窗口大小或硬件通知机制进一步优化。
- 当前 `ingressWindow=1` 是严格的相邻 group credit，不等同于可配置的多 credit 滑动窗口。扩展到 window > 1 时，需要重新设计 entry 状态或 token/ack 关系，不能只放宽 Host 参数校验。

## 13. 可直接提取的实际代码

本节不是伪代码，来自当前 `b51dc2e` 实现。迁移时可以直接复制，再替换目标项目的日志、错误码、通信域类名和 all-gather 接口。

### 13.1 依赖和唯一适配点

Host 代码依赖 CANN ACL/Runtime IPC API：

```cpp
#include <acl/acl_rt.h>
#include "runtime/dev.h"
#include "runtime/mem.h"

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
```

目标项目需要提供以下已有上下文：

```text
rank_ / rankSize_                 当前 rank 和通信域大小
devList_[rank_]                   当前 rank 对应的逻辑 device id
physicalInfo_.chipName            芯片型号，用于选择 PID/SDID 模式
socketExchange_->AllGather(...)   Host 侧全通信域 all-gather
SkipUnusedChannel910B2C(...)      可选；没有特殊拓扑时删除该判断
TILEXR_SUCCESS / TILEXR_ERROR_*   替换成目标项目错误码
TILEXR_LOG                        替换成目标项目日志
```

其中 `AllGather` 必须在所有 rank 上以完全相同的顺序调用。支持 SDID 的芯片依次执行 PID、SDID 和 IPC name 三次 collective；其它芯片执行 PID 和 IPC name 两次 collective。通信域内不能让部分 rank 单独跳过其中一次。

### 13.2 Host/Device 共用头文件

这部分必须放在 Host 和 Ascend C kernel 都包含的公共头文件中：

```cpp
#pragma once

#include <cstdint>

#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

namespace TileXR {

constexpr int TILEXR_MAX_RANK_SIZE = 1024;
constexpr int IPC_NAME_SIZE = 65;
constexpr int HCCL_IPC_PID_ARRAY_SIZE = 1;
constexpr int64_t CREDIT_IPC_STRIDE = 512;
constexpr int64_t CREDIT_IPC_SLOT_BYTES =
    TILEXR_MAX_RANK_SIZE * CREDIT_IPC_STRIDE;
constexpr int64_t CREDIT_IPC_BYTES = 2 * CREDIT_IPC_SLOT_BYTES;

struct CommArgs {
    int rank = 0;
    int localRank = -1;
    int rankSize = 0;
    int localRankSize = -1;
    // 其它通信参数放在这里。
    GM_ADDR creditMems[TILEXR_MAX_RANK_SIZE] = {};
};

} // namespace TileXR
```

如果目标项目已经有 `CommArgs`，只添加 `creditMems[]` 字段。Host 和 kernel 必须使用同一份结构体定义，不要维护两份相似 ABI。

### 13.3 通信域类成员

当前通信域需要的成员和方法声明如下：

```cpp
private:
    int GetPid(uint32_t *pids);
    int GetSidId(int64_t sdids[TileXR::TILEXR_MAX_RANK_SIZE], int rankSize);
    int GetName(
        std::string &name,
        char names[TileXR::TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE]) const;
    int SetIpcPidSdid(
        std::string &name, const uint32_t *pids,
        const int64_t *sdids) const;
    int InitCreditCommMem();
    int InitCreditIpcMem(const uint32_t *pids, const int64_t *sdids);
    int OpenCreditIpcMem(
        const char names[TileXR::TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE]);
    void CloseCreditIpcMem();

    GM_ADDR creditIpcMem_[TileXR::TILEXR_MAX_RANK_SIZE] = {};
    bool creditIpcMemInited_ = false;
```

`IPC_NAME_SIZE` 使用目标 CANN/Runtime 头文件中的 IPC name 容量定义；所有 rank 必须交换固定的 `IPC_NAME_SIZE` 字节，而不是 `strlen(name)` 字节。

### 13.4 PID、SDID 和 IPC name all-gather

以下是当前实际实现。`socketExchange_->AllGather(local, count, output)` 的 `count` 是元素数：

```cpp
int TileXRComm::GetPid(uint32_t *pids)
{
    if (rtDeviceGetBareTgid(&pids[rank_]) != RT_ERROR_NONE) {
        return TILEXR_ERROR_INTERNAL;
    }
    return socketExchange_->AllGather(&pids[rank_], 1, pids);
}

int TileXRComm::GetSidId(
    int64_t sdids[TileXR::TILEXR_MAX_RANK_SIZE], int rankSize)
{
    if (rank_ >= rankSize) {
        return TILEXR_ERROR_INTERNAL;
    }

    if (physicalInfo_.chipName >= ChipName::CHIP_910_9391 &&
        physicalInfo_.chipName < ChipName::RESERVED) {
        constexpr int kRtModuleTypeSystem = 0;
        constexpr int kInfoTypeSdid = 26;
        if (rtGetDeviceInfo(
                devList_[rank_], kRtModuleTypeSystem, kInfoTypeSdid,
                &sdids[rank_]) != RT_ERROR_NONE) {
            return TILEXR_ERROR_INTERNAL;
        }
        return socketExchange_->AllGather(&sdids[rank_], 1, sdids);
    }
    return TILEXR_SUCCESS;
}

int TileXRComm::GetName(
    std::string &name,
    char names[TileXR::TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE]) const
{
    return socketExchange_->AllGather<char>(
        name.c_str(), IPC_NAME_SIZE, names[0]);
}
```

如果目标项目的 all-gather 以字节数为单位，PID/SDID 的发送长度应分别改为 `sizeof(uint32_t)` 和 `sizeof(int64_t)`；不要机械保留这里的 `1`。

### 13.5 IPC PID/SDID 授权

这是当前实际授权函数：

```cpp
int TileXRComm::SetIpcPidSdid(
    std::string &name, const uint32_t *pids,
    const int64_t *sdids) const
{
    const char *modeEnv = std::getenv("TILEXR_IPC_PID_MODE");
    const bool forcePid =
        modeEnv != nullptr && std::string(modeEnv) == "pid";
    const bool forceSdid =
        modeEnv != nullptr && std::string(modeEnv) == "sdid";
    const bool defaultSdid =
        physicalInfo_.chipName >= ChipName::CHIP_910_9391 &&
        physicalInfo_.chipName < ChipName::CHIP_950;
    const bool useSdid = forceSdid || (!forcePid && defaultSdid);

    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_) {
            continue;
        }

        int32_t pid = static_cast<int32_t>(pids[i]);
        if (!useSdid) {
            if (rtSetIpcMemPid(
                    name.c_str(), &pid,
                    HCCL_IPC_PID_ARRAY_SIZE) != RT_ERROR_NONE) {
                return TILEXR_ERROR_INTERNAL;
            }
            continue;
        }

        int ret = rtSetIpcMemorySuperPodPid(
            name.c_str(), sdids[i], &pid, HCCL_IPC_PID_ARRAY_SIZE);
        if (ret != RT_ERROR_NONE) {
            ret = rtSetIpcMemPid(
                name.c_str(), &pid, HCCL_IPC_PID_ARRAY_SIZE);
            if (ret != RT_ERROR_NONE) {
                return TILEXR_ERROR_INTERNAL;
            }
        }
    }
    return TILEXR_SUCCESS;
}
```

### 13.6 IPC memory 申请、交换和打开

以下函数构成通信域 credit 初始化主体：

```cpp
int TileXRComm::InitCreditCommMem()
{
    uint32_t pids[TileXR::TILEXR_MAX_RANK_SIZE] = {};
    int ret = GetPid(pids);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }

    int64_t sdids[TileXR::TILEXR_MAX_RANK_SIZE] = {};
    ret = GetSidId(sdids, rankSize_);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    return InitCreditIpcMem(pids, sdids);
}

int TileXRComm::InitCreditIpcMem(
    const uint32_t *pids, const int64_t *sdids)
{
    const auto policy =
        GetChipName() == ChipName::CHIP_310P3
            ? ACL_MEM_MALLOC_HUGE_FIRST_P2P
            : ACL_MEM_MALLOC_HUGE_FIRST;

    aclError aclRet = aclrtMalloc(
        reinterpret_cast<void **>(&creditIpcMem_[rank_]),
        TileXR::CREDIT_IPC_BYTES, policy);
    if (aclRet != ACL_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }

    aclRet = aclrtMemset(
        creditIpcMem_[rank_], TileXR::CREDIT_IPC_BYTES,
        0, TileXR::CREDIT_IPC_BYTES);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(creditIpcMem_[rank_]);
        creditIpcMem_[rank_] = nullptr;
        return TILEXR_ERROR_INTERNAL;
    }

    char nameBuffer[IPC_NAME_SIZE] = {};
    if (rtIpcSetMemoryName(
            creditIpcMem_[rank_], TileXR::CREDIT_IPC_BYTES,
            nameBuffer, IPC_NAME_SIZE) != RT_ERROR_NONE) {
        aclrtFree(creditIpcMem_[rank_]);
        creditIpcMem_[rank_] = nullptr;
        return TILEXR_ERROR_INTERNAL;
    }

    std::string name(nameBuffer);
    if (SetIpcPidSdid(name, pids, sdids) != TILEXR_SUCCESS) {
        aclrtFree(creditIpcMem_[rank_]);
        creditIpcMem_[rank_] = nullptr;
        return TILEXR_ERROR_INTERNAL;
    }

    char names[TileXR::TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE] = {};
    name.resize(IPC_NAME_SIZE);
    if (GetName(name, names) != TILEXR_SUCCESS) {
        aclrtFree(creditIpcMem_[rank_]);
        creditIpcMem_[rank_] = nullptr;
        return TILEXR_ERROR_INTERNAL;
    }
    return OpenCreditIpcMem(names);
}

int TileXRComm::OpenCreditIpcMem(
    const char names[TileXR::TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE])
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    for (int peer = 0; peer < rankSize_; ++peer) {
        if (peer == rank_) {
            continue;
        }
        if (SkipUnusedChannel910B2C(rank_, peer, GetChipName())) {
            continue;
        }
        if (rtIpcOpenMemory(
                reinterpret_cast<void **>(&creditIpcMem_[peer]),
                names[peer]) != RT_ERROR_NONE) {
            CloseCreditIpcMem();
            aclrtFree(creditIpcMem_[rank_]);
            creditIpcMem_[rank_] = nullptr;
            return TILEXR_ERROR_INTERNAL;
        }
    }
    creditIpcMemInited_ = true;
    return TILEXR_SUCCESS;
}
```

上面在当前代码基础上补齐了初始化失败时本 rank owner allocation 的释放，适合作为其它项目直接采用的版本。没有 910B2C 特殊拓扑时，应删除 `SkipUnusedChannel910B2C()` 判断，确保每个通信 peer 都完成映射。

通信域主初始化中的调用位置必须在 `SyncCommArgs()` 之前：

```cpp
if (IsEnvEnabled("TILEXR_ENABLE_CREDIT_IPC", false)) {
    const int ret = InitCreditCommMem();
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
}

// InitUDMA / InitSDMA ...

const int ret = SyncCommArgs();
if (ret != TILEXR_SUCCESS) {
    return ret;
}
```

### 13.7 `CommArgs` Host 到 Device

实际同步代码的 credit 部分如下：

```cpp
int TileXRComm::SyncCommArgs()
{
    commArgs_.rank = rank_;
    commArgs_.localRank = localRank_;
    commArgs_.rankSize = rankSize_;
    commArgs_.localRankSize = localRankSize_;

    for (int peer = 0; peer < rankSize_; ++peer) {
        commArgs_.creditMems[peer] = creditIpcMem_[peer];
    }

    int ret = aclrtMalloc(
        reinterpret_cast<void **>(&commArgsPtr_), sizeof(commArgs_),
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return TILEXR_ERROR_INTERNAL;
    }

    ret = aclrtMemcpy(
        commArgsPtr_, sizeof(commArgs_),
        &commArgs_, sizeof(commArgs_),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(commArgsPtr_);
        commArgsPtr_ = nullptr;
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}
```

启用 credit 的 Host 在 launch 前必须验证：

```cpp
for (int peer = 0; peer < rankSize; ++peer) {
    if (commArgsHost.creditMems[peer] == nullptr) {
        std::cerr << "missing credit IPC mapping, peer=" << peer << '\n';
        return false;
    }
}

static_assert(
    kAllToAllGroupCreditStride ==
        static_cast<size_t>(TileXR::CREDIT_IPC_STRIDE));
static_assert(
    kAllToAllGroupCreditSlotBytes ==
        static_cast<size_t>(TileXR::CREDIT_IPC_SLOT_BYTES));
```

### 13.8 Device 侧完整 MTE set/wait helper

以下代码可直接放入 Ascend C kernel。调用者提供至少 512 B 的 `relayLocal`，并保证同一个 core 不会同时把它用于其它未完成的 MTE 操作：

```cpp
#include "kernel_operator.h"

namespace {

constexpr uint32_t kCreditStride = 512U;
constexpr uint32_t kCreditWords = kCreditStride / sizeof(uint64_t);
constexpr uint64_t kCreditWaitTimeoutCycles = 10000000000ULL;

static_assert(
    kCreditStride == TileXR::CREDIT_IPC_STRIDE,
    "credit stride must match Host IPC layout");

__aicore__ inline uint64_t MakeCreditToken(
    uint32_t invocationId, uint32_t group, uint32_t pass)
{
    const uint64_t invocation = static_cast<uint64_t>(invocationId) + 1ULL;
    const uint64_t slot = static_cast<uint64_t>(invocationId & 1U);
    return (invocation << 32U) |
        (slot << 31U) |
        (static_cast<uint64_t>(group) << 16U) |
        (static_cast<uint64_t>(pass) + 1ULL);
}

__aicore__ inline uint64_t LoadCreditMte(
    __gm__ uint64_t *credit,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    AscendC::GlobalTensor<uint64_t> creditGlobal;
    creditGlobal.SetGlobalBuffer(credit, kCreditWords);
    auto creditLocal = relayLocal.ReinterpretCast<uint64_t>();

    AscendC::DataCopy(creditLocal, creditGlobal, kCreditWords);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return creditLocal.GetValue(0);
}

__aicore__ inline bool WaitCreditMte(
    __gm__ uint64_t *credit, uint64_t expectedToken,
    uint64_t timeoutCycles,
    AscendC::LocalTensor<uint8_t> relayLocal,
    uint64_t &observed)
{
    const uint64_t begin =
        static_cast<uint64_t>(AscendC::GetSystemCycle());
    observed = LoadCreditMte(credit, relayLocal);
    while (observed < expectedToken) {
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - begin >=
            timeoutCycles) {
            return false;
        }
        observed = LoadCreditMte(credit, relayLocal);
    }
    return true;
}

__aicore__ inline void PublishCreditMte(
    __gm__ uint8_t *remoteOwnerBuffer,
    uint64_t slotOffset, int32_t grantingRank,
    uint64_t creditToken,
    AscendC::LocalTensor<uint8_t> relayLocal)
{
    auto remoteCredit = reinterpret_cast<__gm__ uint64_t *>(
        remoteOwnerBuffer + slotOffset +
        static_cast<uint64_t>(grantingRank) * kCreditStride);

    auto creditLocal = relayLocal.ReinterpretCast<uint64_t>();
    creditLocal.SetValue(0, creditToken);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<uint64_t> remoteCreditGlobal;
    remoteCreditGlobal.SetGlobalBuffer(remoteCredit, kCreditWords);
    AscendC::DataCopy(remoteCreditGlobal, creditLocal, kCreditWords);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

} // namespace
```

`PublishCreditMte()` 是从当前 `AllToAllGroupPublishNextCredit()` 提取出的通用形式，数据搬运和同步指令完全相同，只把 grouped peer 计算移到调用侧，方便其它项目复用。

### 13.9 Device 调用位置和地址计算

接收端 `R` 确认 group `g` 的 payload 到达后，为下一 group 的发送端 `S` 发布 credit：

```cpp
const uint32_t slot = invocationId & 1U;
const uint64_t slotOffsets[2] = {
    0ULL,
    static_cast<uint64_t>(TileXR::CREDIT_IPC_SLOT_BYTES)
};
const uint64_t token = MakeCreditToken(invocationId, group + 1U, 0U);

// args->creditMems[S] 是 S 拥有的 buffer 在 R 地址空间中的 IPC mapping。
PublishCreditMte(
    args->creditMems[S], slotOffsets[slot], R, token, relayLocal);
```

发送端 `S` 在向 `R` 发送 group `g > 0` 前等待：

```cpp
const uint32_t slot = invocationId & 1U;
const uint64_t slotOffset = slot == 0U
    ? 0ULL
    : static_cast<uint64_t>(TileXR::CREDIT_IPC_SLOT_BYTES);
const uint64_t expected = MakeCreditToken(invocationId, group, 0U);

// args->creditMems[S] 是 S 自己的 owner buffer；entry R 由 R 写入。
auto credit = reinterpret_cast<__gm__ uint64_t *>(
    args->creditMems[S] + slotOffset +
    static_cast<uint64_t>(R) * kCreditStride);

uint64_t observed = 0ULL;
if (!WaitCreditMte(
        credit, expected, kCreditWaitTimeoutCycles,
        relayLocal, observed)) {
    // 必须记录 S、R、group、expected 和 observed，并让整个算子报错。
    return;
}
```

Grouped all-to-all 当前实际变量代入为：

```text
S = 当前 send core 所在 rank
R = 当前 group/lane 算出的 peer
发布侧 next S = 接收 rank 在下一 group/同 lane 算出的 nextPeer
group 0 不 wait
最后一个 group 不再发布 next credit
routeElements == 0 时不 wait，也不发送该 route
```

### 13.10 Kernel launch 的实际 Host/Device ABI

当前代码不是通过 `<<<>>>` 启动，而是注册 kernel binary 后使用 `rtKernelLaunchWithFlagV2()`。Credit kernel 的实际参数结构如下：

```cpp
struct alignas(8) GroupedAllToAllCreditKernelArgs {
    uint8_t* commArgs;
    uint8_t* input;
    uint8_t* output;
    uint8_t* registeredMemory;
    uint8_t* debug;
    uint32_t invocationId;
    int32_t elementsPerPeer;
    int32_t chunkElements;
    uint32_t passCount;
    uint32_t groupCount;
    uint64_t payloadOffset0;
    uint64_t payloadOffset1;
    uint64_t signalOffset0;
    uint64_t signalOffset1;
    uint64_t creditOffset0;
    uint64_t creditOffset1;
    uint8_t* groupTrace;
    uint32_t traceIteration;
    uint32_t routeStage;
    uint32_t multiChannel;
    uint32_t primaryRouteParts;
    uint32_t groupWidth;
    uint32_t quietBatch;
    uint32_t ingressWindow;
};

static_assert(sizeof(GroupedAllToAllCreditKernelArgs) == 152U);
static_assert(offsetof(GroupedAllToAllCreditKernelArgs, creditOffset0) == 96U);
static_assert(offsetof(GroupedAllToAllCreditKernelArgs, groupTrace) == 112U);
```

实际组参和 launch 方式：

```cpp
GroupedAllToAllCreditKernelArgs args {
    commArgs, input, output, registeredMemory, debug, invocationId,
    elementsPerPeer, chunkElements, passCount, groupCount,
    payloadOffset0, payloadOffset1, signalOffset0, signalOffset1,
    creditOffset0, creditOffset1, groupTrace, traceIteration,
    routeStage, multiChannel, primaryRouteParts, groupWidth,
    quietBatch, ingressWindow,
};

rtArgsEx_t argsInfo {};
argsInfo.args = static_cast<void*>(&args);
argsInfo.argsSize = sizeof(args);

rtTaskCfgInfo_t cfgInfo {};
cfgInfo.schemMode = RT_SCHEM_MODE_NORMAL;
const rtError_t ret = rtKernelLaunchWithFlagV2(
    groupedCreditKernelFunctionSignature,
    blockDim,
    &argsInfo,
    nullptr,
    static_cast<rtStream_t>(stream),
    0U,
    &cfgInfo);
```

Device kernel 声明必须保持完全相同的参数顺序：

```cpp
extern "C" __global__ __aicore__ void grouped_alltoall_credit_kernel(
    GM_ADDR commArgsGM, GM_ADDR inputGM, GM_ADDR outputGM,
    GM_ADDR registeredMemoryGM, GM_ADDR debugGM,
    uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    uint64_t creditOffset0, uint64_t creditOffset1,
    GM_ADDR groupTraceGM, uint32_t traceIteration,
    uint32_t routeStage, uint32_t multiChannel,
    uint32_t primaryRouteParts, uint32_t groupWidth,
    uint32_t quietBatch, uint32_t ingressWindow)
{
    // 调用包含 PublishCreditMte/WaitCreditMte 的 kernel 实现。
}
```

如果目标项目使用直接 kernel launch，也仍应保留参数结构的 `sizeof/offsetof` 校验；Host 和 Device 参数错位通常不会在编译期报错，而会表现为错误地址或 kernel 卡死。

### 13.11 完整释放代码

```cpp
void TileXRComm::CloseCreditIpcMem()
{
    for (int peer = 0; peer < rankSize_; ++peer) {
        if (peer == rank_ || creditIpcMem_[peer] == nullptr) {
            continue;
        }
        rtIpcCloseMemory(static_cast<void *>(creditIpcMem_[peer]));
        creditIpcMem_[peer] = nullptr;
    }
}

TileXRComm::~TileXRComm()
{
    // 析构前必须保证所有使用 creditMems 的 kernel/stream 已结束。
    if (creditIpcMemInited_) {
        CloseCreditIpcMem();
        creditIpcMemInited_ = false;
    }

    if (creditIpcMem_[rank_] != nullptr) {
        aclrtFree(creditIpcMem_[rank_]);
        creditIpcMem_[rank_] = nullptr;
    }

    if (commArgsPtr_ != nullptr) {
        aclrtFree(commArgsPtr_);
        commArgsPtr_ = nullptr;
    }
}
```

如果通信域析构前没有隐式 stream synchronize，目标项目必须显式同步；仅 Host barrier 不能证明 Device 已停止访问 IPC mapping。
