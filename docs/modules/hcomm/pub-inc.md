# 模块：hcomm/pub-inc（公共接口头文件层）

## 1. 模块作用

定义 hcomm 对外暴露的所有公共接口、类型、枚举和常量。是 hcomm 内部各层与外部调用方（TileXR、ops-transformer 等）之间的契约层。分为三类：新 API（`new/`）、AICPU 接口（`aicpu/`）、内部 RMA buffer 抽象（`inner/`）。

---

## 2. 目录结构

```
src/pub_inc/
├── hccl_common.h              # 核心公共类型、枚举、常量（427 行）
├── hccl_network_pub.h         # 网络层公共接口
├── transport_pub.h            # Transport 层公共接口
├── dispatcher.h               # Dispatcher 接口
├── adapter_pub.h              # 平台适配器公共接口
├── workflow_pub.h             # 工作流公共接口
├── new/                       # 新一代接口
│   ├── hccl_dispatcher_ctx.h  # Dispatcher 上下文
│   ├── hccl_mem_transport.h   # 内存 transport 接口
│   ├── hccl_primitive.h       # 通信原语接口
│   └── hccl_socket.h          # Socket 接口
├── aicpu/                     # AICPU 接口
│   ├── aicpu_hccl_sqcq.h      # SQ/CQ 队列接口（通用）
│   ├── aicpu_hccl_sqcqv1.h    # SQ/CQ v1
│   └── aicpu_hccl_sqcqv2.h    # SQ/CQ v2
└── inner/                     # 内部 RMA buffer 抽象
    ├── local_ipc_rma_buffer.h
    ├── remote_rdma_rma_buffer.h
    └── ...
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `hccl_common.h` | 全局类型别名（`HcclRtStream`、`RdmaHandle`、`SocketHandle`）、设备限制常量、超时常量、`LinkTypeInServer`/`TransportType`/`SyncMode` 枚举、`RemoteRankInfo`/`HcclSignalInfo` 结构体 |
| `transport_pub.h` | Transport 层对外接口：创建/销毁 transport、发送/接收原语 |
| `dispatcher.h` | Dispatcher 接口：任务调度、stream 绑定 |
| `new/hccl_socket.h` | 新 socket 接口：`HcclSocket` 类，封装 ibverbs/TCP socket 生命周期 |
| `new/hccl_primitive.h` | 通信原语：Put/Get/Send/Recv 的统一抽象 |
| `aicpu/aicpu_hccl_sqcq.h` | AICPU 侧 SQ/CQ 队列接口，用于 AICPU 模式下的任务提交 |

---

## 4. 核心类型 / 枚举 / 常量

### `hccl_common.h` 关键定义

```cpp
// 类型别名
using HcclRtStream   = rtStream_t;
using HcclRtNotify   = rtNotify_t;
using RdmaHandle     = void*;
using SocketHandle   = void*;

// 设备与内存常量
constexpr u32 HCCL_MAX_DEVICE_NUM     = 8;
constexpr u32 HCCL_MAX_RANK_NUM       = 4096;
constexpr u64 HCCL_MAX_MEMORY_SIZE    = 512ULL * 1024 * 1024 * 1024; // 512GB

// 超时常量
constexpr u32 HCCL_SOCKET_TIMEOUT_MS  = 120000; // 2 min
constexpr u32 HCCL_CONNECT_TIMEOUT_MS = 30000;  // 30 s

// 链路类型（LinkTypeInServer）
enum class LinkTypeInServer {
    HCCS_TYPE, PIX_TYPE, PIB_TYPE, PHB_TYPE, SYS_TYPE, SIO_TYPE, HCCS_SW_TYPE
};

// Transport 类型
enum class TransportType {
    TRANS_TYPE_P2P,       // 片内 P2P（SDMA）
    TRANS_TYPE_IBV,       // ibverbs RDMA
    TRANS_TYPE_SOCKET,    // TCP socket
    TRANS_TYPE_HCCS,      // HCCS 互联
};

// 关键结构体
struct RemoteRankInfo {
    u32 userRank;
    HcclIpAddress nicIp;
    u32 devicePort;
    // ...
};
```

---

## 5. 数据流向

```
外部调用方（TileXR / ops-transformer）
  └── 引用 pub_inc/ 头文件
        ├── hccl_common.h → 类型定义（全局共享）
        ├── transport_pub.h → Transport 接口
        ├── dispatcher.h → 任务调度接口
        └── new/hccl_socket.h → Socket 操作

hcomm 内部各层
  └── 同样引用 pub_inc/，保证接口一致性
```

---

## 6. 关键业务逻辑

### 接口分层
- `new/` 目录是新一代接口，逐步替代旧接口；旧接口保留在根目录以兼容存量代码。
- `aicpu/` 接口专用于 AICPU 模式（算子在 AICPU 上执行而非 AICore），有独立的 SQ/CQ 队列协议。
- `inner/` 的 RMA buffer 抽象统一了 IPC（本地）和 RDMA（远端）两种内存访问路径。

### `TransportType` 选择逻辑
上层通过 `LinkTypeInServer` 和拓扑信息决定使用哪种 `TransportType`：
- 同卡内：`P2P`（SDMA）
- 同机跨卡 HCCS：`HCCS`
- 同机跨卡 PCIe：`IBV`（单机 ibverbs）
- 跨机：`IBV`（RDMA）或 `SOCKET`（TCP fallback）

---

## 7. 开发注意事项

- `hccl_common.h` 是全局共享头文件，修改任何类型或常量都会影响所有模块，需谨慎评估。
- `new/` 接口与旧接口并存期间，新功能优先使用 `new/` 接口，避免在旧接口上叠加逻辑。
- `aicpu/` 的 SQ/CQ 版本（v1/v2）对应不同硬件代际，新芯片优先使用 v2。

---

## 8. 未来可扩展点

- **接口统一**：将根目录旧接口逐步迁移到 `new/`，最终废弃旧接口。
- **Python 绑定**：`hccl_common.h` 中的枚举和结构体可通过 pybind11 暴露给 Python 侧。
- **版本化接口**：在 `new/` 中增加接口版本字段，支持滚动升级时的向后兼容。
