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
| `hccl_common.h` | 全局类型别名、设备限制常量、超时常量、`LinkTypeInServer`/`TransportType`/`SyncMode`/`NICDeployment`/`QPMode` 枚举、`HcclSignalInfo`/`IpSocket`/`RaResourceInfo`/`OpCounterInfo` 结构体（427 行） |
| `transport_pub.h` | Transport 类（完整 Tx/Rx/RDMA 接口）、`MachinePara`/`TransportDeviceNormalData`/`TransportDeviceP2pData`/`TransportDeviceIbverbsData` 结构体、`HcclQpInfoV2`/`AddrKey`/`MemDetails` 结构体 |
| `dispatcher.h` | `DispatcherType` 枚举、`HcclDispatcherInit`/`HcclD2DMemcpyAsync`/`HcclReduceAsync`/`HcclSignalRecord`/`InitTask`/`LaunchTask` 等 C 接口（约 40 个函数） |
| `new/hccl_socket.h` | C 接口：`HcclSocketCreate`/`Close`/`Listen`/`Accept`/`Connect`/`ISend`/`IRecv` + 白名单管理；`SocketWlistInfo` 结构体 |
| `new/hccl_primitive.h` | 通信原语：Put/Get/Send/Recv 的统一抽象 |
| `aicpu/aicpu_hccl_sqcq.h` | AICPU 侧 SQ/CQ 队列接口，用于 AICPU 模式下的任务提交 |

---

## 4. 核心类型 / 枚举 / 常量

### `hccl_common.h` 关键定义

```cpp
// 类型别名
using HcclDispatcher = void*;
using HcclRtStream   = void*;
using HcclRtNotify   = void*;
using RdmaHandle     = void*;
using SocketHandle   = void*;

// 设备与内存常量
constexpr u32 HCCL_AISERVER_DEVICE_NUM = 8;      // 每台服务器最多设备数
constexpr u32 MAX_MODULE_DEVICE_NUM    = 32;     // 双模块最大设备数
constexpr u64 DEVICE_MEMORY_MAX_ALLOC_SIZE = 16ULL * 1024 * 1024 * 1024; // 16GB

// 通知等待超时
constexpr s32 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;   // 1836
constexpr s32 NOTIFY_MAX_WAIT_TIME     = 255 * 68;
constexpr u32 NOTIFY_INVALID_WAIT_TIME = 0xFFFFFFFF;

// 端口常量
constexpr u32 HETEROG_CCL_PORT        = 16666;   // 默认通信端口
constexpr u32 HOST_CONTROL_BASE_PORT  = 60000;
constexpr u32 HOST_PORT_MAX           = 65520;

// 链路类型
enum class LinkTypeInServer {
    HCCS_TYPE = 0, PXI_TYPE = 1, SIO_TYPE = 2, HCCS_SW_TYPE = 3, RESERVED_LINK_TYPE
};

// Transport 类型（完整枚举）
enum class TransportType {
    TRANS_TYPE_IBV_EXP    = 0,
    TRANS_TYPE_P2P        = 1,
    TRANS_TYPE_HOST_SHM   = 2,
    TRANS_TYPE_HOST_TCP   = 3,
    TRANS_TYPE_ROCE       = 4,
    TRANS_TYPE_HETEROG_P2P    = 5,
    TRANS_TYPE_HETEROG_ROCE   = 6,
    TRANS_TYPE_DEVICE_P2P     = 7,
    TRANS_TYPE_DEVICE_IBVERBS = 8,  // MC2 单机 ibverbs
    TRANS_TYPE_DEVICE_DIRECT  = 9,
    TRANS_TYPE_RESERVED   = 255
};

// NIC 部署方式
enum class NICDeployment {
    NIC_DEPLOYMENT_HOST   = 0,
    NIC_DEPLOYMENT_DEVICE,
    NIC_DEPLOYMENT_RESERVED
};

// QP 模式
enum class QPMode { INVALID = -1, NORMAL = 0, OFFLOAD = 1 };

// 同步模式
enum class SyncMode {
    DEFAULT_TIMEWAITSYNCMODE = 0,
    CONFIGURABLE_TIMEWAITSYNCMODE = 1,
    UNLIMITED_TIMEWAITSYNCMODE
};

// 关键结构体
struct HcclSignalInfo {
    u64 resId;   // eventid or notifyid
    u64 addr;
    u32 devId;
    u32 tsId;
    u32 rankId;
    u32 flag;
};

struct OpCounterInfo {
    u64  headCountMem    = 0;
    u64  tailCountMem    = 0;
    u64  addOneMem       = 0;
    u32  memSize         = 0;
    bool isEnableCounter = false;
};

// IpSocket：单个 NIC 上的 socket 句柄
using IpSocket = struct IpSocket {
    SocketHandle nicSocketHandle;   // nullptr by default
    RdmaHandle   nicRdmaHandle;     // nullptr by default
    std::set<u32> listenedPort;
};

// RaResourceInfo：VNIC/NIC/host 网络的 socket 映射
using RaResourceInfo = struct TagRaResourceInfo {
    std::map<HcclIpAddress, IpSocket> vnicSocketMap;
    std::map<HcclIpAddress, IpSocket> nicSocketMap;
    std::map<HcclIpAddress, IpSocket> hostNetSocketMap;
};

struct RemoteRankInfo {
    s32 remoteDeviceId;
    u32 remoteRank;
    s32 remotePid;
    s32 remoteSdid;
};
```

### `transport_pub.h` 核心结构体

```cpp
// QP 信息（带重试配置）
#pragma pack(push, 4)
struct HcclQpInfoV2 {
    u64 qpPtr;
    u32 sqIndex;
    u32 dbIndex;
    u16 retryCnt;
    u16 retryTime;
};
#pragma pack(pop)

// 内存地址+key 对（RDMA 操作必用）
struct AddrKey {
    u64 addr     = 0;
    u32 key      = 0;
    u32 notifyId = INVALID_UINT;
};

struct MemDetails { u64 size = 0; u64 addr = 0; u32 key = 0; };

// P2P transport 数据（IPC 共享内存）
struct TransportDeviceP2pData {
    void* inputBufferPtr;
    void* outputBufferPtr;
    std::shared_ptr<LocalNotify>  ipcPreWaitNotify;
    std::shared_ptr<LocalNotify>  ipcPostWaitNotify;
    std::shared_ptr<RemoteNotify> ipcPreRecordNotify;
    std::shared_ptr<RemoteNotify> ipcPostRecordNotify;
    TransportAttr transportAttr;
};

// ibverbs transport 数据（RDMA）
struct TransportDeviceIbverbsData {
    void*           inputBufferPtr;
    void*           outputBufferPtr;
    MemDetails      localInputMem;
    MemDetails      localOutputMem;
    AddrKey         remoteAckNotifyDetails;
    AddrKey         remoteDataNotifyDetails;
    AddrKey         remoteDataAckNotifyDetails;
    std::vector<HcclQpInfoV2> qpInfo;
    uint32_t        remoteInputKey;
    uint32_t        remoteOutputKey;
    uint32_t        notifySize;
    u32             multiQpThreshold;
    u32             qpsPerConnection;
    bool            useAtomicWrite;
};

// 普通（ibverbs RDMA Read/Write）transport 数据
struct TransportDeviceNormalData {
    MemDetails  remoteInputMem;
    MemDetails  remoteOutputMem;
    MemDetails  localInputMem;
    MemDetails  localOutputMem;
    HcclQpInfoV2 qpInfo;
    QPMode      qpMode;
};

// 连接参数（涵盖 RDMA/TCP/PCIe 所有场景）
using MachinePara = struct TagMachinePara {
    MachineType  machineType;
    LinkMode     linkMode;
    std::string  collectiveId;
    std::string  tag;
    HcclIpAddress localIpAddr;
    HcclIpAddress remoteIpAddr;
    u32          localSocketPort;
    u32          remoteSocketPort;
    s32          localDeviceId;
    s32          remoteDeviceId;
    NICDeployment nicDeploy;
    DevType       deviceType;
    std::vector<std::shared_ptr<HcclSocket>> sockets;
    std::vector<u8> exchangeInfo;
    DeviceMem    inputMem;
    DeviceMem    outputMem;
    QPMode       qpMode;
    u32          notifyNum;
    bool         isAicpuModeEn;
    // ...（还有 tc/sl/specifyLink/enableAtomicWrite 等）
};

// Transport 关系常量
constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_CHIP     = 0x1U << 0;
constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER   = 0x1U << 1;
constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD = 0x1U << 2;
constexpr u64 MAX_EXCHANGE_DATA_LEN = 2ULL * 1024 * 1024; // 2 MB
```

### `Transport` 类（`transport_pub.h`，`namespace hccl`）

```cpp
using LINK = std::shared_ptr<Transport>;

class Transport {
    // 生命周期
    HcclResult Init();
    HcclResult DeInit();

    // Tx 接口
    HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream);
    HcclResult TxWithReduce(UserMemType, u64, const void*, u64, HcclDataType, HcclReduceOp, Stream&);
    HcclResult TxData(UserMemType, u64, const void*, u64, Stream&);
    HcclResult TxAck(Stream&);
    HcclResult TxPrepare(Stream&);
    HcclResult TxDone(Stream&);

    // Rx 接口
    HcclResult RxAsync(UserMemType, u64, void*, u64, Stream&);
    HcclResult RxWithReduce(...);
    HcclResult RxData(UserMemType, u64, void*, u64, Stream&);
    HcclResult RxAck(Stream&);
    HcclResult RxDone(Stream&);

    // RDMA Write/Read（原始接口）
    HcclResult WriteAsync(Buffer& remoteBuf, Buffer& localBuf, Stream& stream);
    HcclResult WriteSync(Buffer& remoteBuf, Buffer& localBuf, Stream& stream);
    HcclResult ReadAsync(Buffer& localBuf, Buffer& remoteBuf, Stream& stream);
    HcclResult ReadSync(Buffer& localBuf, Buffer& remoteBuf, Stream& stream);

    // 批量 RDMA（静态方法，MC2 场景用）
    static HcclResult HcclBatchRead(const TransportDeviceNormalData& ibvData,
        MemDetails* localMems, MemDetails* remoteMems, u32 memNum, u64& dbInfo);
    static HcclResult HcclBatchWrite(const TransportDeviceNormalData& ibvData,
        MemDetails* localMems, MemDetails* remoteMems, u32 memNum, u64& dbInfo);

    // Notify 同步
    HcclResult Post(u32 notifyIdx, Stream& stream);
    HcclResult Wait(u32 notifyIdx, Stream& stream, u32 timeOut = NOTIFY_INVALID_WAIT_TIME);
    HcclResult PostReady(Stream& stream);
    HcclResult WaitReady(Stream& stream);

    // Getters
    TransportType  GetTransportType() const;
    u32            GetRemoteRank();
    bool           IsTransportRoce();
    HcclResult     GetRemoteMem(UserMemType, void** remotePtr);
    HcclResult     GetLocalMemDetails(UserMemType, MemDetails&);
    HcclResult     GetAiQpInfo(std::vector<HcclQpInfoV2>&);
    static HcclResult GetTransportErrorCqe(HcclNetDevCtx,
        std::vector<std::pair<Transport*, CqeInfo>>&, u32& num);
};
```

### `new/hccl_socket.h` — Socket C 接口

```cpp
typedef void* HcclSocket;
const uint32_t HCCL_SOCK_CONN_TAG_MAX_SIZE = 192;

typedef struct {
    HcclAddr remoteAddr;
    uint32_t connMaxNum;   // 0 = 无限制
    char     handShakeTag[HCCL_SOCK_CONN_TAG_MAX_SIZE];
} SocketWlistInfo;

// 生命周期
HcclResult HcclSocketCreate(HcclNetDevDeployment, int32_t devicePhyId,
    int domain, const struct sockaddr* addr, socklen_t addrlen, HcclSocket* socket);
HcclResult HcclSocketClose(HcclSocket socket);

// 服务端
HcclResult HcclSocketListen(HcclSocket socket, int32_t backlog);
HcclResult HcclSocketAccept(void* serverSocket, char* handShakeTag,
    uint32_t tagLen, HcclSocket* socket);   // 非阻塞

// 客户端
HcclResult HcclSocketConnect(HcclSocket socket, const struct sockaddr* addr,
    socklen_t addrlen, char* handShakeTag, uint32_t tagLen);  // 非阻塞

// 数据传输（非阻塞）
HcclResult HcclSocketISend(HcclSocket, void* data, uint64_t len, uint64_t* sendLen);
HcclResult HcclSocketIRecv(HcclSocket, void* recvBuf, uint64_t len, uint64_t* recvLen);
HcclResult HcclSocketGetStatus(HcclSocket, int32_t* status);

// 白名单管理
HcclResult HcclSocketEnableWhiteList(HcclSocket);
HcclResult HcclSocketDisableWhiteList(HcclSocket);
HcclResult HcclSocketAddWhiteList(HcclSocket, SocketWlistInfo* whitelists, uint32_t num);
HcclResult HcclSocketDelWhiteList(HcclSocket, SocketWlistInfo* whitelists, uint32_t num);
```

### `dispatcher.h` — Dispatcher 接口

```cpp
enum class DispatcherType {
    DISPATCHER_NORMAL  = 0,
    DISPATCHER_VIRTURAL,
    DISPATCHER_AICPU,
};

// 初始化/销毁
HcclResult HcclDispatcherInit(DispatcherType type, s32 devicePhyId, HcclDispatcher* dispatcher);
HcclResult HcclDispatcherDestroy(HcclDispatcher dispatcher);

// 配置
HcclResult HcclSetGlobalWorkSpace(HcclDispatcher, std::vector<void*>& globalWorkSpaceAddr);
HcclResult HcclSetNotifyWaitMode(HcclDispatcher, SyncMode notifyWaitMode);
HcclResult HcclSetExecTimeOut(HcclDispatcher, s32 execTimeOut = NOTIFY_DEFAULT_WAIT_TIME);

// 内存/数据操作
HcclResult HcclD2DMemcpyAsync(HcclDispatcher, DeviceMem& dst, const DeviceMem& src, Stream&, ...);
HcclResult HcclMemcpyAsync(HcclDispatcher, void* dst, uint64_t destMax,
    const void* src, uint64_t count, HcclRtMemcpyKind, Stream&, ...);
HcclResult HcclReduceAsync(HcclDispatcher, void* src, uint64_t count,
    HcclDataType, HcclReduceOp, Stream&, void* dst, ...);

// 信号同步
HcclResult HcclSignalRecord(HcclDispatcher, HcclRtNotify signal, Stream&, ...);
HcclResult HcclSignalWait(HcclDispatcher, HcclRtNotify signal, Stream&, ...);

// 任务管理
HcclResult InitTask(HcclDispatcher, Stream&, bool enableCache, const std::string& key, ...);
HcclResult LaunchTask(HcclDispatcher, Stream&);
HcclResult LaunchTaskExtend(HcclDispatcher, Stream&, std::vector<Stream>& subStreams);
HcclResult StreamSync(HcclDispatcher, Stream&);
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
