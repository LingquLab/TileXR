# 模块：comm（核心通信库）

## 1. 模块作用

`comm` 是 TileXR 的核心通信模块，负责多 rank 分布式通信的初始化、IPC 共享内存管理、peer 设备内存访问，以及与底层 HCCL/CANN 运行时的对接。最终以 `libtile-comm.so` 形式对外提供服务。

---

## 2. 目录结构

```
src/comm/
├── tilexr_comm.h              # TileXRComm 类声明
├── tilexr_comm.cpp            # TileXRComm 实现（Init、IPC、peer 内存）
├── tilexr_internal.h          # 内部工具函数声明
├── tilexr_internal.cpp        # RegistKernel / LoadMTE / GetChipName 等实现
├── comm_wrap.cpp              # C API 包装，对接 tilexr_api.h
├── ccl_kernel_args.h          # AICore kernel 参数结构体
├── tilexr_sock_exchange.h     # socket 交换头文件
├── tilexr_sock_exchange.cpp   # rank 间 socket 同步（init 阶段握手）
└── CMakeLists.txt
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `tilexr_comm.cpp` | `TileXRComm::Init()` 的主体逻辑：分配 IPC 共享内存（100 MB buffer + 2 MB flag）、注册 peer 设备内存、socket 握手同步 |
| `tilexr_internal.cpp` | `RegistKernel()`（注册 AICore 核函数）、`LoadMTE()`（加载 MTE 指令）、`GetChipName()`（SOC 类型探测）、`GetCoreNum()`（核心数获取） |
| `comm_wrap.cpp` | 将 C++ 的 `TileXRComm` 实例封装为不透明指针（`TileXRCommPtr`），实现 `tilexr_api.h` 中所有 C 函数 |
| `tilexr_sock_exchange.cpp` | 基于 TCP socket 实现 rank 之间的 meta 信息交换（IPC handle、设备地址等） |

---

## 4. 核心函数 / 类 / 接口

### `class TileXRComm`（`tilexr_comm.h`）

```cpp
// 构造函数（三种，对应不同初始化方式）
TileXRComm(int rank, int rankSize);
TileXRComm(int rank, int rankSize, int commDomain, int bufferSize);
TileXRComm(int rank, int rankSize, TileXRUniqueId commId);

// 初始化
HcclResult Init();
HcclResult InitThread();

// 查询接口
int GetRank() const;
int GetRankSize() const;
int GetCommSize() const;
PhysicalInfo GetPhysicalInfo() const;

// 通信参数（传递给 kernel）
CommArgs* GetCommArgsPtr();
CommArgs  GetCommArgs();

// 诊断
void PrintDFX();
```

### 内部工具函数（`tilexr_internal.h`）

```cpp
HcclResult RegistKernel(...);        // 注册 AICore kernel
size_t     Count2Size(size_t count); // 元素数 → 字节数
HcclResult LoadMTE(...);             // 加载 MTE 指令集
ChipName   GetChipName();            // 返回当前 SOC 类型
int        GetCoreNum();             // 返回 AICore 数量
```

### 关键 C API（`comm_wrap.cpp` 实现，`tilexr_api.h` 声明）

```c
TileXRCommPtr TileXRCommInitRankLocal(int rank, int rankSize);
TileXRCommPtr TileXRCommInitRank(int rank, int rankSize, TileXRUniqueId id);
TileXRCommPtr TileXRCommInitRankWithCustDomainSize(int rank, int rankSize, int domain, int bufSize);
int           TileXRGetCommArgsDev(TileXRCommPtr comm, GM_ADDR *argsAddr);
int           TileXRGetCommArgsHost(TileXRCommPtr comm, CommArgs *args);
void          LcclCommDestroy(TileXRCommPtr comm);
```

---

## 5. 数据流向

```
用户代码
  │
  ▼
TileXRCommInitRank*()   ← comm_wrap.cpp（C API入口）
  │
  ▼
TileXRComm::Init()      ← tilexr_comm.cpp
  ├── GetChipName() / GetCoreNum()   ← tilexr_internal.cpp
  ├── 分配 IPC 共享内存（ipcMem_）  ← CANN hrt API
  ├── SockExchange 握手             ← tilexr_sock_exchange.cpp
  │       交换 IPC handle、peer 地址
  ├── 注册 peer 设备内存（peerMems_）
  ├── SyncCommArgs() → CommArgs 写入 GPU 内存
  └── RegistKernel()                ← tilexr_internal.cpp
  │
  ▼
CommArgs（设备侧结构）
  └── 通过 TileXRGetCommArgsDev() 传入 kernel
```

---

## 6. 关键业务逻辑

### IPC 共享内存布局
每个 rank 分配两段连续内存：
- `[0, 204MB)` — 数据 buffer（`TILEXR_BUFF_BYTES`）
- `[204MB, 208MB)` — flag buffer（`TILEXR_FLAG_BUFF_BYTES = 4MB`）

这两段内存通过 CANN IPC 机制对所有 rank 可见，地址写入 `CommArgs.peerMems[]`。

### 910B2C 特殊处理
`SkipUnusedChannel910B2C()` 在 910B2C 上会跳过部分未接线的通信通道（拓扑枚举：HCCS/PIX/PIB/PHB/SYS/SIO/HCCS_SW），避免初始化不存在的链路。

### SyncCommArgs
`SyncCommArgs()` 在 host 侧填充 `CommArgs` 结构，然后通过 `aclrtMemcpy` H2D 同步到 device，供 kernel 直接访问。

---

## 7. 开发注意事项

- `TileXRComm` 必须在每个 rank 进程中独立实例化；`TileXRCommInitRankLocal` 不依赖 socket 握手，适合同进程多 rank 场景。
- `CommArgs.peerMems` 数组最多支持 `TILEXR_MAX_RANK_SIZE = 128` 个 rank。
- `Init()` 内部依赖 `TILEXR_HOME`、`TILEXR_CANN_HOME` 环境变量（由 `common_env.sh` 设置）。
- `tilexr_sock_exchange.cpp` 中的 socket 握手不可复用，每次 `Init()` 独立建立并在握手完成后关闭。
- 修改 `CommArgs` 结构体字段时，需同步更新 `comm_args.h` 和所有 kernel 侧引用（`mc2/` 下的 kernel 文件）。

---

## 8. 未来可扩展点

- **多 buffer 域**：`TileXRCommInitRankWithCustDomainSize` 已预留 `commDomain` 参数，目前未充分利用，可扩展为不同算子使用独立内存域。
- **RDMA 路径**：`ExtraFlag::RDMA` 已预留，当前单机走 IPC，多机扩展可通过此 flag 切换 ibverbs 路径。
- **动态 buffer 大小**：目前 `TILEXR_BUFF_BYTES` 是编译期常量，可改为运行时从环境变量读取（`TILEXR_COMM_BUFFER_SIZE` 已部分支持）。
- **零拷贝 flag 优化**：`SyncCommArgs` 目前全量 H2D 拷贝，可按需增量更新 flag 字段。
