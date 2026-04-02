# 模块：hcomm/hccd（HCCL 集群守护进程）

## 1. 模块作用

HCCD（HCCL Cluster Daemon）是 hcomm 的集群级守护进程，负责：
- 生成全局唯一的通信域 ID（`CommId`）
- 管理跨进程的通信连接（`HcclCommConn`）
- 提供 PML（Portable Message Layer）接口，支持异构通信（CPU rank 与 NPU rank 混合）
- 作为集群级协调者，处理 rank table 分发和通信初始化

---

## 2. 目录结构

```
src/hccd/
├── hccd.cc/.h                 # HCCD 主入口：CommId 生成、通信初始化
├── hccd_comm.cc/.h            # HCCD 通信管理：连接建立与维护
├── hccd_impl_pml.cc/.h        # PML 接口实现（异构通信）
├── hccl_comm_conn.cc/.h       # 单条通信连接管理
└── hccl_comm_conn_mgr.cc/.h   # 通信连接管理器（多连接）
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `hccd.cc` | `HccdGenerateCommId()`（生成唯一 CommId）、`HcclInitComm()`（通过 rank table 初始化通信）；包含异构 transport 和 rank 一致性检查 |
| `hccd_comm.cc` | HCCD 通信层：管理 HCCD 进程与各 rank 进程之间的连接，处理 rank table 分发 |
| `hccd_impl_pml.cc` | PML（Portable Message Layer）实现：支持 CPU rank 与 NPU rank 混合通信场景 |
| `hccl_comm_conn.cc` | 单条通信连接的生命周期：建立、保活、断开 |
| `hccl_comm_conn_mgr.cc` | 多连接管理器：维护所有活跃连接的映射表，支持连接复用 |

---

## 4. 核心函数 / 类 / 接口

### HCCD 主接口（`hccd.cc`）

```cpp
// 生成全局唯一通信域 ID
// 实现：HccdComm::GetUniqueId() + snprintf(format="%s%s%s",  uniqueId + "-" + "hccl_heterog_group")
HcclResult HccdGenerateCommId(HcclRootHandle* commId);

// 通过 rank table 初始化通信（HCCD 模式下的入口）
HcclResult HcclInitComm(const char* rankTable, u32 rank, HcclComm* comm);
```

### HccdComm 初始化流程

```cpp
class HccdComm {
    // 原子保护的初始化
    HcclResult init() {
        // AtomicInitSet() → impl_->Init()
        // 失败时 AtomicInitClear()
    }

    // 生成唯一 ID（供 HccdGenerateCommId 使用）
    HcclResult GetUniqueId(std::string& uniqueId);
};
```

### 通信连接管理（`hccl_comm_conn_mgr.cc`）

```cpp
class HcclCommConnMgr {
    // 建立到指定 rank 的连接
    HcclResult Connect(u32 remoteRank, const HcclIpAddress& ip, u32 port);

    // 获取已建立的连接
    std::shared_ptr<HcclCommConn> GetConn(u32 remoteRank) const;

    // 断开所有连接
    HcclResult DisconnectAll();
};
```

### PML 接口（`hccd_impl_pml.cc`）

```cpp
// 异构通信：CPU rank 通过 PML 与 NPU rank 通信
class HccdImplPml {
    // 发送消息（CPU → NPU 或 NPU → CPU）
    HcclResult Send(u32 dstRank, const void* buf, size_t size);

    // 接收消息
    HcclResult Recv(u32 srcRank, void* buf, size_t size);
};
```

---

## 5. 数据流向

```
集群启动
  └── HCCD 进程启动
        └── HccdGenerateCommId() → 生成唯一 CommId
              └── 广播 CommId 给所有 rank

各 rank 进程
  └── HcclInitComm(rankTable, rank)
        ├── 解析 rank table
        ├── 连接 HCCD（HcclCommConnMgr::Connect）
        └── 获取全局拓扑信息

异构通信（CPU + NPU 混合）
  CPU rank → HccdImplPml::Send() → NPU rank
  NPU rank → HccdImplPml::Recv() → CPU rank
```

---

## 6. 关键业务逻辑

### CommId 生成
`HccdGenerateCommId()` 的实现路径：
1. 调用 `HccdComm::GetUniqueId()` 获取唯一基础 ID
2. 调用 `snprintf` 将 uniqueId + `"-"` + `"hccl_heterog_group"` 拼接为 CommId 字符串
3. CommId 是 128 字节的结构体（`HcclRootHandle`），确保在分布式环境中全局唯一

### HccdComm 初始化保护
`HccdComm::init()` 使用原子操作保护：
- `AtomicInitSet()` 设置初始化标志
- 调用 `impl_->Init()` 执行实际初始化
- 若失败，调用 `AtomicInitClear()` 回滚标志，保证线程安全

### HCCD 模式 vs 非 HCCD 模式
- **HCCD 模式**：有独立的 HCCD 守护进程，负责协调；适合大规模集群（rank 数多）
- **非 HCCD 模式**：rank 0 直接作为协调者；适合小规模（rank 数少）

### 异构通信（PML）
`hccd_impl_pml.cc` 支持 CPU rank 参与集合通信（`isHaveCpuRank_=true` 场景），CPU rank 通过 PML 接口与 NPU rank 交换数据，不需要 NPU 硬件。

---

## 7. 开发注意事项

- HCCD 进程必须在所有 rank 进程启动前运行，否则 `HcclInitComm` 连接 HCCD 会超时。
- `HccdGenerateCommId()` 生成的 CommId 必须通过带外（out-of-band）方式分发给所有 rank（如环境变量、共享文件），hcomm 本身不提供分发机制。
- PML 接口（`hccd_impl_pml.cc`）是实验性功能，生产环境中 CPU rank 参与通信的场景较少，使用前需充分测试。

---

## 8. 未来可扩展点

- **高可用 HCCD**：当前 HCCD 是单点，可扩展为主备模式，避免 HCCD 宕机导致整个集群通信失败。
- **CommId 服务化**：将 CommId 生成和分发封装为独立的微服务，支持 REST API 调用。
- **异构扩展**：PML 接口可扩展支持 GPU（CUDA）rank 与 NPU rank 的混合通信。
