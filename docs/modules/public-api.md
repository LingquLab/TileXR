# 模块：public-api（公共 API 与类型定义）

## 1. 模块作用

定义 TileXR 对外暴露的所有 C/C++ 接口、类型枚举、常量、同步原语和通信参数结构体。是用户代码与底层实现之间的契约层，也是 kernel 侧与 host 侧共享数据结构的唯一来源。

---

## 2. 目录结构

```
src/include/
├── tilexr_api.h       # C 公共 API 函数声明
├── tilexr_types.h     # 枚举、常量、类型定义
├── tilexr_sync.h      # AICore 侧同步原语（SyncCollectives 类）
└── comm_args.h        # CommArgs 结构体（host/device 共享数据）
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `tilexr_api.h` | 对外 C API，用 `extern "C"` 包装，通过 `comm_wrap.cpp` 实现 |
| `tilexr_types.h` | `ChipName`、`PhysicalLink`、`TileXRType`、buffer 尺寸常量、错误码 |
| `tilexr_sync.h` | `SyncCollectives` 类：kernel 侧 flag 同步原语，两类 flag 区（inner/outer） |
| `comm_args.h` | `CommArgs` 结构体：rank 信息、peer 内存指针、发送矩阵、DFX、dump 地址 |

---

## 4. 核心函数 / 类 / 接口

### `tilexr_api.h` — C API

```c
// Comm 对象生命周期
TileXRCommPtr TileXRCommInitRankLocal(int rank, int rankSize);
TileXRCommPtr TileXRCommInitRank(int rank, int rankSize, TileXRUniqueId id);
TileXRCommPtr TileXRCommInitRankWithCustDomainSize(int rank, int rankSize, int domain, int bufSize);
TileXRCommPtr TileXRCommInitRankWithDomain(int rank, int rankSize, int domain);
void          LcclCommDestroy(TileXRCommPtr comm);

// 参数获取
int TileXRGetCommArgsDev(TileXRCommPtr comm, GM_ADDR *argsAddr); // 设备侧地址
int TileXRGetCommArgsHost(TileXRCommPtr comm, CommArgs *args);   // host 侧结构体拷贝

// 工具
TileXRUniqueId TileXRGetUniqueId();
int TileXRCommInit(int rank, int rankSize, TileXRCommPtr *comm);
int TileXRCommInitAll(int rankSize, TileXRCommPtr *comms);       // 进程内多 rank
int TileXRCommInitThread(int rank, int rankSize, TileXRCommPtr *comm);
```

### `tilexr_types.h` — 关键枚举与常量

```cpp
// 错误码
TILEXR_SUCCESS = 0
TILEXR_ERROR_NOT_INITIALIZED = -1  TILEXR_ERROR_TIMEOUT = -5

// 内存常量
TILEXR_BUFF_BYTES       = 204 * 1024 * 1024   // 每 rank 数据 buffer
TILEXR_FLAG_BUFF_BYTES  = 4   * 1024 * 1024   // 每 rank flag buffer
TILEXR_COMM_BUFFER_SIZE = 200                 // MB 单位，配置值

// 芯片枚举（ChipName）
CHIP_TYPE_ASCEND_910B, 910B2C, 910A5, 310P3 等

// 互联类型（PhysicalLink）
HCCS, PCIE

// 算子类型（TileXRType，70+ 条目）
ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_GATHER_MATMUL, MATMUL_ALL_REDUCE ...
```

### `comm_args.h` — 核心结构体

```cpp
// CommArgs：host/device 共享，通过 H2D 拷贝后 kernel 可直接读取
struct CommArgs {
    int32_t rank;
    int32_t localRank;
    int32_t rankSize;
    int32_t localRankSize;
    ExtraFlag extraFlag;         // 特性标志位（RDMA/QUANT/TOPO 等）
    GM_ADDR peerMems[TILEXR_MAX_RANK_SIZE];   // 各 rank 的 IPC 内存地址
    int64_t sendCountMatrix[...];             // 发送矩阵（AllToAll 等）
    LcclDumpLogInfo dfx;                      // DFX 诊断信息
    GM_ADDR dumpAddr;            // profiling dump 地址
    uint32_t magics[...];        // 同步魔数
    uint64_t fftsVal;            // FFTS 值
};

// 特性标志（ExtraFlag 枚举）
RDMA, TOPO_HCCS, TOPO_PCIE, QUANT_INT8, ATOMIC_ENABLE ...

// 操作类型（Op 枚举）
COPYONLY, ADD, MUL, MAX, MIN
```

### `tilexr_sync.h` — AICore 同步原语

```cpp
class SyncCollectives {                   // __aicore__ 类，在 kernel 内使用
    void Init(int rank, int rankSize,
              __gm__ uint8_t* innerFlagAddr,  // 片内同步区（2MB）
              __gm__ uint8_t* outerFlagAddr); // 片间同步区（2MB）

    // flag 写入与等待（支持多轮重用，magic 防止 ABA）
    void SetSyncFlag(int dstRank, uint64_t value, uint32_t magic);
    void WaitSyncFlag(int srcRank, uint64_t value, uint32_t magic);
    void WaitSyncGreaterFlag(int srcRank, uint64_t threshold);

    // 多 block 场景下 event ID 计算
    uint32_t CalEventIdByMulBlockNum(int rank, int blockNum);
};
```

---

## 5. 数据流向

```
host 初始化
  └── TileXRComm::Init()
        └── 填充 CommArgs（rank、peerMems、magics ...）
              └── aclrtMemcpy H2D
                    └── GPU 侧 CommArgs*（GM 地址）

kernel 启动
  └── 从 kernel args 接收 CommArgs* 指针
        ├── 读取 peerMems[i]（访问 peer 设备内存）
        ├── 使用 SyncCollectives（读写 flag buffer）
        └── 根据 ExtraFlag 决定路径（RDMA / IPC / QUANT ...）
```

---

## 6. 关键业务逻辑

### flag 编码机制（`tilexr_sync.h`）
flag 值编码为 `(magic << 32) | value`，同一 flag 地址在多轮通信中不需要清零，通过 magic 区分轮次，避免 ABA 问题。

### ExtraFlag 位图
`CommArgs.extraFlag` 是一个枚举位掩码，kernel 根据该字段决定走哪条执行路径（如是否使用 RDMA、是否做量化、拓扑类型等）。修改此字段时需同时更新 kernel 侧的判断分支。

### TILEXR_MAX_RANK_SIZE = 128
`peerMems` 数组固定大小为 128，超过此 rank 数量的场景不支持。

---

## 7. 开发注意事项

- `CommArgs` 是 host 和 device 的共享结构，字段布局必须严格对齐，不可随意增删字段（会破坏二进制兼容性）。
- `tilexr_sync.h` 中的类是 `__aicore__` 修饰的 AICore 类，只能在 kernel 侧调用，不可在 host 代码中实例化。
- `TileXRType` 枚举值在 tiling 选择和 kernel dispatch 中作为键值使用，新增算子时需同步添加枚举项。
- `ExtraFlag` 修改需要全链路评估（host 端 `Init()`、`comm_wrap.cpp`、各 kernel `.cpp`）。

---

## 8. 未来可扩展点

- **`CommArgs` 版本字段**：增加一个 `version` 字段，支持滚动升级时 host/device 不同版本的兼容。
- **动态 rank 数**：目前 `peerMems[128]` 静态分配，可改为在 `Init()` 时按实际 rankSize 动态分配。
- **更多 ExtraFlag**：当前 flag 为枚举，可改为 `uint64_t` 位域以支持更多特性组合。
- **Python 绑定**：`TileXRUniqueId` / `CommArgs` 可通过 pybind11 暴露给 Python 侧使用。
