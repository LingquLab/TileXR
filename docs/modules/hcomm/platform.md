# 模块：hcomm/platform（平台抽象层）

## 1. 模块作用

对 Ascend 硬件驱动、运行时（RTS）、HCCP、ibverbs、TDT、Trace 等底层系统接口进行统一封装，向上层提供与硬件无关的适配器接口。是 hcomm 与 CANN 驱动/运行时之间的隔离层，也是多芯片（910B/91093/310P）差异化处理的集中地。

---

## 2. 目录结构

```
src/platform/
├── common/
│   ├── adapter/                   # 各子系统适配器实现
│   │   ├── adapter_hal.cc/.h      # HAL（硬件抽象层）适配
│   │   ├── adapter_rts.cc/.h      # RTS（运行时系统）适配
│   │   ├── adapter_hccp.cc/.h     # HCCP（片间通信协议）适配
│   │   ├── adapter_verbs.cc/.h    # ibverbs RDMA 适配
│   │   ├── adapter_tdt.cc/.h      # TDT（数据传输）适配
│   │   ├── adapter_trace.cc/.h    # Trace/profiling 适配
│   │   └── adapter_error_manager.cc/.h  # 错误管理适配
│   ├── buffer_manager/            # RMA buffer 管理
│   │   ├── rma_buffer_mgr.h       # RMA buffer 管理器接口
│   │   └── buffer_key.h           # buffer 键值定义
│   ├── device_capacity.cc/.h      # 设备能力查询（核数、内存等）
│   ├── dlhal_function.cc          # 动态加载 HAL 函数
│   ├── dlibv_function.cc          # 动态加载 ibverbs 函数
│   ├── dlhns_function.cc          # 动态加载 HNS 函数
│   ├── dlra_function.cc           # 动态加载 RA 函数
│   ├── dlprof_func.cc             # 动态加载 profiling 函数
│   ├── dltdt_function.cc          # 动态加载 TDT 函数
│   └── dltrace_function.cc        # 动态加载 Trace 函数
└── （设备特化目录，按芯片型号组织）
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `adapter_hal.cc` | HAL 适配：`hrtGetgrpId()`（获取 group/device ID）、内存预取线程（24 线程，256MB 阈值）、`HcclSetGrpIdCallback()` 回调注册 |
| `adapter_rts.cc` | RTS 适配：stream 创建/销毁、event/notify 管理、内存分配（`hrtMalloc`/`hrtFree`）、kernel launch |
| `adapter_verbs.cc` | ibverbs 适配：QP 创建/销毁、MR 注册、RDMA 发送/接收、CQ 轮询 |
| `adapter_hccp.cc` | HCCP 适配：片间通信协议初始化、HCCS 链路管理 |
| `device_capacity.cc` | 设备能力查询：AICore 数量、L2 cache 大小、支持的通信协议 |
| `dlibv_function.cc` | 动态加载 libibverbs.so，避免编译期强依赖 ibverbs 库 |
| `rma_buffer_mgr.h` | RMA buffer 管理器：统一管理 IPC 和 RDMA 两种内存访问路径的 buffer 生命周期 |

---

## 4. 核心函数 / 类 / 接口

### HAL 适配（`adapter_hal.cc`）

```cpp
// 常量
// PRE_FETCH_THREADS_NUM      = 24
// PRE_FETCH_MEMORY_THRESHOLD = 268435456 (256 MB)
// MEMORY_PAGE_SIZE           = 4096

// 回调注册
HcclResult HcclSetGrpIdCallback(int (*grpIdCallback)(int tag, int* grpId, int* devId));

// HAL 事件与设备管理
HcclResult hrtHalSubmitEvent(u32 devId, u32 eventId, u32 groupId);
HcclResult hrtHalEschedAttachDevice(unsigned int devId);
HcclResult hrtHalEschedCreateGrp(unsigned int devId, unsigned int grpId, GROUP_TYPE type);
HcclResult hrtHalGetDeviceType(const uint32_t devId, DevType& devType);

// 内存预取（后台线程，减少首次 page fault）
HcclResult MemoryPreFetch(u64 size, void* hostPtr);

// Host 内存注册（用于 RDMA 零拷贝）
HcclResult hrtHalHostRegister(void* hostPtr, u64 size, u32 flag, u32 devid, void*& devPtr);
HcclResult hrtHalHostUnregister(void* hostPtr, u32 devid);

// 带宽查询（level=0 片内, 1 跨机, 2 超节点间, 3 HBM）
HcclResult GetBandWidthPerNPU(u32 level, u32 userRankSize,
    u32 deviceNumPerAggregation, float& bandWidth);

// MC2 维护线程
HcclResult hrtHalStartMC2MaintenanceThread(mc2Funcs f1, void* p1, mc2Funcs f2, void* p2);

// 资源 ID 恢复（重试场景）
HcclResult hrtHalResourceIdRestore(u32 devId, u32 tsId,
    drvIdType_t resType, u32 resId, u32 flag);
```

### RTS 适配（`adapter_rts.cc`）

```cpp
// Stream 管理
HcclResult hrtStreamCreate(HcclRtStream* stream);
HcclResult hrtStreamDestroy(HcclRtStream stream);
HcclResult hrtStreamSynchronize(HcclRtStream stream);

// 内存管理
HcclResult hrtMalloc(void** ptr, size_t size, HcclRtMemType type);
HcclResult hrtFree(void* ptr);
HcclResult hrtMemcpy(void* dst, const void* src, size_t size, HcclRtMemcpyKind kind);

// Kernel launch
HcclResult hrtKernelLaunch(const char* funcName, dim3 blockDim,
    void** args, HcclRtStream stream);
```

### ibverbs 适配（`adapter_verbs.cc`）

```cpp
// NIC 设备生命周期
HcclResult HcclNetOpenDev(HcclNetDevCtx* ctx, NicType nicType,
    u32 devicePhyId, u32 deviceLogicId, HcclIpAddress ip);
HcclResult HcclNetCloseDev(HcclNetDevCtx ctx);
HcclResult HcclNetDevGetNicType(HcclNetDevCtx ctx, SocketType* type);

// ibverbs 底层操作（封装 libibverbs 动态加载函数）
HcclResult hrtIbvPostSrqRecv(struct ibv_srq* srq,
    struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr);
HcclResult hrtIbvPostRecv(struct ibv_qp* qp,
    struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr);
HcclResult hrtIbvPostSend(struct ibv_qp* qp,
    struct ibv_send_wr* wr, struct ibv_send_wr** badWr);
HcclResult hrtIbvPollCq(struct ibv_cq* cq, int maxNum,
    struct ibv_wc* wc, s32& num);
HcclResult hrtIbvReqNotifyCq(struct ibv_cq* cq, int solicitedOnly);
HcclResult hrtIbvGetCqEvent(struct ibv_comp_channel* channel,
    struct ibv_cq** cq, void** cq_context);
void       hrtIbvAckCqEvent(struct ibv_cq* qp, unsigned int nevents);
HcclResult hrtIbvQueryQp(struct ibv_qp* qp);
// HNS 扩展发送（带 expRsp 响应）
HcclResult HrtHnsIbvExpPostSend(struct ibv_qp* qp,
    struct ibv_send_wr* wr, struct ibv_send_wr** badWr, struct WrExpRsp* expRsp);
```

### 设备能力查询（`device_capacity.cc`）

```cpp
// 带宽常量（GB/s，有效利用率 = 标称 × 效率因子）
// BANDWIDTH_HCCS_910A = 10.0f,  BANDWIDTH_HCCS_910B = 18.3f
// BANDWIDTH_RDMA_910A = 10.0f,  BANDWIDTH_RDMA_910B = 20.0f
// BANDWIDTH_HBM_910_93 = 585.0f (650 × 0.9)
// BANDWIDTH_SIO_910_93 = 204.0f (240 × 0.85)
// BANDWIDTH_PCIE_GEN4  = 27.2f, BANDWIDTH_PCIE_GEN5 = 54.4f

// AIV 能力检查
bool IsSupportAIVCopy(HcclDataType dataType);       // FP16/INT16/FP32/INT8/BFP16 等
bool IsSupportAIVReduce(HcclDataType dataType, HcclReduceOp op); // SUM/MAX/MIN

// 地址对齐检查（按芯片类型差异化）
bool IsAddressAlign(const void* inputPtr, const void* outputPtr, DevType devType);
// 数据类型与 Reduce 操作支持（按芯片查表）
bool IsDataTypeSupport(HcclDataType dataType, DevType devType);
bool IsRedOpSupport(HcclReduceOp op, DevType devType);

// SDMA/RDMA reduce 能力组合检查
bool IsSupportSDMAReduce(const void* inputPtr, const void* outputPtr,
    HcclDataType dataType, HcclReduceOp op);
bool IsSupportRDMAReduce(HcclDataType dataType, HcclReduceOp op);

// 硬件能力查询
HcclResult GetBandWidthPerNPU(u32 level, u32 userRankSize,
    u32 deviceNumPerAggregation, float& bandWidth);   // 带宽查找表（level 0-3）
HcclResult GetMaxDevNum(u32& MaxDevNum);              // 310P3=32, 其他=16（带缓存）
HcclResult IsSupportAtomicWrite(DevType deviceType,
    u32 devicePhyId, bool& isSupportAtomicWrite);     // 仅 910_93 支持

// 通知超时（设备类型相关）
u32 GetNotifyMaxWaitTime();
```

---

## 5. 数据流向

```
上层（framework / algorithm）
  └── 调用 adapter_*.h 中的接口
        │
        ▼
adapter_*.cc（适配器实现）
  ├── 直接调用 CANN RTS API（hrtXxx）
  ├── 通过 dlXxx_function.cc 动态加载的函数指针调用 ibverbs/HNS
  └── 通过 device_capacity.cc 查询硬件能力
        │
        ▼
CANN 驱动 / libibverbs.so / 硬件
```

---

## 6. 关键业务逻辑

### 动态加载策略
所有外部库（ibverbs、HNS、RA、TDT、Trace）均通过 `dlopen`/`dlsym` 动态加载，而非编译期链接。好处：
1. 在没有 ibverbs 的环境（纯 HCCS 机器）也能正常启动
2. 支持运行时切换不同版本的底层库

### 内存预取优化
`adapter_hal.cc` 启动 24 个后台线程对超过 256MB 的内存区域做预取，减少首次访问的 page fault 延迟，对大 buffer 的 AllGather/AllReduce 有明显收益。

### 设备能力缓存
`device_capacity.cc` 在首次查询时缓存设备能力（AICore 数、L2 大小等），避免重复调用驱动接口。

---

## 7. 开发注意事项

- 所有 `hrtXxx` 调用必须检查返回值，使用 `CHK_RET` 宏，不可忽略错误。
- 动态加载函数指针在使用前必须检查是否为 `nullptr`（库未安装时为 null）。
- `adapter_verbs.cc` 中的 QP 操作是线程不安全的，调用方需保证串行访问或加锁。
- 新增芯片支持时，在 `device_capacity.cc` 中添加对应的能力查询分支。

---

## 8. 未来可扩展点

- **新互联协议**：增加 `adapter_xxx.cc` 适配新的互联协议（如 NVLink 兼容层）。
- **统一内存接口**：将 `hrtMalloc`/`hrtFree` 与 ibverbs MR 注册合并为统一的 `HcclMemAlloc` 接口，简化上层调用。
- **能力热更新**：`device_capacity.cc` 目前是静态缓存，可扩展为支持热插拔设备的动态更新。
