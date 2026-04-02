# 模块：hcomm/common（公共基础设施）

## 1. 模块作用

提供 hcomm 所有层共享的横切关注点实现：profiling/tracing、调试日志、错误码、流工具、健康检查（CRC、rank 一致性）、设备 launch 工具。这些功能不属于任何单一业务层，但被 framework、algorithm、platform 三层普遍依赖。

---

## 2. 目录结构

```
src/common/
├── debug/
│   ├── config/
│   │   ├── config_log.cc/.h       # 日志配置（级别、输出目标）
│   ├── profiling/
│   │   ├── profiling_manager.cc/.h    # Profiling 管理器（单例）
│   │   ├── profiler_manager.cc/.h     # Profiler 实例管理
│   │   ├── adapter_prof.cc            # Msprof 适配
│   │   ├── command_handle.cc          # Profiling 命令处理
│   │   ├── task_profiling.cc/.h       # 任务级 profiling
│   │   ├── task_exception_handler.cc  # 任务异常处理
│   │   ├── task_overflow.cc           # 任务溢出检测
│   │   ├── plugin_runner.cc           # Profiling 插件运行器
│   │   └── inc/
│   │       ├── profiler_base_pub.h    # Profiler 公共基类
│   │       ├── profiling_manager_pub.h
│   │       └── task_profiling_pub.h
├── error_code/
│   └── error_code.cc              # 错误码定义与字符串映射
├── health/
│   ├── calc_crc.cc/.h             # CRC 校验（rank 数据完整性）
│   └── rank_consistentcy_checker.cc/.h  # Rank 一致性检查
├── launch_device/
│   └── launch_device.cc/.h        # 设备端 launch 工具函数
└── stream/
    ├── stream_utils.cc/.h         # Stream 工具（同步、等待、状态查询）
    └── stream_utils_aicpu.cc      # AICPU 模式下的 stream 工具
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `debug/profiling/profiling_manager.cc` | 单例 ProfilingManager：管理 Msprof 上报队列（每设备独立）、FFTS launch 注册、task API 存储 |
| `debug/profiling/task_profiling.cc` | 任务粒度的 profiling：记录每个集合通信算子的开始/结束时间戳、数据量 |
| `health/rank_consistentcy_checker.cc` | 检查所有 rank 的算法选择、tiling 参数是否一致，防止因 rank 间不一致导致死锁 |
| `health/calc_crc.cc` | CRC32 计算，用于 tiling 参数的完整性校验 |
| `stream/stream_utils.cc` | `WaitStream()`、`QueryStreamStatus()`、`AddStreamDependency()` 等 stream 操作封装 |
| `error_code/error_code.cc` | 所有 `HCCL_E_*` 错误码的字符串描述，用于日志输出 |

---

## 4. 核心函数 / 类 / 接口

### ProfilingManager（`profiling_manager.cc`）

```cpp
class ProfilingManager {
public:
    static ProfilingManager& Instance();  // 单例访问

    // 上报到 Msprof（华为性能分析工具）
    HcclResult ReportToMsprof(u32 deviceId, const ProfilingData& data);

    // 注册 FFTS launch 回调
    void RegisterFftsLaunch(u32 deviceId, FftsLaunchCallback cb);

    // Task API 存储队列（每设备）
    std::vector<TaskApiInfo> taskApiQueue_[HCCL_MAX_DEVICE_NUM];
    std::vector<CompactInfo> compactInfoQueue_[HCCL_MAX_DEVICE_NUM];
};
```

### Rank 一致性检查（`rank_consistentcy_checker.cc`）

```cpp
class RankConsistencyChecker {
    // 在所有 rank 间广播并比较关键参数（算法类型、tiling 参数的 CRC）
    HcclResult CheckConsistency(const std::string& tag,
        const AlgType& algType,
        const void* tilingData, size_t tilingSize);
};
```

### Stream 工具（`stream_utils.cc`）

```cpp
// 等待 stream 上所有任务完成
HcclResult WaitStream(HcclRtStream stream, u32 timeoutMs = 0);

// 查询 stream 当前状态（IDLE / RUNNING / ERROR）
HcclResult QueryStreamStatus(HcclRtStream stream, StreamStatus* status);

// 在两个 stream 之间建立依赖（stream B 等待 stream A 的某个 event）
HcclResult AddStreamDependency(HcclRtStream streamA, HcclRtStream streamB);
```

---

## 5. 数据流向

```
集合通信算子执行
  │
  ├── ProfilingManager::Instance().ReportToMsprof()  ← task_profiling.cc
  │       记录算子时间戳、数据量到 Msprof 队列
  │
  ├── RankConsistencyChecker::CheckConsistency()     ← rank_consistentcy_checker.cc
  │       广播 CRC → 所有 rank 比较 → 不一致则报错
  │
  └── StreamUtils::WaitStream()                      ← stream_utils.cc
          等待通信 stream 完成后再执行计算 stream
```

---

## 6. 关键业务逻辑

### Profiling 上报路径
`ProfilingManager` 维护每设备的独立上报队列，避免多设备并发写冲突。Msprof 上报是异步的（放入队列后立即返回），由后台线程批量发送到 Msprof daemon。

### Rank 一致性保护
在算子执行前，`RankConsistencyChecker` 通过已有的通信通道广播 tiling 参数的 CRC。若任一 rank 的 CRC 与 rank 0 不同，立即报错并中止，防止因 tiling 不一致导致的挂起（hang）。

### CRC 在健康检查中的应用
`calc_crc.cc` 的 CRC32 也用于心跳数据的完整性验证（见 `framework/cluster_maintenance/health/`），确保心跳包未被损坏。

---

## 7. 开发注意事项

- `ProfilingManager` 是单例，不可在析构后再调用（进程退出时注意析构顺序）。
- `RankConsistencyChecker` 的一致性检查依赖通信通道，必须在 `Init()` 完成后才能调用。
- `stream_utils_aicpu.cc` 是 AICPU 模式的特化版本，与普通模式的 stream 语义不同，不可混用。
- 错误码（`error_code.cc`）应通过 `HCCL_ERROR_CODE(err)` 宏格式化输出，不要直接打印数值。

---

## 8. 未来可扩展点

- **结构化日志**：将当前的字符串日志改为结构化 JSON 格式，便于日志聚合分析。
- **Profiling 插件化**：`plugin_runner.cc` 已有框架，可扩展接入第三方 profiling 工具（如 nsight）。
- **异步一致性检查**：当前一致性检查是同步阻塞的，可改为异步模式，减少算子启动延迟。
