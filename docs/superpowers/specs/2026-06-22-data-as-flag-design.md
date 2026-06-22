# DataAsFlag 通用传输接口设计

日期：2026-06-22

## 背景

DataAsFlag 将数据与到达标记合并在同一段传输布局中，减少额外 flag 写入步骤。第一版面向 TileXR 集合通信内核中的 GM 到 GM 数据传输，不引入新的 host 生命周期或通信资源，只提供设备侧 header-only 工具接口。

参考 PPT 与 `dispatch_v2_fullmesh_0608.h` 中的实现后，第一版固定采用 512B block 协议：

```text
每个 DataAsFlag block:
[0, 480)     payload，有效数据区
[480, 512)   flag 区，其中首个 float 固定为 1.0f
```

调用方用普通 byte 数描述有效数据长度，接口内部根据 480B payload 自动计算所需 512B block 数。

## 目标

提供通用 DataAsFlag 设备侧接口，包含：

- 初始化：预置 UB scratch 中每个 block 的 flag 值。
- 发送：从连续源 GM 读取普通数据，打包写入目标 GM 的 DataAsFlag 512B 布局。
- 校验：从 DataAsFlag GM 布局读取每个 block 的 flag，通过 `Sum` 判断是否全部到达。
- 校验并接收：校验通过后，将 512B 布局中的 payload 解包为连续接收 GM 数据。

## 非目标

- 不提供 host API 或新增 comm 初始化资源。
- 不支持自定义 flag 值，第一版固定 `float(1.0f)`。
- 不以 token 数描述数据长度。
- `DataAsFlagCheck` 不提供阻塞等待语义，只做一次校验并返回 bool。
- `DataAsFlagCheckAndRecv` 提供阻塞式分片接收语义：每片数据校验通过后立即 copy 到 `recvGM`，再进入下一片，直到 `dataBytes` 对应的数据全部完成校验与 copy。
- 不保证最后一个 block 中超出 `dataBytes` 的 payload padding 内容。

## 公共头文件

新增 `src/include/tilexr_data_as_flag.h`，namespace 使用 `TileXR`。该头文件与 `tilexr_sdma.h`、`tilexr_udma.h` 类似，作为可安装的 header-only 设备侧工具。

核心常量：

```cpp
constexpr uint32_t DATA_AS_FLAG_BLOCK_BYTES = 512;
constexpr uint32_t DATA_AS_FLAG_PAYLOAD_BYTES = 480;
constexpr uint32_t DATA_AS_FLAG_FLAG_BYTES = 32;
constexpr uint32_t DATA_AS_FLAG_FLAG_OFFSET_BYTES = 480;
constexpr float DATA_AS_FLAG_READY_VALUE = 1.0f;
```

设备侧接口：

```cpp
__aicore__ inline uint32_t DataAsFlagInit(
    AscendC::LocalTensor<uint8_t>& sendScratch);

__aicore__ inline uint32_t DataAsFlagSend(
    __gm__ uint8_t* dstDataAsFlagGM,
    const __gm__ uint8_t* srcGM,
    uint64_t dataBytes,
    AscendC::LocalTensor<uint8_t>& sendScratch);

__aicore__ inline bool DataAsFlagCheck(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    AscendC::LocalTensor<uint8_t>& recvScratch);

__aicore__ inline bool DataAsFlagCheckAndRecv(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    __gm__ uint8_t* recvGM,
    AscendC::LocalTensor<uint8_t>& recvScratch);
```

## 接口语义

### DataAsFlagInit

`DataAsFlagInit` 从 `sendScratch.GetSize()` 计算可容纳的 512B block 数：

```cpp
sendBlocks = sendScratch.GetSize() / DATA_AS_FLAG_BLOCK_BYTES;
```

容量不足 512B 时返回 0。容量足够时，接口把 `sendScratch` 转成 `LocalTensor<float>`，用 `Duplicate<float>` 将 scratch 初始化为 `1.0f`。之后 `DataAsFlagSend` 只覆盖每个 block 的前 480B payload，因此 flag 区首个 float 保持为 `1.0f`。

调用方必须在复用 `sendScratch` 发送前调用该初始化接口；如果 scratch 被其他逻辑覆盖，需要重新初始化。

### DataAsFlagSend

`DataAsFlagSend` 做 GM 到 GM 打包发送：

```cpp
totalBlocks = ceil(dataBytes / DATA_AS_FLAG_PAYLOAD_BYTES);
```

接口按 `sendScratch` 容量分批：

1. 从 `srcGM` 连续读取有效数据。
2. 将每个 480B payload 写入 `sendScratch` 中对应 512B block 的前 480B。
3. 将 `sendScratch` 中完整的 `batchBlocks * 512B` 写入 `dstDataAsFlagGM`。
4. 最后一个 block 只读取 `dataBytes` 范围内有效字节。

返回实际写出的 512B block 数。输入指针为空、`dataBytes == 0` 或 scratch 容量不足时返回 0。

### DataAsFlagCheck

`DataAsFlagCheck` 使用一个 `recvScratch` 完成 flag 批量读取与 Sum 归约。Sum 采用显式 `sharedTmpBuffer` 的重载，避免依赖框架隐式临时空间。内部将 scratch 划分为：

```text
recvScratch:
[flag 接收区][sum 区]
```

其中 sum 区不是单纯 32B 结果输出区，而是包含：

- 32B 对齐的 Sum 结果输出区。
- Sum 内部计算所需的临时 workspace。

Sum workspace 大小按本批 `batchBlocks` 对应的 `SumParams` 计算；`SumParams.inner * sizeof(float)` 必须 32B 对齐。实现阶段使用目标 CANN 提供的 Sum Tiling 能力确认 `sharedTmpBuffer` 所需大小，并在 `recvScratch` 中显式切出该空间。

校验流程：

1. 根据 `dataBytes` 计算 `totalBlocks`。
2. 从 `dataAsFlagGM + DATA_AS_FLAG_FLAG_OFFSET_BYTES` 起，按 512B stride 读取每个 block 的首个 `float` flag。
3. 将 flag 放入 `recvScratch` 前段，并在内部 `ReinterpretCast<float>()`。
4. 对本批 flag 使用带 `sharedTmpBuffer` 的 `Sum`。
5. Sum 结果等于本批 block 数时继续，否则返回 `false`。

全部批次通过时返回 `true`。输入为空或 `recvScratch` 无法同时容纳 1 个 flag、Sum 结果输出区与 Sum workspace 时返回 `false`。

### DataAsFlagCheckAndRecv

`DataAsFlagCheckAndRecv` 是阻塞式接口。它不会先等待全部 `dataBytes` 对应的 block 都到齐后再统一接收，而是按 `recvScratch` 容量把数据划分成多个片段，对每个片段执行：

1. `while` 循环校验本片段的 flag。
2. 本片段所有 flag 的 Sum 结果等于本片段 block 数后，认为该片段完整到达。
3. 立即将该片段中每个 512B block 的前 480B payload copy 到 `recvGM` 的连续地址。
4. 进入下一片段，直到 `dataBytes` 对应的数据全部完成校验并 copy 到 `recvGM`。

`recvScratch` 在该接口内划分为三段，顺序固定为：

```text
recvScratch:
[flag 接收区][sum 区][copy 到 recvGM 的 UB 区]
```

其中 sum 区包含 32B 对齐的 Sum 结果输出区与 Sum workspace。三段大小的依据是“单轮能够 copy 到 `recvGM` 的有效数据总量”，即先确定本轮 `batchBlocks` 个 block 对应的 copy 数据量，再反推本轮需要的 flag 与 Sum 空间：

```cpp
copyBytes = batchBlocks * DATA_AS_FLAG_PAYLOAD_BYTES;
flagBytes = AlignUp(batchBlocks * sizeof(float), 32);
sumBytes = AlignUp(sizeof(float), 32) + AlignUp(SumWorkspaceBytes(batchBlocks), 32);
requiredBytes = flagBytes + sumBytes + copyBytes;
```

`batchBlocks` 取 `recvScratch` 能同时容纳上述三段的最大 block 数；本轮完成的有效接收数据量就是 `copyBytes`，最后一轮按剩余 `dataBytes` 裁剪尾 block 的有效字节数。若 `recvScratch` 无法容纳 1 个 block 对应的三段空间，即小于 `32B flag 对齐区 + Sum 结果输出区 + Sum workspace + 480B copy 区`，接口返回 `false`。

成功写完 `dataBytes` 后返回 `true`。输入为空、`recvGM` 为空、`dataBytes == 0` 或 scratch 容量不足时返回 `false`。

## AscendC API 约束

- 外部统一传入 `LocalTensor<uint8_t>` scratch，接口内部按需 `ReinterpretCast<float>()`。
- 数据搬运使用 `DataCopyPad`，用于处理最后一批和尾 block 的非 480B 数据。
- 不使用生产路径低效的 `GlobalTensor::GetValue()` 或 `GlobalTensor::SetValue()`。
- Sum 使用显式 `sharedTmpBuffer` 重载，Sum 区包含结果输出与 workspace；workspace 大小按本批 `SumParams` 通过 Sum Tiling 规则计算。
- `DataAsFlagCheck` 内部将 `recvScratch` 划分为 `[flag 接收区][sum 区]`。
- `DataAsFlagCheckAndRecv` 内部将 `recvScratch` 划分为 `[flag 接收区][sum 区][copy 到 recvGM 的 UB 区]`，三段大小依据单轮 copy 到 `recvGM` 的有效数据量确定。
- stride 计算遵守 AscendC 约定：GM stride 单位为字节，UB stride 单位为 32B data block。

## 安装与测试

需要更新：

- `src/comm/CMakeLists.txt`：安装 `tilexr_data_as_flag.h`。
- `tests/collectives/CMakeLists.txt`：加入 DataAsFlag header/source guard 测试。

建议测试：

- Header 编译测试：include `tilexr_data_as_flag.h` 并检查常量值。
- 源码结构测试：确认接口名存在、使用 `DataCopyPad` 与 `Sum`、不使用 `GlobalTensor::GetValue/SetValue`。
- 远端代码验证：在 `root@141.62.19.152:/home/h00580772` 下创建独立验证目录。传输文件时先去除后缀名，传输完成后在服务器上恢复后缀名，再执行编译或静态验证。

## 完成标准

- 公共头文件可被集合通信 kernel 直接 include。
- `DataAsFlagSend` 按 byte 数完成 480B payload 到 512B block 布局的打包。
- `DataAsFlagCheck` 通过 `Sum` 判断所有 flag 是否为 `1.0f`。
- `DataAsFlagCheckAndRecv` 阻塞式分片轮询校验，每片校验通过后立即 copy 到接收 GM，直到全部有效数据连续输出完成。
- 新头文件随 `tile-comm` 安装。
- 本地静态测试和远端验证通过。
