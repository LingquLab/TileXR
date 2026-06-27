# UDMA P2P 传输方式对比

**主题：** `memory_consume` 与 `data_as_flag_epoch_ordered`

## 摘要

`data_as_flag_epoch_ordered` 已经消除了最严重的按块写 flag 瓶颈，但其 payload 路径仍比 `memory_consume` 重得多。在相同的有效 payload 带宽统计口径下，`~10 GB/s` 对 `~50 GB/s` 这样的结果与当前实现是相符的。

总体来说：

```text
memory_consume:
  连续 payload 传输 + 独立同步 flag

data_as_flag_epoch_ordered:
  data-as-flag 协议校验，提交信息内嵌
  在 480B payload + 32B flag/gap 的块布局中
```

`memory_consume` 更接近高效的连续 payload 传输。`data_as_flag_epoch_ordered` 更接近一条协议路径——通过在数据窗口中内嵌就绪信息来证明数据已就绪。

## `memory_consume` 数据路径

`memory_consume` kernel 保持 payload 窗口连续：

```text
发送端:
  连续 src payload
    -> 对端 IPC 数据窗口
  设置独立的外部同步 flag

接收端:
  等待独立的外部同步 flag
  连续的本地对端 IPC 数据窗口
    -> 连续的 dst payload
```

在代码中，发送端从 `srcGM + offset` 拷贝到 `peerBase + dstByteOffset + offset`，然后调用 `sync.SetOuterFlag(magic, step)`。接收端用 `sync.WaitOuterFlag(...)` 等待，再从 `localBase + dstByteOffset + offset` 拷贝到 `dst + offset`。

两端的 payload 布局都保持连续。虽然 helper 内部仍会经过 UB 中转，但拷贝的 payload 是简单的连续 GM 到 GM 流。对于大消息，独立同步 flag 的开销会被大 payload 摊薄，因此这条路径可以达到高得多的带宽。

相关文件：

```text
tests/udma/demo/tilexr_udma_demo_kernel.cpp
  tilexr_memory_consume_p2p_perf_kernel
  TileXRUdmaDemoCopyBytesGmToGm
```

## `data_as_flag_epoch_ordered` 数据路径

`data_as_flag_epoch_ordered` kernel 使用 data-as-flag 块布局：

```text
512B 块:
  480B payload
   32B flag/gap
```

发送端路径：

```text
连续 src payload
  -> UB 暂存
  -> 打包成 512B 块:
       480B payload + 32B flag/gap
  -> 对端 data-as-flag IPC 窗口
  -> 在该 batch 最后一个块的 flag 区写入 32B batch commit flag
```

接收端路径：

```text
轮询 batch commit flag
  -> 从 512B data-as-flag 窗口读取
  -> 解包/提取 480B payload 区域
  -> 写入连续的 dst payload
```

epoch-ordered 变体相比旧的按块 data-as-flag 路径有所改进：每个 batch 只写一次 commit flag，而不是每个 480B payload 块写一次。这消除了破坏性最大的小写瓶颈，但该路径仍然存在额外的布局转换、轮询，以及每个 batch 一次的小 commit 写入。

相关文件：

```text
src/include/tilexr_data_as_flag.h
  DATA_AS_FLAG_BLOCK_BYTES = 512
  DATA_AS_FLAG_PAYLOAD_BYTES = 480
  DATA_AS_FLAG_FLAG_BYTES = 32
  DataAsFlagSendEpochOrdered
  DataAsFlagWriteBatchCommitFlag
  DataAsFlagCheckAndRecvEpochOrdered
  DataAsFlagCopyBatchToRecvGM

tests/udma/demo/tilexr_udma_demo_kernel.cpp
  tilexr_data_as_flag_epoch_ordered_p2p_perf_kernel
```

## 差距的主要来源

### 1. 布局开销

`data_as_flag_epoch_ordered` 把每个 `480B` payload 存进一个 `512B` 块：

```text
512 / 480 = 1.067x
```

因此在考虑任何控制或解包开销之前，数据窗口就至少有约 `6.7%` 的额外空间和传输开销。

该开销在 host 侧窗口计算中可见：

```text
DataAsFlagWindowBytes(payloadBytes)
  = ceil(payloadBytes / 480) * 512
```

但带宽报告使用的是有效 payload 字节数，而不是展开后的 data-as-flag 窗口字节数。因此额外的块布局字节会拉低报告的有效带宽。

相关文件：

```text
tests/udma/demo/tilexr_udma_p2p_perf_config.h
  DataAsFlagWindowBytes
  P2PEffectiveTransferBytes
  FormatP2PPerfCsvRow
```

### 2. 经过 UB 的打包与解包

`memory_consume` 是从连续 payload 拷贝到连续 payload。

`data_as_flag_epoch_ordered` 必须对 payload 做重塑：

```text
发送端:
  连续 src
    -> UB
    -> 512B 块中的 480B payload 分块
    -> 对端 GM

接收端:
  512B 块/gap 窗口
    -> UB
    -> 提取 480B payload 分块
    -> 连续 dst
```

这增加了连续 `memory_consume` 路径所不需要的额外 MTE 操作、barrier 以及跨步拷贝行为。

### 3. Batch Commit 小写入

epoch-ordered 路径不再为每个 `480B` payload 块写一次 flag。而是在一个 batch 的 payload 写完后，向该 batch 最后一个块的 flag 区写入一个 `32B` commit flag：

```text
payload batch 写入:
  batchBlocks * 512B

batch commit 写入:
  32B commit flag
```

这就是“batch commit 小写入”。

它比按块写 flag 好得多，但仍存在固定开销：

```text
准备 32B epoch flag
S -> MTE3 同步
MTE3 向 GM/对端窗口写入 32B commit flag
MTE3 -> S 同步
接收端轮询并读取 commit flag
```

对于大 batch，该开销会被摊薄，但并非免费。结合打包/解包和 512B 块布局，它仍会限制最终带宽。

## 为什么 `0.7 GB/s` 能恢复到 `~10 GB/s`

旧的 data-as-flag 路径实际上以极高频率付出小 flag 写入/检查开销——大约每个 `480B` payload 块一次。这使小写入和轮询主导了 payload 传输。

epoch-ordered 路径将其降为每个 batch 一次 commit 写入：

```text
旧路径:
  每个 480B payload 块都有 ready/flag 开销

epoch-ordered:
  每个 batch 只有一个最终的 32B commit flag
```

这就解释了为什么性能可以显著恢复，例如从约 `0.7 GB/s` 恢复到约 `10 GB/s`。

但它并未把该路径变成连续 payload 传输。该协议仍要为 data-as-flag 布局付出代价，接收端仍需等待，再从内嵌布局窗口中解包。

## 最终解读

`memory_consume`：

```text
用独立的同步区来证明 payload 就绪。
保持 payload 数据窗口连续。
针对大块连续 payload 搬移做优化。
```

`data_as_flag_epoch_ordered`：

```text
用数据窗口内嵌的 commit flag 来证明 payload 就绪。
以 480B + 32B 的 data-as-flag 块存放 payload。
付出打包/解包、轮询和 batch commit 开销。
```

因此在当前实现和相同有效 payload 带宽统计口径下，出现如下差距：

```text
data_as_flag_epoch_ordered: ~10 GB/s
memory_consume:            ~50 GB/s
```

是符合预期的。这并不一定说明 epoch-ordered 的修复失败；它说明该修复消除了最严重的按块 flag 瓶颈，而剩余的 data-as-flag 协议路径仍比带外部同步的连续 payload 传输重得多。
