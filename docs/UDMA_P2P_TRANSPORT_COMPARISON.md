# UDMA P2P Transport Comparison

**Topic:** `memory_consume` vs `data_as_flag_epoch_ordered`

## Summary

`data_as_flag_epoch_ordered` has removed the worst per-block flag write bottleneck, but its payload path is still much heavier than `memory_consume`. Under the same effective-payload bandwidth accounting, a result such as `~10 GB/s` vs `~50 GB/s` is consistent with the current implementation.

At a high level:

```text
memory_consume:
  continuous payload transfer + independent sync flag

data_as_flag_epoch_ordered:
  data-as-flag protocol validation with embedded commit information
  inside a 480B payload + 32B flag/gap block layout
```

`memory_consume` is closer to an efficient continuous payload transport. `data_as_flag_epoch_ordered` is closer to a protocol path that proves data readiness by embedding ready information in the data window.

## `memory_consume` Data Path

The `memory_consume` kernel keeps the payload window continuous:

```text
sender:
  continuous src payload
    -> peer IPC data window
  set independent outer sync flag

receiver:
  wait independent outer sync flag
  continuous local peer IPC data window
    -> continuous dst payload
```

In code, the sender copies from `srcGM + offset` to `peerBase + dstByteOffset + offset`, then calls `sync.SetOuterFlag(magic, step)`. The receiver waits with `sync.WaitOuterFlag(...)`, then copies from `localBase + dstByteOffset + offset` to `dst + offset`.

The payload layout remains continuous on both sides. Although the helper still stages through UB internally, the copied payload is a simple continuous GM-to-GM stream. For large messages, the independent sync flag is amortized across a large payload, so this path can reach much higher bandwidth.

Relevant files:

```text
tests/udma/demo/tilexr_udma_demo_kernel.cpp
  tilexr_memory_consume_p2p_perf_kernel
  TileXRUdmaDemoCopyBytesGmToGm
```

## `data_as_flag_epoch_ordered` Data Path

The `data_as_flag_epoch_ordered` kernel uses a data-as-flag block layout:

```text
512B block:
  480B payload
   32B flag/gap
```

The sender path is:

```text
continuous src payload
  -> UB scratch
  -> pack into 512B blocks:
       480B payload + 32B flag/gap
  -> peer data-as-flag IPC window
  -> write 32B batch commit flag in the last block of the batch
```

The receiver path is:

```text
poll batch commit flag
  -> read from 512B data-as-flag window
  -> unpack/extract 480B payload regions
  -> write continuous dst payload
```

The epoch-ordered variant improves over the older per-block data-as-flag path by writing a commit flag once per batch instead of once per 480B payload block. This removes the most damaging small-write bottleneck, but the path still has extra layout conversion, polling, and a small commit write for each batch.

Relevant files:

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

## Main Sources Of The Gap

### 1. Layout Overhead

`data_as_flag_epoch_ordered` stores every `480B` payload in a `512B` block:

```text
512 / 480 = 1.067x
```

So the data window has at least about `6.7%` extra space and transfer overhead before considering any control or unpacking cost.

This overhead is visible in the host-side window calculation:

```text
DataAsFlagWindowBytes(payloadBytes)
  = ceil(payloadBytes / 480) * 512
```

The bandwidth report, however, uses effective payload bytes, not expanded data-as-flag window bytes. Therefore the extra block-layout bytes reduce the reported effective bandwidth.

Relevant file:

```text
tests/udma/demo/tilexr_udma_p2p_perf_config.h
  DataAsFlagWindowBytes
  P2PEffectiveTransferBytes
  FormatP2PPerfCsvRow
```

### 2. Pack And Unpack Through UB

`memory_consume` copies continuous payload to continuous payload.

`data_as_flag_epoch_ordered` has to reshape the payload:

```text
sender:
  continuous src
    -> UB
    -> 480B payload chunks inside 512B blocks
    -> peer GM

receiver:
  512B block/gap window
    -> UB
    -> extract 480B payload chunks
    -> continuous dst
```

This adds extra MTE operations, barriers, and strided copy behavior that the continuous `memory_consume` path does not need.

### 3. Batch Commit Small Write

The epoch-ordered path no longer writes a flag for every `480B` payload block. Instead, after a batch payload write, it writes one `32B` commit flag into the flag area of the last block in that batch:

```text
payload batch write:
  batchBlocks * 512B

batch commit write:
  32B commit flag
```

This is the "batch commit small write".

It is much better than per-block flag writes, but it still has fixed overhead:

```text
prepare 32B epoch flag
S -> MTE3 synchronization
MTE3 writes 32B commit flag to GM/peer window
MTE3 -> S synchronization
receiver polls and reads the commit flag
```

For large batches this cost is amortized, but it is not free. Combined with pack/unpack and the 512B block layout, it still limits the final bandwidth.

## Why `0.7 GB/s` Can Recover To `~10 GB/s`

The older data-as-flag path effectively paid a small flag-write/check cost at very high frequency, roughly once per `480B` payload block. That made small writes and polling dominate the payload transfer.

The epoch-ordered path reduces this to one commit write per batch:

```text
old:
  every 480B payload block has ready/flag cost

epoch-ordered:
  each batch has one final 32B commit flag
```

This explains why performance can recover significantly, for example from around `0.7 GB/s` to around `10 GB/s`.

However, it does not turn the path into a continuous payload transport. The protocol still pays for the data-as-flag layout and the receiver still has to wait for, then unpack from, the embedded-layout window.

## Final Interpretation

`memory_consume`:

```text
Use an independent sync area to prove payload readiness.
Keep the payload data window continuous.
Optimize for large continuous payload movement.
```

`data_as_flag_epoch_ordered`:

```text
Use an embedded commit flag inside the data window to prove payload readiness.
Store payload in 480B + 32B data-as-flag blocks.
Pay pack/unpack, polling, and batch commit costs.
```

Therefore, under the current implementation and the same effective-payload bandwidth accounting, a gap such as:

```text
data_as_flag_epoch_ordered: ~10 GB/s
memory_consume:            ~50 GB/s
```

is expected. It does not necessarily indicate that the epoch-ordered fix failed; it indicates that the fix removed the worst per-block flag bottleneck while the remaining data-as-flag protocol path is still substantially heavier than continuous payload transfer with external synchronization.
