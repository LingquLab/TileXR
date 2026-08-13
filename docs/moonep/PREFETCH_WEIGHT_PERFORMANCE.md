# MoonEP PrefetchWeight Performance

## Result

The shared-QP PrefetchWeight implementation was validated on 2026-08-13 against
TileXR main and native MoonEP. The optimized implementation keeps logical slot
ownership unchanged but maps its four workers to physical QPs `{0,1,2,16}`.
This distributes equal-size slot traffic across the six-port and two-port CLOS
groups in the hardware's 3:1 port ratio.

The test used eight physical Ascend950PR devices on `141.61.49.195`, CANN
`9.1.T560`, driver `25.1.rc1.b188`, TileXR base commit `c39c433`, and native
MoonEP commit `53e03002655d07cfc39e7e9ca2c2aa18583c6c0b`.

| Implementation | P50 | P99 | Effective bandwidth |
| --- | ---: | ---: | ---: |
| TileXR main `c39c433` | 1272.83 us | 2662.19 us | 276.80 GB/s |
| TileXR shared-QP mapping | 986.99 us | 1044.07 us | 356.97 GB/s |
| Native MoonEP `53e0300` | 7554.13 us | 7810.42 us | 46.64 GB/s |

Relative to TileXR main, the optimized P50 is 22.46% lower and effective
bandwidth is 28.96% higher. It is 7.65 times faster than native MoonEP by P50
and reaches 89.24% of the 400 GB/s one-card one-way external-port ceiling.
The independent dual-card validation in
`docs/UDMA_DUAL_CARD_BANDWIDTH_VALIDATION.md` measured about 389 GB/s per card,
so the result is close to the demonstrated transport limit but does not yet
close the remaining data-plane gap.

## Workload And Timing

The fixed EP8 case is:

```text
ranks:                 8 physical ranks, one rank per NPU
experts:               32 total, 4 local experts per rank
remote slots:          4 per rank
dtype:                 BF16
gate/up row:           [7168, 2048], 28 MiB each
down row:              [2048, 7168], 28 MiB
bytes per slot:        84 MiB
bytes per rank/round:  336 MiB (352321536 bytes)
samples:               3 x (5 warmup + 20 measured)
```

Before timing, every rank pulls the four experts owned by the previous rank and
checks every BF16 value in all twelve destination projection slots exactly.
The timed interval uses events on the current NPU stream. TileXR times one fused
gate/up/down launch; native MoonEP times its three sequential `launch_prefetch`
calls. Allocation, MR or SHMEM registration, correctness validation, the
pre-iteration distributed barrier, and post-launch status reads are excluded.

For every iteration the report selects the maximum event time across all eight
ranks. P50 and P99 use linear interpolation over those cross-rank maxima.
Effective bandwidth is `352321536 / P50_us / 1000` GB/s.

## Sample Quality

The TileXR event API intermittently returned `0.044 us`, which is below any
possible 336 MiB transfer time. The benchmark preserves every raw event sample,
but classifies readings at or below 1 us as invalid and excludes only those
readings from P50, P99, and bandwidth:

| Implementation | Raw samples | Valid samples | Invalid `0.044 us` samples |
| --- | ---: | ---: | ---: |
| TileXR main | 60 | 55 | 5 |
| TileXR shared-QP mapping | 60 | 52 | 8 |

The optimized repeat P50 values were 975.78, 990.03, and 985.84 us. Main
contained one 4143.04 us system spike; no high sample was filtered, so its
aggregate P99 is 2662.19 us. Main's three repeat P99 values were 1384.15,
3662.42, and 1310.24 us. The optimized repeat P99 values were 1021.40,
1052.77, and 1017.52 us. P50 and bandwidth are the primary throughput comparison;
larger runs are required before treating either P99 as a production tail claim.

## Retained Artifacts

The complete rank JSON files, aggregate JSON, and launch logs remain on the
test host:

```text
/tmp/TileXR-prefetch-opt-20260813-c39c433/
  artifacts/prefetch-final-main-20260813/
  artifacts/prefetch-final-opt-20260813/
  prefetch-final-main-20260813.log
  prefetch-final-opt-20260813.log

/tmp/TileXR-prefetch-baseline-20260813-c39c433/
  artifacts/prefetch-formal-native-dev/
  native-formal-dev.log
  hccl-aiv-only-8r.log
```

Each artifact directory contains `rank_0.json` through `rank_7.json` plus
`summary.json`. The TileXR artifact labels identify the tested main and optimized
source snapshots; both are based on `c39c433`. The native aggregate records the
full native commit. The matching official HCCL AIV-only environment baseline
passed before TileXR performance investigation.

## Implementation Boundaries

The optimization separates logical workers from physical QPs. Logical worker
`w` still owns slots `w`, `w + workerCount`, and so on. Only physical queue
selection changes:

| Shared-domain workers | Physical QPs |
| ---: | --- |
| 1 | `0` |
| 2 | `0,1` |
| 4 | `0,1,2,16` |
| 8 | `0,1,2,3,4,5,16,17` |

Non-shared domains retain identity mapping. The selected QPs are packed in a
private 64-bit Kernel argument. Public MoonEP structures and Python APIs are
unchanged. Registered memory, peer-memory behavior, slot assignment, WQE
construction in UB, MTE3 SQ publication, `st_dev` doorbells, and CQ completion
semantics are unchanged. There is no second PrefetchWeight implementation path.

## Experiments Not To Repeat

The following experiments did not improve the large-transfer bottleneck:

| Experiment | Result | Conclusion |
| --- | --- | --- |
| `blockDim=1/2/4` before QP remapping | 270.75 / 274.42 / 275.30 GB/s | More QPs inside the same six-port CLOS do not add useful bandwidth. |
| Reverse `TILEXR_UDMA_QP_ROUTE_SPEC` | No material change | A shared communicator constructs the fixed 32-QP profile, so this variable does not remap that profile. |

The following setup failures were environmental or invocation mistakes, not
PrefetchWeight defects:

- The remote `torchrun` shebang referenced a removed Conda environment. Use
  `python -m torch.distributed.run`.
- `TILEXR_BUILD_EP=ON` does not build MoonEP. Use
  `TILEXR_BUILD_MOONEP=ON`.
- Streaming a Windows-produced tar archive into Linux caused archive-format
  problems. The successful workflow used a remote main snapshot and copied
  changed files individually.

## Next Stage

1. Profile the remaining 32.46 GB/s gap to the independently measured
   approximately 389 GB/s card transport rate. Use msprof and physical-port
   counters to separate queue/WQE issue limits, the fused Kernel's scalar work,
   and link utilization. Do not tune small Host overhead before this is known.
2. Increase the sample count and determine why NPU events sometimes report
   `0.044 us`. Keep raw samples and correctness checks; do not silently discard
   high values or quote production P99 until the event anomaly is understood.
3. Sweep large payload and slot-count distributions around the production
   shape. Confirm that the 3:1 mapping remains optimal when workers own unequal
   byte counts; derive a byte-aware mapping only if evidence shows imbalance.
4. Measure the transfer-size crossover against the unchanged peer-memory path.
   Memory remains preferable for small transfers, while registered UDMA should
   remain the large-transfer path. Do not select either transport from topology
   alone.
5. After the data plane is saturated, measure fixed launch/status overhead and
   consider reducing it only if it becomes a material fraction of stage time.

These measurements prove the registered-memory UDMA data plane on this
Ascend950 topology. They do not establish UDMA performance on 910B, a simulator,
another CLOS layout, an oversubscribed rank topology, or cross-node MoonEP.
