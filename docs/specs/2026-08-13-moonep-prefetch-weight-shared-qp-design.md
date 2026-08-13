# MoonEP PrefetchWeight Shared-QP Design

## Goal

Increase PrefetchWeight throughput on Ascend950 shared UDMA domains by using
both external-port CLOS groups. Preserve the current fused three-projection
launch, registered-memory transport, public MoonEP ABI, and peer-memory paths.

The target workload is EP8 with four remote expert slots per rank. Each BF16
slot contains 28 MiB gate, 28 MiB up, and 28 MiB down rows, for 336 MiB per
rank per launch.

## Evidence And Result

On `141.61.49.195`, the final back-to-back run measured TileXR `c39c433` at
1272.83 us P50 and 276.80 GB/s. Native MoonEP `53e0300` achieved 7554.13 us
P50 and 7810.42 us P99, or 46.64 GB/s, for the same data and explicit plan.

PrefetchWeight currently uses the logical worker index as the physical QP.
The fixed shared-domain profile assigns QPs 0-15 to the six-port CLOS and QPs
16-31 to the two-port CLOS. A block-dimension sweep measured 270.75, 274.42,
and 275.30 GB/s with one, two, and four workers. More QPs within the six-port
CLOS therefore provide little additional throughput, while the two-port CLOS
is unused.

Changing `TILEXR_UDMA_QP_ROUTE_SPEC` did not alter the baseline because a
shared-QP communicator always constructs the fixed 32-QP profile. This was a
diagnostic experiment, not a supported tuning mechanism for this path.

The implemented four-worker map `{0,1,2,16}` measured 986.99 us P50,
1044.07 us P99, and 356.97 GB/s. P50 improved by 22.46% and bandwidth by
28.96% over main, and the optimized path was 7.65 times faster than native
MoonEP by P50. Exact BF16 slot validation passed on all eight physical ranks.
`docs/moonep/PREFETCH_WEIGHT_PERFORMANCE.md` records the full measurement
method, raw artifact paths, event anomalies, ineffective experiments, and
next-stage work.

## Design

Keep logical work assignment unchanged: worker `w` owns slots `w`,
`w + workerCount`, and so on. Add a private logical-worker-to-physical-QP map
to `PrefetchWeightLayout` and the direct-launch Kernel ABI.

For a fixed 32-QP shared domain:

| Workers | Physical QPs | CLOS traffic ratio |
| ---: | --- | ---: |
| 1 | `0` | 1:0 |
| 2 | `0,1` | 2:0 |
| 4 | `0,1,2,16` | 3:1 |
| 8 | `0,1,2,3,4,5,16,17` | 6:2 |

All non-shared domains retain identity mapping. Host validation requires every
selected QP to be below the transport QP count. The map is packed as eight
8-bit indices in one `uint64_t`; this keeps the private launch block compact
and covers the transport maximum of 32 QPs.

The Kernel continues to use the logical worker for slot partitioning and uses
the mapped physical QP only for WQ lookup, UDMA GET submission, and CQ wait.
There is no change to registered regions, transfer sizes, WQE construction,
doorbell ordering, or completion semantics.

## Compatibility And Non-Goals

- Preserve C++14, CANN 9.1, and the registered direct-Kernel launch path.
- Do not change `TileXRMoonEpPrefetchWeightArgsV1` or Python-facing APIs.
- Do not change UDMA registration or the fixed shared-QP profile.
- Do not change peer-memory, Dispatch, Combine, or ReduceGrad.
- Do not optimize small fixed Host overhead until the large-transfer path is
  measured after using both CLOS groups.

## Verification

1. Host tests cover identity mapping, 32-QP four/eight-worker mappings, and
   packed-map propagation into the launch context.
2. Source/launch tests cover the private ABI and require all UDMA queue
   operations to use the mapped physical QP.
3. A complete CANN 9.1 build proves Host and embedded Kernel ABI consistency.
4. Ascend950 EP8 tests validate exact BF16 slot contents and run the same
   `3 x (5 warmup + 20 measured)` NPU-event benchmark as the baseline.
5. Retain the native MoonEP baseline and report P50, P99, effective bandwidth,
   repeat-level results, and any ineffective experiments.

The original performance hypothesis was 350-390 GB/s, corresponding to roughly
900-1000 us P50. The measured 356.97 GB/s and 986.99 us P50 satisfy that gate,
so the shared-QP mapping is retained.
