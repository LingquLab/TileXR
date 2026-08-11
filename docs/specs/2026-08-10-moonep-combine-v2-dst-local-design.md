# MoonEP Combine V2 dstLocal and Reduction Design

## Goal

Make the native MoonEP Python path use Combine V2 while preserving the public
`Buffer.combine()` contract and `tools/moonep/test_npu_e2e.py`. Planner V3 must
produce the reverse route map required by Combine V2, and Combine V2 must return
expert rows and reduce `[S, K, H]` into `[S, H]` in FP32 accumulation precision.

## Route Contracts

Planner's existing forward route is:

```text
dst[token * K + topk] = expertRank * NvS + expertRecvSlot
```

Combine V2 consumes the inverse route:

```text
dstLocal[expertRecvSlot] = srcRank * NvS + token * K + topk
```

`dstLocal` has `NvS` entries. Planner initializes padding and unused slots to
`-1`; Combine V2 skips those slots.

Planner cannot derive a destination rank's inverse map from local TopK input
alone. Each rank therefore publishes its completed forward `dst[S*K]` into the
existing planner peer window. After a cross-rank ready barrier, every rank scans
the published tables and builds its local inverse map in the persistent planner
workspace. The workspace layout exposes the `dstLocal` offset to the internal
Python plan without changing the public compatibility plan.

## Combine V2 Data Flow

Dispatch and Combine V2 share one registered workspace, sized to the maximum of
their layout requirements. The stages are serialized on the same NPU stream.

1. Copy local expert output `[NvS, H]` to the registered workspace source area.
2. Decode each active `dstLocal` with integer division and modulo by `NvS`.
3. Reverse-scatter each expert row to the target rank and target route slot.
4. Wait for send completions and all inbound completion tokens.
5. Use the existing active-core convergence point as the local completion
   barrier before reading the receive scratch.
6. Partition the `S` tokens over active vector cores with remainder-safe ranges.
7. For each token, process TopK rows in ordered batches of at most four, cast
   BF16 rows to FP32, accumulate in TopK order, and cast the final row to BF16.
8. Write the final `[S, H]` result to the dedicated workspace output region and
   copy it to the caller-provided output tensor on the same stream.

`NvS` is not required to be a power of two. The PR113 E2E shape uses `NvS=2040`,
so rank and slot decoding must not use shift/mask operations.

## Route Weights

When `routeWeightsNvs` is present, run the same reverse-scatter mapping with
FP32 rows of width one and no reduction. Copy the first `S*K` received values to
`routeWeightsSk`. Combine does not multiply hidden rows by route weights because
the forward path already applies them to expert output.

## Python Integration

The internal native plan stores the planner-workspace `dstLocal` offset. The
runtime loads and calls the Combine V2 stage entry point directly; it no longer
constructs `TileXRMoonEPCombineArgsV1` or calls `TileXRMoonEpCombineV1`.
Public compatibility objects and `test_npu_e2e.py` remain unchanged.

## Validation

- Planner layout and route inversion unit tests, including padding sentinels.
- Combine V2 layout/schedule/source tests for dynamic PR113 shapes and non-power-
  of-two destination decoding.
- Python runtime and public compatibility tests proving the V2 symbol is used.
- A5/Ascend950 single-node 8P run of `tools/moonep/test_npu_e2e.py`, covering all
  six MoonEP stages.
