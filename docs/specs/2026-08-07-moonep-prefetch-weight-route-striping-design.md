# MoonEP PrefetchWeight Route Striping Design

## Goal

Make each remote expert projection row use all active PrefetchWeight UDMA QPs.
For the two-worker same-server configuration, QP 0 selects the topology
(FullMesh) route and QP 1 selects the six-port aggregate (Clos) route. The row
is split with 1:6 weights.

## Scope

- Preserve the public MoonEP V1 C ABI.
- Preserve the direct embedded-kernel launch path.
- Keep one AIV owner per QP so SQ producer state is never shared by AIVs.
- Split gate, up, and down rows independently at 64-byte boundaries.
- Keep the one-QP path behaviorally unchanged.

The change does not add cross-node route-weight discovery, change the Planner,
or make Dispatch, Combine, and ReduceGrad native.

## Contract

Host parses the effective QP route specification into positive route weights:
`topology` has weight 1 and `port_count:N` has weight N. The weights are packed
into the private direct-launch kernel argument ABI. Missing explicit route
configuration uses weight 1 for each QP.

For an aligned row of `R` bytes, worker `i` owns:

```text
begin = align_down(R * prefixWeight(i) / totalWeight, 64)
end   = i is last ? R : align_down(R * prefixWeight(i + 1) / totalWeight, 64)
```

Empty slices are legal. Non-empty slices are disjoint, aligned, and cover the
entire row. Every active worker visits every plan slot, submits only its slice
on its matching QP, and quiets only that QP for peers on which it submitted.

## Compatibility

`--prefetch-workers 1` remains `port_count:6`. `--prefetch-workers 2` becomes
`topology,port_count:6`. Four- and eight-worker route specifications remain
unchanged in this change so their behavior can be evaluated separately.

The benchmark's logical transferred-byte accounting remains the full row size,
not the sum of route-local instrumentation, because the slices partition rather
than duplicate each row.

## Validation

- Host tests prove parsing, weighted boundaries, tails, empty slices, overflow
  rejection, launch propagation, and one-QP compatibility.
- Source guards prove the direct-launch ABI and all-slot kernel traversal.
- CANN 9.1.0 compiles the final A5/Ascend950 kernel.
- A two-card single-remote-slot case proves both slices reconstruct exact BF16
  and FP16 rows while unused slots and source rows remain unchanged.
- Route logs or port counters must show non-zero FullMesh and Clos traffic before
  claiming special-P2P data-plane coverage.
- Performance compares Clos-only with FullMesh+Clos using identical cases,
  warmup, iterations, correctness, and physical devices.
