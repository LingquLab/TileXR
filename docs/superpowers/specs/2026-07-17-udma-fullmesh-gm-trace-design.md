# UDMA Full-Mesh GM Trace Design

## Goal

Add per-core, per-pass, per-peer cycle spans to the physical full-mesh
AllToAll path. Device code records raw system-cycle timestamps in a dedicated
8 MiB GM allocation. Host code only initializes, copies, and writes the trace;
analysis and Chrome Tracing conversion happen after the run.

## Scope

- Instrument the non-remote-put-only multi-node bigdata full-mesh kernel.
- Record up to 50 measured iterations for cores 0 through 34.
- Index task spans by iteration, core, pass, peer, and phase.
- Preserve the current 12:4 remote-send split and communication behavior.
- Keep tracing disabled by default.
- Reject trace dimensions that do not fit in 8 MiB before kernel launch.

## Storage Layout

The trace allocation contains a 4 KiB header, a fixed kernel-span region, and
a runtime-sized task-span region. Every span is two `uint64_t` values:
`beginCycle` and `endCycle`.

Kernel spans use fixed indexing:

```text
[iteration][core][kernel | work]
```

Task spans use runtime dimensions from the header:

```text
[iteration][core][pass][peer][phase]
```

For repeat50, 35 cores, one pass, 16 peers, and 14 phases, task spans consume
6,272,000 bytes. The complete layout fits in 8 MiB. Host validation computes
the exact required bytes from the requested repeat, pass count, and rank size.

## Phases

The task phase enum contains:

1. pass
2. self-copy
3. peer-copy
4. publish-copy-ready
5. wait-copy-ready
6. data-put
7. quiet
8. segment-done
9. publish-ready
10. wait-ready
11. output-copy
12. publish-recv-done
13. wait-recv-done
14. ACK

Core responsibilities map to phases as follows:

- Cores 0-15: self-copy, peer-copy, and publish-copy-ready where applicable.
- Cores 16-17: wait-copy-ready, data-put, quiet, and segment-done. Core 16
  additionally records publish-ready.
- Core 18: wait-copy-ready, data-put, and quiet for local peers.
- Cores 19-34: wait-ready, output-copy, and publish-recv-done. Core 34
  additionally records wait-recv-done and ACK.

## Device Integration

The host passes a nullable trace pointer and an iteration index to the kernel.
Trace helpers return immediately when the pointer is null or any dimension is
out of range. `GetSystemCycle()` and GM span writes only execute when tracing
is enabled.

Worker helpers receive enough trace context to record boundaries around the
actual wait, copy, put, quiet, signal, and ACK operations. Existing
synchronization and data movement remain unchanged.

## Host Integration

Tracing is enabled by `TILEXR_UDMA_FULLMESH_TRACE=1`. The output directory is
selected with `TILEXR_UDMA_FULLMESH_TRACE_DIR`, defaulting to the current
directory.

When enabled, Host code:

1. validates repeat/pass/rank dimensions against the 8 MiB capacity;
2. allocates and zero-initializes trace GM;
3. writes the trace header;
4. launches each measured iteration with its trace index;
5. copies the complete trace allocation to Host after stream synchronization;
6. writes `tilexr_fullmesh_trace_rank_<rank>.bin`.

Failure to allocate, initialize, copy, or write the trace fails the demo run.

## Conversion

`tilexr_udma_fullmesh_trace_to_chrome.py` accepts one or more rank binary files
and writes Chrome Trace Event JSON. Each rank is a process and each core is a
thread. Events include iteration, pass, peer, phase, begin cycle, end cycle,
and duration. Timestamps are normalized to the earliest recorded kernel span
for each rank and converted with 1000 cycles per microsecond.

## Verification

- Header/layout tests validate offsets, capacity, and overflow rejection.
- Source-structure tests require nullable trace plumbing and phase recording in
  every full-mesh core role.
- Converter tests validate metadata, event names, peer/pass arguments, and
  Chrome-compatible JSON without unsupported display-time units.
- Remote builds run layout and converter tests before hardware execution.
- Physical 2x8 runs repeat50 with full correctness and trace enabled. The
  resulting binaries are downloaded and converted to Chrome JSON.
