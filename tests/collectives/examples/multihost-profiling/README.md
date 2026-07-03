# Multi-Host Profiling Example Artifacts

This directory contains small example outputs from a real two-host standalone collectives profiling run. These files are documentation/test artifacts only; they are not consumed by the build.

## Files

- `fine-allgather-big/trace.json`: Root-level aggregate Perfetto trace from a two-host `allgather` profiling run.

## How To Use

Open `fine-allgather-big/trace.json` in `https://ui.perfetto.dev`.

Useful searches:

```text
kernel_total
chunk_total
flag_poll_wait
peer_ipc_to_output
rank0@
rank1@
launch0/
```

The trace demonstrates the expected aggregate event naming scheme:

```text
launch0/rank0@<host>/kernel_total
launch0/rank1@<host>/flag_poll_wait
```

It is intentionally kept as a checked-in example so reviewers and AI agents can inspect the expected Perfetto schema without rerunning the two-host benchmark.
