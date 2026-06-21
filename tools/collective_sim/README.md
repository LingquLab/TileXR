# TileXR Collective Simulation Tool

This directory contains the offline collective simulation tool. It evaluates generic collective algorithm DAGs without Ascend hardware, ACL, CANN runtime calls, or compiled TileXR libraries.

## Constraints

- Algorithms are standard JSON DAGs.
- Cross-rank communication acts on communication buffer roles.
- DataCopy-mode and UDMA-mode transfers require at least one communication buffer endpoint.
- Local copy-in and copy-out are explicit DAG `copy` operations.
- The first topology model is simplified 1D-Clos.
- P2P uses a message-size curve that approaches 50 GB/s.
- Uplink uses a message-size curve that approaches 300 GB/s.
- SDMA local copy uses a default 800 GB/s curve until calibration overrides it.
- Resource contention is a deterministic reservation model: operations with the same computed start time fair-share matching resources as a batch, then reserve those resources until the batch completes. It does not model continuous dynamic contention between staggered transfers.
- The tool requires no Ascend hardware.

## Commands

The CLI command surface is `collective-sim validate`, `collective-sim run`, and `collective-sim sweep`. The examples below invoke the same commands through the Python module entrypoint so they work without installation.

Run from the repository root with:

```bash
export PYTHONPATH=tools/collective_sim
python3 -m tilexr_collective_sim.cli validate tools/collective_sim/examples/allgather_1d_clos/algorithm.json --topology tools/collective_sim/examples/allgather_1d_clos/topology_64p.yaml
python3 -m tilexr_collective_sim.cli run tools/collective_sim/examples/allgather_1d_clos/case.yaml --out run/collective_sim/allgather
python3 -m tilexr_collective_sim.cli sweep tools/collective_sim/examples/allgather_1d_clos/sweep.yaml --out run/collective_sim/allgather_sweep
```

The run and sweep commands write:

- `result.json`
- `results.json`
- `summary.csv`
- `report.html`: lightweight algorithm-selection index with rank/card navigation
- `rank_reports/rank_XXX.html`: per-rank Perfetto UI entry pages
- `profiles/profile_XXX_<algorithm>_<ranks>p_<bytes>b.trace.json`: one Perfetto-compatible Chrome JSON trace per collective test; rank count and message size are not mixed inside a profile

## Sweep Schema

Use `cases` when algorithms and topologies must be paired by compatible rank count:

```json
{
  "sweep": {
    "cases": [
      {"algorithm": "algorithm.json", "topology": "topology_64p.yaml"},
      {"algorithm": "algorithm_128p.json", "topology": "topology_128p_2to1.yaml"}
    ],
    "calibration": "calibration.yaml",
    "message_bytes": [1024, 4096, 1048576, 67108864],
    "validate": true
  }
}
```

The legacy top-level `algorithms` plus `topologies` form is still accepted and expands to the Cartesian product.

## Test

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest discover -s tests/collective_sim -v
```
