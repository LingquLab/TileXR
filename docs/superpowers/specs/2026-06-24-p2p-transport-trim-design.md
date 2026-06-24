# P2P Transport Trim Design

## Goal

Reduce the UDMA P2P performance demo transport matrix to three user-facing modes:

- `direct_urma`
- `memory`
- `data_as_flag`

`direct_urma` will remain the only UDMA benchmark transport name, but it will use the current parallel multi-jetty implementation internally.

## Behavior

`direct_urma` represents direct registered-memory UDMA transfer. It supports both a single-jetty baseline and multi-jetty parallel transfer through the existing `block_dim` and `TILEXR_UDMA_QP_NUM` controls:

- `block_dim=1` with one QP behaves like the previous single-QP direct URMA path.
- `block_dim=N` with `TILEXR_UDMA_QP_NUM=N` uses up to `N` QPs/jettys in parallel.

The removed user-facing variants are:

- `direct_urma_multi_wqe`
- `direct_urma_multi_jetty`
- `direct_urma_multi_jetty_parallel`
- `direct_urma_multi_jetty_parallel_fixed_wqe`

The previous `direct_urma_multi_jetty_parallel` behavior is folded into `direct_urma`.

## Code Changes

Update the P2P transport enum, parser, formatter, validation messages, CSV output tests, runner scripts, and documentation so only the three supported names appear.

Remove the obsolete benchmark kernels and host launch wrappers for:

- single-WQE `direct_urma`
- `multi_wqe`
- serial `multi_jetty`
- fixed-WQE parallel multi-jetty

Keep the existing QP-indexed device helpers and UDMA transport layout support because the new `direct_urma` implementation depends on them.

Remove fixed-WQE-only window sizing and mismatch-check logic from the P2P config helpers and host demo code.

## Scripts And Docs

The P2P runner should set `TILEXR_UDMA_QP_NUM` from `block_dim` when `transport=direct_urma`. The concurrency sweep defaults should cover:

- `direct_urma`
- `memory`
- `data_as_flag`

Documentation and PR wording should describe `direct_urma` as a scalable direct UDMA path rather than listing multiple UDMA variants.

## Testing

Use TDD for the implementation:

1. First update host-only unit tests to expect the trimmed transport set and the new `direct_urma` CSV semantics.
2. Run the unit target and confirm the tests fail before production changes.
3. Apply the production changes.
4. Re-run the focused UDMA unit tests and any available build/test script.

Hardware P2P demo runs remain a manual verification item on supported Ascend hardware.
