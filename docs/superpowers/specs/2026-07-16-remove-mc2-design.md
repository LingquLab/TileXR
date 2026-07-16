# Remove MC2 Design

## Goal

Remove TileXR-owned MC2 operators and their build/test surface. Keep the upstream
`ops-transformer` repository available only as on-demand reference source under
`reference/ops-transformer`.

## Repository Changes

- Delete the complete `examples/mc2` tree.
- Remove MC2-only build and run scripts from `scripts/`.
- Remove active environment setup and dependency-install steps that exist only
  for MC2.
- Remove the `3rdparty/ops-transformer` submodule and its `.gitmodules` entry.
- Add `ops-transformer` to `reference/download_cann_repos.sh`, following the
  repository's ignored, on-demand reference checkout model.
- Remove the MC2-specific Ascend C review rules because their target code no
  longer exists in TileXR.

## Documentation Changes

Update current project documentation so it no longer advertises MC2 as a TileXR
feature, dependency, build flow, repository directory, or validation target.
Document `reference/ops-transformer` as optional upstream reference code.

Historical specs and plans remain unchanged because they record the context and
constraints of earlier work. Any current document that links to the removed
code or commands must be updated.

## Dependent Workflows

EP and collectives are TileXR-owned implementations and must not initialize,
copy, include, or link `ops-transformer`. Their deployment scripts and source
guards will be updated to preserve this boundary after the submodule moves to
`reference`.

## Validation

- Build and run the EP source-only test suite.
- Run shell syntax checks for changed shell scripts.
- Verify the reference downloader lists and dry-runs `ops-transformer` at
  `reference/ops-transformer`.
- Verify no tracked `examples/mc2` files or MC2-only scripts remain.
- Scan active code and current documentation for stale MC2 paths, commands, and
  `3rdparty/ops-transformer` references.
- Run `git diff --check` before commit and PR creation.
