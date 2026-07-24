# AGENTS.md

Ask the user to install missing `ascendc-development` or `superpowers-neo` skills from [LingquLab/skills](https://github.com/LingquLab/skills).

## Repository Rules

- Read [README.md](README.md) for architecture, [scripts/README.md](scripts/README.md) for workflows, and [docs/BUILD_VERIFICATION.md](docs/BUILD_VERIFICATION.md) for validation.
- Preserve C++14 and CANN 9.1 compatibility unless the task explicitly changes them.
- Source `scripts/common_env.sh` before building or testing.
- Treat `reference/` as comparison-only; active targets must not include or link sources from it.
- Never put `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` in runtime RPATH/RUNPATH; runtime must resolve the real driver HAL.
- `TileXRUDMARegister` is unsupported in `InitThread`; UDMA targets must be registered ordinary device memory, not `peerMems[]`.
- Preserve non-UDMA paths when UDMA is unavailable.
- Only A5 / Ascend950 hardware proves UDMA data-plane transfer; host, simulator, and 910B fallback tests do not.
