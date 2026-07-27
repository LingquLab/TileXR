# AGENTS.md

Ask the user to install missing `ascendc-development` or `superpowers-neo` skills from [LingquLab/skills](https://github.com/LingquLab/skills).

## Project

TileXR is a C++14 communication runtime for Huawei Ascend NPUs. It provides tile-level synchronization, optional collectives and EP libraries, IPC/MTE communication, registered-memory UDMA, and opt-in local SDMA.

- Current build target: CANN 9.1.0 and NPU driver 25.5.0 or later.
- Core runtime targets Ascend 910B and 910A5. UDMA data-plane validation requires A5 / Ascend950 / 950 hardware.

## Key Paths

```text
src/comm/          Core runtime and IPC, UDMA, and SDMA transports
src/collectives/   Optional collectives library
src/ep/            Optional expert-parallel dispatch/combine library
src/include/       Public host and device headers
tests/             Module, source, integration, and hardware tests
scripts/           Environment, build, test, and utility scripts
reference/         Ignored comparison-only upstream checkouts
docs/              Architecture and validation guides
```

## Build and Test

Initialize submodules and source the environment before building or testing:

```bash
git submodule update --init --recursive
source scripts/common_env.sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build -j"$(nproc)"
cmake --install build
```

Optional CMake switches are `TILEXR_BUILD_COLLECTIVES`, `TILEXR_BUILD_EP`, `TILEXR_BUILD_TESTS`, and `TILEXR_BUILD_CHECKER`. Run focused tests for the changed module; use [docs/BUILD_VERIFICATION.md](docs/BUILD_VERIFICATION.md) for build and hardware validation details. The complete script catalog is in [scripts/README.md](scripts/README.md).

## Architecture

- `src/comm` builds `libtile-comm.so`, owns communicator setup, peer mappings, capability flags, and `CommArgs`, and exposes the core API through `tilexr_api.h`.
- Device synchronization uses reusable magic-tagged flags. Use `TileXRCommNextMagic` for new rounds rather than resetting shared flag memory.
- `src/collectives` builds `libtilexr-collectives.so` when enabled. Multi-rank AllToAll requires the supported `TOPO_910_93` topology.
- `src/ep` builds standalone dispatch/combine host and kernel libraries. Same-node traffic uses IPC peer windows; cross-node traffic uses registered UDMA workspaces.
- UDMA is a registered remote-memory transport for A5 / Ascend950. SDMA is an opt-in same-device GM copy transport enabled with `TILEXR_ENABLE_SDMA=1`.

## Critical Constraints

- Preserve C++14 and CANN 9.1 compatibility unless the task explicitly changes them.
- Treat `reference/` as comparison-only; active targets must not include or link sources from it.
- UDMA and SDMA are best-effort capabilities. Communicator initialization must preserve existing paths when either is unavailable.
- `TileXRUDMARegister` is unsupported in `InitThread`; UDMA targets must be registered ordinary device memory, not `peerMems[]`.
- Never put `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` in runtime RPATH/RUNPATH; runtime must resolve the real driver HAL.
- Host, simulator, and 910B fallback tests do not prove UDMA data-plane transfer. Keep validation claims scoped to the hardware actually exercised.
