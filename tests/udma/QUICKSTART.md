# TileXR UDMA Quickstart

Use this when you need to validate TileXR UDMA on real Ascend hardware.

UDMA here means UnifiedBus DMA. Runtime validation requires A5 / Ascend950 / 950 hardware. Do not treat 910B or 310P runs as valid UDMA validation.

## 1. Check Hardware

```bash
cd /path/to/TileXR
source scripts/common_env.sh
npu-smi info
```

Confirm the machine has usable Ascend950-class devices. Prefer at least two devices for the demo.

## 2. Build

```bash
cd /path/to/TileXR
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-udma -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-udma --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build-udma

cd tests/udma
bash build.sh
```

Expected:

```bash
test -x install/bin/test_tilexr_udma_registry
test -x install/bin/test_tilexr_udma
test -x install/bin/tilexr_udma_demo
```

## 3. Run Smoke Checks

```bash
cd /path/to/TileXR/tests/udma
./install/bin/test_tilexr_udma_registry

source ../../scripts/common_env.sh
export LD_LIBRARY_PATH="$PWD/install/lib:../../install/lib:${LD_LIBRARY_PATH:-}"
RANK=0 RANK_SIZE=1 ./install/bin/test_tilexr_udma
```

Expected:

- `TileXR UDMA registry checks passed`;
- `test_tilexr_udma` exits 0 and prints `Failed: 0`.

## 4. Run UDMA Data-Plane Demo

All-gather style put:

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
```

Put with signal:

```bash
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

Expected:

- both commands exit 0;
- every rank log contains `TileXR UDMA demo success`;
- every rank log contains `TileXRUDMARegister success`;
- all-gather samples include `seg0=1000 seg1=1001`;
- put-signal logs show non-local signal values equal to `1000`.

Quick check:

```bash
latest=$(ls -td logs/tilexr_udma_demo_* | head -n1)
grep -R "TileXR UDMA demo success" "$latest"
grep -R "TileXRUDMARegister success" "$latest"
grep -R "DATA MISMATCH\\|expected non-local signals\\|TileXR UDMA demo failed\\|ERROR" "$latest" || true
```

The final grep should print nothing unless the run failed.

## Report Template

```text
Machine:
NPU model:
NPU count:
CANN version:
Driver version:
TileXR commit:

Build result:
Registry test result:
TileXR smoke test result:
All-gather demo result:
Put-signal demo result:
Log directory:

Any ERROR lines:
Any DATA MISMATCH lines:
Any signal mismatch lines:
```
