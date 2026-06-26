#!/bin/bash
#
# Run TileXR P2P perf across transports, traffic modes, and blockDim values.
#

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UDMA_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

min_bytes=${1:-4096}
max_bytes=${2:-16777216}
step_factor=${3:-2}
iters=${4:-20}
warmup_iters=${5:-5}
first_npu=${6:-2}
check=${7:-1}
transports_csv=${8:-direct_urma,memory,memory_consume,data_as_flag}
traffic_csv=${9:-unidir,bidir}
block_dims_csv=${10:-1,2,4,8}

IFS=',' read -r -a transports <<<"${transports_csv}"
IFS=',' read -r -a traffics <<<"${traffic_csv}"
IFS=',' read -r -a block_dims <<<"${block_dims_csv}"

base_port=${TILEXR_P2P_SWEEP_BASE_PORT:-13600}
run_index=0

echo "=========================================="
echo "  TileXR P2P Concurrency Sweep"
echo "=========================================="
echo "Range:       ${min_bytes} -> ${max_bytes}, step=${step_factor}"
echo "Iters:       ${iters}, warmup=${warmup_iters}"
echo "First NPU:   ${first_npu}"
echo "Check:       ${check}"
echo "Transports:  ${transports_csv}"
echo "Traffic:     ${traffic_csv}"
echo "Block dims:  ${block_dims_csv}"
echo "=========================================="

cd "${UDMA_DIR}"
for transport in "${transports[@]}"; do
    for traffic in "${traffics[@]}"; do
        for block_dim in "${block_dims[@]}"; do
            port=$((base_port + run_index))
            export TILEXR_COMM_ID="127.0.0.1:${port}"
            echo "==== transport=${transport} traffic=${traffic} block_dim=${block_dim} port=${port} ===="
            bash demo/run_tilexr_udma_p2p_perf.sh \
                0 1 "${min_bytes}" "${max_bytes}" "${step_factor}" "${iters}" "${warmup_iters}" \
                "${first_npu}" "${check}" "${transport}" "${block_dim}" "${traffic}"
            run_index=$((run_index + 1))
        done
    done
done
