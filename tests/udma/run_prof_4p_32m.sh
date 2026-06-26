#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
TILEXR_ROOT=/home/tileXR-new
UDMA_DIR=${TILEXR_ROOT}/tests/udma
export LD_LIBRARY_PATH=${UDMA_DIR}/install/lib:${UDMA_DIR}/install/lib64:${TILEXR_ROOT}/install/lib:${TILEXR_ROOT}/install/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH}
export TILEXR_DEMO_ALLTOALL_USE_UDMA=1
export TILEXR_DEMO_ALLTOALL_REPEAT=1

RANK_SIZE=4
ELEM=8388608
BIN=${UDMA_DIR}/install/bin/tilexr_udma_demo

pids=()
for rank in $(seq 0 $((RANK_SIZE-1))); do
  RANK=${rank} RANK_SIZE=${RANK_SIZE} "${BIN}" "${RANK_SIZE}" "${rank}" 2 "${ELEM}" "${RANK_SIZE}" 0 \
    > /tmp/a2a32m_rank${rank}.log 2>&1 &
  pids+=("$!")
done
ret=0
for idx in "${!pids[@]}"; do
  wait "${pids[$idx]}" && echo "rank ${idx} ok" || { echo "rank ${idx} FAIL $?"; ret=1; }
done
echo "=== pass/kernel lines ==="
for rank in $(seq 0 $((RANK_SIZE-1))); do grep -E "chunk plan|launch all-to-all kernel pass|success" /tmp/a2a32m_rank${rank}.log 2>/dev/null | tail -6; done
exit ${ret}