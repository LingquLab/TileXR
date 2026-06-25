#!/bin/bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.T560
export PATH=/usr/local/Ascend/cann-9.1.T560/bin:/usr/local/Ascend/cann-9.1.T560/aarch64-linux/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.T560/lib64:/usr/local/Ascend/cann-9.1.T560/aarch64-linux/lib64:/usr/local/Ascend/cann-9.1.T560/runtime/lib64:/usr/local/Ascend/cann-9.1.T560/aarch64-linux/devlib:${LD_LIBRARY_PATH:-}
export TILEXR_DEMO_DEVICES=2,3,6,7
export TILEXR_DEMO_ALLTOALL_PLAIN_IPC=1
export TILEXR_DEMO_ALLTOALL_REPEAT=30
export TILEXR_DEMO_ALLTOALL_SYNC_AT_END=1
unset TILEXR_DEMO_ALLTOALL_USE_UDMA
export TILEXR_COMM_ID=127.0.0.1:12891
cd /home/tileXR-new/tests/udma
exec /usr/bin/bash demo/run_tilexr_udma_demo.sh 2 4 524288 4 0
