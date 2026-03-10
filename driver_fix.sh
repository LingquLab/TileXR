#!/usr/bin/env bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

for ((i=0; i<ASCEND_DEV_NUM; i++)); do

yes | npu-smi set -t custom-op-secverify-enable -i ${i} -d 1
npu-smi set -t custom-op-secverify-mode -i ${i} -d 0

done
