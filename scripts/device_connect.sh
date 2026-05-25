#!/usr/bin/env bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

DEVICE_IF=`ifconfig -a endvnic | grep inet`
if [ "${DEVICE_IF}" == "" ]; then
    ifconfig endvnic 192.168.1.100
fi	

DEVICE_ID=${1:-0}
DEVICE_IP="192.168.1.$((199 - $DEVICE_ID))"

line
success "connect to device ${DEVICE_ID} with ip ${DEVICE_IP}"
line

sshpass -p Huawei2012# ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null HwHiAiUser@${DEVICE_IP}
