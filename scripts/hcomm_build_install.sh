#!/bin/bash

script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh

cp ${TILEXR_HOME}/.gitignore ${TILEXR_HCOMM_HOME}/.gitignore

cd ${TILEXR_HCOMM_HOME}
_hcomm_build --noclean
if [ $? -ne 0 ]; then exit 1; fi
cd ${TILEXR_HOME}

bash ${script_path}/hcomm_local_install.sh
