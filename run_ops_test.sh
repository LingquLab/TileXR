script_path=`realpath $(dirname "${BASH_SOURCE[0]}")`
source ${script_path}/common_env.sh
ops=${1:-all_gather_matmul}
export PATH=${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:/usr/local/lib:$LD_LIBRARY_PATH
export HCCL_SOCKET_IFNAME=enp189s0f0
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export FIRST_RANK_ID=0
export TILEXR_COMM_ID=127.0.0.1:10067
mpirun -np 2 -host 10.118.241.6 -bind-to none -map-by node ${TILEXR_OPS_HOME}/build/test_aclnn_${ops}