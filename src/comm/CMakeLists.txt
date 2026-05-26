#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

# shmem 依赖配置
set(SHMEM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/shmem")

set(TILEXR_SOURCE_FILE tilexr_comm.cpp
        comm_wrap.cpp
        tilexr_internal.cpp
        tilexr_internal.h
        tools/socket/tilexr_sock_exchange.h
        tools/socket/tilexr_sock_exchange.cpp
)

add_library(tile-comm SHARED ${TILEXR_SOURCE_FILE})

# 查找 shmem host 库
find_library(SHMEM_HOST_LIB shmem
    HINTS "${SHMEM_ROOT}/install/shmem/lib" "${SHMEM_ROOT}/build/lib"
    REQUIRED)

target_include_directories(tile-comm
        PUBLIC
        ${ASCEND_DRIVER_PATH}/kernel/inc
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty
        ${SHMEM_ROOT}/install/shmem/include
)
target_link_directories(tile-comm
        PUBLIC
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

target_link_libraries(tile-comm ascendcl runtime profapi ascend_hal ${SHMEM_HOST_LIB})

install(TARGETS tile-comm LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
