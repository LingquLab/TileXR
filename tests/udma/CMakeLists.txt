#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# CMakeLists.txt for UDMA tests
#

cmake_minimum_required(VERSION 3.16)
project(TileXR_UDMA_Tests)

# 使用 C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 获取环境变量
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ARCH $ENV{ARCH})
if(NOT ARCH)
    set(ARCH "x86_64")
endif()

set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
if(NOT ASCEND_DRIVER_PATH)
    set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver")
endif()

# shmem 路径
set(SHMEM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/shmem")

# TileXR 路径
set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

message(STATUS "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
message(STATUS "ARCH: ${ARCH}")
message(STATUS "SHMEM_ROOT: ${SHMEM_ROOT}")
message(STATUS "TILEXR_ROOT: ${TILEXR_ROOT}")

# 查找库
find_library(SHMEM_HOST_LIB shmem
    HINTS "${SHMEM_ROOT}/install/shmem/lib" "${SHMEM_ROOT}/build/lib"
    REQUIRED)

find_library(TILEXR_LIB tile-comm
    HINTS "${TILEXR_ROOT}/install/lib" "${TILEXR_ROOT}/build/src/comm"
    REQUIRED)

message(STATUS "Found shmem: ${SHMEM_HOST_LIB}")
message(STATUS "Found tile-comm: ${TILEXR_LIB}")

# 包含目录
include_directories(
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${SHMEM_ROOT}/install/shmem/include
    ${TILEXR_ROOT}/src/include
)

# 链接目录
link_directories(
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
    ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

# 单元测试：shmem API
add_executable(test_shmem_api
    unit/test_shmem_api.cpp
)

target_link_libraries(test_shmem_api
    ${SHMEM_HOST_LIB}
    ascendcl
    runtime
    ascend_hal
)

# 集成测试：TileXR UDMA
add_executable(test_tilexr_udma
    integration/test_tilexr_udma.cpp
)

target_link_libraries(test_tilexr_udma
    ${TILEXR_LIB}
    ${SHMEM_HOST_LIB}
    ascendcl
    runtime
    ascend_hal
)

# 安装目标
install(TARGETS test_shmem_api test_tilexr_udma
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
