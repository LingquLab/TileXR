# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.12)
project(OpSimulator LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

option(USE_CXX11_ABI "USE_CXX11_ABI" 0)

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

message("== CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")
message("== CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")

# 获取环境变量
if (DEFINED ENV{ASCEND_HOME_PATH})
    message("ASCEND_HOME_PATH is set by env")
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else ()
    message("ASCEND_HOME_PATH not set! using default path!")
    set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif ()
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

message("== ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
message("== ASCEND_DRIVER_PATH: ${ASCEND_DRIVER_PATH}")
message("== ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/op")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
add_link_options(-Wl,-z,relro,-z,now)
add_link_options(-s)

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/include/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/msprof/
        ${CMAKE_SOURCE_DIR}/src/3rdparty/spdlog/include
        ${CMAKE_SOURCE_DIR}/src/3rdparty/
)
link_directories(
        ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib/linux/${ARCH}/
)

list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/ascendc_kernel_cmake/asc_modules")

add_subdirectory(src)
