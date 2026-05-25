# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

project(tilexr LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

add_link_options(-Wl,-z,relro,-z,now)
add_link_options(-s)
# 获取环境变量
IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()
if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else ()
    message("ASCEND_HOME_PATH not set! using default path!")
    set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif ()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

message("== ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
message("== ASCEND_DRIVER_PATH: ${ASCEND_DRIVER_PATH}")

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/src/include/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/msprof/
        ${CMAKE_SOURCE_DIR}/3rdparty/spdlog/include
        ${CMAKE_SOURCE_DIR}/3rdparty/
)
add_subdirectory(src/comm)
