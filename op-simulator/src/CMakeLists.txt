# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

enable_language(ASC)
set(ASC_COMPILE_OPTION
    -O2 
    -std=c++17
    --cce-aicore-only
    -Wno-deprecated-declarations
    "SHELL:-mllvm -cce-aicore-long-call"
    "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm --cce-aicore-jump-expand=true"
)

set(CCE_INCLUDE_BASE ${ASCEND_HOME_PATH}/${ARCH}-linux)
include_directories(
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/interface/
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/impl/
    ${CCE_INCLUDE_BASE}/pkg_inc/
    ${CCE_INCLUDE_BASE}/pkg_inc/runtime/
    ${CCE_INCLUDE_BASE}/pkg_inc/runtime/runtime/
    .
)

file(GLOB_RECURSE KERNEL_FILES *.cpp)
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE ASC)

add_library(kernel_demo OBJECT ${KERNEL_FILES})

target_compile_options(kernel_demo PRIVATE
        ${ASC_COMPILE_OPTION}
        --cce-aicore-arch=dav-c220
        -Wno-c99-designator
        -Wno-unused-command-line-argument
)

set(OP_OUTPUT_PATH "${CMAKE_INSTALL_PREFIX}/my_kernel.o")

set(CCE_LLD "${ASCEND_HOME_PATH}/${ARCH}-linux/bin/ld.lld")

# 3. 添加自定义命令来执行链接操作
add_custom_command(
    OUTPUT ${OP_OUTPUT_PATH}
    # 依赖 kernel_demo 这个目标产生的对象文件
    DEPENDS kernel_demo
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_INSTALL_PREFIX}"
    COMMAND ${CCE_LLD} 
            -m aicorelinux 
            -Ttext 0 
            -static
            -o ${OP_OUTPUT_PATH}
            $<TARGET_OBJECTS:kernel_demo>
    COMMENT "Linking Ascend AI Core kernel: ${OP_OUTPUT_PATH}"
    VERBATIM
)

add_custom_target(generate_kernel ALL DEPENDS ${OP_OUTPUT_PATH})
