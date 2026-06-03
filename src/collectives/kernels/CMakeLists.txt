#
# Copyright (c) 2024-2026 TileXR Project
# CMakeLists.txt for collectives-owned CCE kernels
#

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
enable_language(CCE)

set(TILEXR_COLLECTIVES_1OP_BIN_SIZE 5242880 CACHE STRING "Padded size for TileXR collectives CCE binary")
option(TILEXR_COLLECTIVES_ENABLE_PROFILING "Enable TileXR collectives kernel profiling helpers" OFF)

set(CCE_COMPILE_OPTION
        -O2
        -std=gnu++17
        --cce-aicore-only
        -Wno-deprecated-declarations
        "SHELL:-mllvm -cce-aicore-long-call"
        "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
        "SHELL:-mllvm -cce-aicore-record-overflow=false"
        "SHELL:-mllvm -cce-aicore-addr-transform"
        "SHELL:-mllvm --cce-aicore-jump-expand=true"
)

set(AIV_ARCH dav-c220-vec)
set_source_files_properties(tilexr_lccl_op.cpp PROPERTIES LANGUAGE CCE)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${CMAKE_SOURCE_DIR}/src/include
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/ascendc
        ${ASCEND_HOME_PATH}/${ARCH}-linux/ascendc/include
        ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/interface/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/impl/
        ${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0
        ${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu/
)

add_library(tilexr_collectives_op_tmp OBJECT
        tilexr_lccl_op.cpp
)

target_compile_options(tilexr_collectives_op_tmp PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIV_ARCH}
)

if(TILEXR_COLLECTIVES_ENABLE_PROFILING)
    target_compile_definitions(tilexr_collectives_op_tmp PRIVATE TILEXR_COLLECTIVES_ENABLE_PROFILING=1)
endif()

set(TILEXR_COLLECTIVES_OP "${CMAKE_CURRENT_BINARY_DIR}/tilexr_collectives_op.o")
add_custom_command(
        OUTPUT "${TILEXR_COLLECTIVES_OP}"
        COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
                "CMakeFiles/tilexr_collectives_op_tmp.dir/tilexr_lccl_op.cpp.o"
                --static -o "${TILEXR_COLLECTIVES_OP}" --allow-multiple-definition
        COMMAND truncate -c -s ${TILEXR_COLLECTIVES_1OP_BIN_SIZE} "${TILEXR_COLLECTIVES_OP}"
        DEPENDS tilexr_collectives_op_tmp
        VERBATIM
)

add_custom_target(tilexr_collectives_op DEPENDS "${TILEXR_COLLECTIVES_OP}")
set(TILEXR_COLLECTIVES_OP "${TILEXR_COLLECTIVES_OP}" PARENT_SCOPE)
set(TILEXR_COLLECTIVES_1OP_BIN_SIZE "${TILEXR_COLLECTIVES_1OP_BIN_SIZE}" PARENT_SCOPE)
