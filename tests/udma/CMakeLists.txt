#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# CMakeLists.txt for UDMA tests
#

cmake_minimum_required(VERSION 3.16)
project(TileXR_UDMA_Tests)

# 使用 C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_TILEXR_UDMA_DEMO "Build TileXR UDMA communication demo with Ascend C kernel" ON)
set(TILEXR_UDMA_DEMO_SOC_TYPE "Ascend950" CACHE STRING "SOC type used for the TileXR UDMA demo kernel")
set_property(CACHE TILEXR_UDMA_DEMO_SOC_TYPE PROPERTY STRINGS
    Ascend950 Ascend950PR Ascend950DT Ascend910B)

# 获取环境变量
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ARCH $ENV{ARCH})
if(NOT ARCH)
    set(ARCH $ENV{TILEXR_OS_ARCH})
endif()
if(NOT ARCH)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(ARCH STREQUAL "arm64")
        set(ARCH "aarch64")
    endif()
endif()

set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
if(NOT ASCEND_DRIVER_PATH)
    set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver")
endif()

# TileXR 路径
set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

message(STATUS "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
message(STATUS "ARCH: ${ARCH}")
message(STATUS "TILEXR_ROOT: ${TILEXR_ROOT}")

# 查找库
find_library(TILEXR_LIB tile-comm
    HINTS "${TILEXR_ROOT}/install/lib" "${TILEXR_ROOT}/install/lib64" "${TILEXR_ROOT}/build/src/comm"
    REQUIRED)

message(STATUS "Found tile-comm: ${TILEXR_LIB}")

# 包含目录
include_directories(
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${TILEXR_ROOT}/3rdparty
    ${TILEXR_ROOT}/src/include
)

# 链接目录
link_directories(
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
    ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

add_executable(test_tilexr_udma_registry
    unit/test_tilexr_udma_registry.cpp
)

target_include_directories(test_tilexr_udma_registry PRIVATE
    ${TILEXR_ROOT}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(test_tilexr_udma_alltoall_layout
    unit/test_tilexr_udma_alltoall_layout.cpp
)

target_include_directories(test_tilexr_udma_alltoall_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(test_tilexr_udma_alltoall_layout PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_executable(test_tilexr_udma_alltoall_group_layout
    unit/test_tilexr_udma_alltoall_group_layout.cpp
)

target_include_directories(test_tilexr_udma_alltoall_group_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(test_tilexr_udma_alltoall_group_layout PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_executable(test_tilexr_udma_allreduce_layout
    unit/test_tilexr_udma_allreduce_layout.cpp
)

target_include_directories(test_tilexr_udma_allreduce_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(test_tilexr_udma_allreduce_layout PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_executable(test_tilexr_chip_map_sources
    unit/test_tilexr_chip_map_sources.cpp
)

target_compile_definitions(test_tilexr_chip_map_sources PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_executable(test_tilexr_ipc_pid_mode_sources
    unit/test_tilexr_ipc_pid_mode_sources.cpp
)

target_compile_definitions(test_tilexr_ipc_pid_mode_sources PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_executable(test_tilexr_udma_transport_layout
    unit/test_tilexr_udma_transport_layout.cpp
    ${TILEXR_ROOT}/src/comm/udma/tilexr_udma_layout.cpp
)

add_executable(test_tilexr_sock_exchange_layout
    unit/test_tilexr_sock_exchange_layout.cpp
)

target_include_directories(test_tilexr_sock_exchange_layout PRIVATE
    ${TILEXR_ROOT}/src/comm
)

target_include_directories(test_tilexr_udma_transport_layout PRIVATE
    ${TILEXR_ROOT}/src/include
    ${TILEXR_ROOT}/src/comm
)

target_compile_definitions(test_tilexr_udma_transport_layout PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

# 集成测试：TileXR UDMA
add_executable(test_tilexr_udma
    integration/test_tilexr_udma.cpp
)

target_link_libraries(test_tilexr_udma
    ${TILEXR_LIB}
    ascendcl
    runtime
    ascend_hal
)

set(INSTALL_TARGETS
    test_tilexr_udma
    test_tilexr_udma_registry
    test_tilexr_udma_alltoall_layout
    test_tilexr_udma_alltoall_group_layout
    test_tilexr_udma_allreduce_layout
    test_tilexr_chip_map_sources
    test_tilexr_ipc_pid_mode_sources
    test_tilexr_udma_transport_layout
    test_tilexr_sock_exchange_layout
)

if(BUILD_TILEXR_UDMA_DEMO)
    find_program(BISHENG_EXECUTABLE bisheng)
    if(NOT BISHENG_EXECUTABLE)
        message(WARNING "bisheng not found; skip tilexr_udma_demo")
    else()
        if(TILEXR_UDMA_DEMO_SOC_TYPE MATCHES "^Ascend950(PR|DT)?$")
            set(TILEXR_UDMA_NPU_ARCH "dav-3510")
            set(TILEXR_UDMA_AICORE_ARCH "--cce-aicore-arch=dav-c310-vec")
            set(TILEXR_UDMA_CATLASS_ARCH "3510")
        elseif(TILEXR_UDMA_DEMO_SOC_TYPE STREQUAL "Ascend910B")
            set(TILEXR_UDMA_NPU_ARCH "dav-2201")
            set(TILEXR_UDMA_AICORE_ARCH "--cce-aicore-arch=dav-c220-vec")
            set(TILEXR_UDMA_CATLASS_ARCH "2201")
        else()
            message(FATAL_ERROR
                "Unsupported TILEXR_UDMA_DEMO_SOC_TYPE: ${TILEXR_UDMA_DEMO_SOC_TYPE}")
        endif()

        execute_process(
            COMMAND ${BISHENG_EXECUTABLE} -v
            OUTPUT_VARIABLE BISHENG_VERSION_OUTPUT
            ERROR_VARIABLE BISHENG_VERSION_OUTPUT
            RESULT_VARIABLE BISHENG_VERSION_RESULT
        )
        if(BISHENG_VERSION_RESULT EQUAL 0 AND BISHENG_VERSION_OUTPUT MATCHES "([0-9]{8})")
            set(TILEXR_UDMA_BISHENG_DATE "${CMAKE_MATCH_1}")
        else()
            set(TILEXR_UDMA_BISHENG_DATE "0")
        endif()
        if(TILEXR_UDMA_BISHENG_DATE GREATER_EQUAL 20250428)
            set(TILEXR_UDMA_KERNEL_COMPILE_OPTIONS
                -xasc
                -Xhost-start -ftrapv -Xhost-end
                --npu-arch=${TILEXR_UDMA_NPU_ARCH}
                --cce-auto-infer-kernel-type=false
            )
            set(TILEXR_UDMA_KERNEL_LINK_OPTIONS --cce-fatobj-link)
        else()
            set(TILEXR_UDMA_KERNEL_COMPILE_OPTIONS
                -xcce
                -Xhost-start -ftrapv -Xhost-end
                -mllvm -cce-aicore-stack-size=0x8000
                -mllvm -cce-aicore-function-stack-size=0x8000
                -mllvm -cce-aicore-record-overflow=true
                -mllvm -cce-aicore-addr-transform
                -mllvm -cce-aicore-dcci-insert-for-scalar=false
                ${TILEXR_UDMA_AICORE_ARCH}
            )
            set(TILEXR_UDMA_KERNEL_LINK_OPTIONS --cce-fatobj-link ${TILEXR_UDMA_AICORE_ARCH})
        endif()

        set(TILEXR_UDMA_DEMO_KERNEL_SO "${CMAKE_CURRENT_BINARY_DIR}/libtilexr_udma_demo_kernel.so")
        set(TILEXR_UDMA_DEMO_KERNEL_INCLUDES
            -I${ASCEND_HOME_PATH}/compiler/tikcpp
            -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
            -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
            -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/impl
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/interface
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/include/
            -I${ASCEND_HOME_PATH}/${ARCH}-linux/asc/include/
            -I${ASCEND_DRIVER_PATH}/kernel/inc
            -I${TILEXR_ROOT}/3rdparty
            -I${TILEXR_ROOT}/src/include
        )
        add_custom_command(
            OUTPUT "${TILEXR_UDMA_DEMO_KERNEL_SO}"
            COMMAND ${BISHENG_EXECUTABLE}
                ${TILEXR_UDMA_KERNEL_COMPILE_OPTIONS}
                -std=gnu++17
                -fPIC
                -shared
                ${TILEXR_UDMA_KERNEL_LINK_OPTIONS}
                -DCATLASS_ARCH=${TILEXR_UDMA_CATLASS_ARCH}
                -DTILEXR_UDMA_FORCE_ENABLE=1
                ${TILEXR_UDMA_DEMO_KERNEL_INCLUDES}
                "${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_udma_demo_kernel.cpp"
                -L${ASCEND_DRIVER_PATH}/lib64/driver
                -L${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
                -L${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
                -lruntime
                -lascendcl
                -lstdc++
                -lm
                -ltiling_api
                -lplatform
                -lc_sec
                -ldl
                -lnnopbase
                -lpthread
                -o "${TILEXR_UDMA_DEMO_KERNEL_SO}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_udma_demo_kernel.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_udma_alltoall_constants.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_udma_fullmesh_trace.h"
                "${TILEXR_ROOT}/src/include/tilexr_udma.h"
                "${TILEXR_ROOT}/src/include/tilexr_data_as_flag.h"
            VERBATIM
            COMMENT "Building TileXR UDMA demo kernel with bisheng"
        )
        add_custom_target(tilexr_udma_demo_kernel ALL DEPENDS "${TILEXR_UDMA_DEMO_KERNEL_SO}")

        list(APPEND CMAKE_MODULE_PATH "${TILEXR_ROOT}/src/collectives/kernels/cmake")
        enable_language(CCE)
        set(TILEXR_GROUPED_ALLTOALL_KERNEL_SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/demo/tilexr_udma_alltoall_group_kernel.cpp")
        set_source_files_properties(
            "${TILEXR_GROUPED_ALLTOALL_KERNEL_SOURCE}" PROPERTIES LANGUAGE CCE)
        add_library(tilexr_udma_alltoall_group_device OBJECT
            "${TILEXR_GROUPED_ALLTOALL_KERNEL_SOURCE}")
        target_compile_options(tilexr_udma_alltoall_group_device PRIVATE
            -O2
            -std=gnu++17
            --cce-aicore-only
            -Wno-deprecated-declarations
            "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
            "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
            "SHELL:-mllvm -cce-aicore-record-overflow=true"
            "SHELL:-mllvm -cce-aicore-addr-transform"
            "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
            ${TILEXR_UDMA_AICORE_ARCH})
        target_compile_definitions(tilexr_udma_alltoall_group_device PRIVATE
            CATLASS_ARCH=${TILEXR_UDMA_CATLASS_ARCH}
            TILEXR_UDMA_FORCE_ENABLE=1)
        target_include_directories(tilexr_udma_alltoall_group_device PRIVATE
            ${ASCEND_HOME_PATH}/compiler/tikcpp
            ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
            ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
            ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
            ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp
            ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw
            ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/impl
            ${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/interface
            ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
            ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
            ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
            ${ASCEND_HOME_PATH}/${ARCH}-linux/asc/include/
            ${ASCEND_DRIVER_PATH}/kernel/inc
            ${TILEXR_ROOT}/3rdparty
            ${TILEXR_ROOT}/src/include
            ${CMAKE_CURRENT_SOURCE_DIR}/demo)

        set(TILEXR_GROUPED_ALLTOALL_KERNEL_OBJECT
            "${CMAKE_CURRENT_BINARY_DIR}/tilexr_udma_alltoall_group_kernel.o")
        add_custom_command(
            OUTPUT "${TILEXR_GROUPED_ALLTOALL_KERNEL_OBJECT}"
            COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
                $<TARGET_OBJECTS:tilexr_udma_alltoall_group_device>
                --static -o "${TILEXR_GROUPED_ALLTOALL_KERNEL_OBJECT}"
                --allow-multiple-definition
            COMMAND truncate -c -s 8388608
                "${TILEXR_GROUPED_ALLTOALL_KERNEL_OBJECT}"
            DEPENDS tilexr_udma_alltoall_group_device
            COMMAND_EXPAND_LISTS
            VERBATIM)
        add_custom_target(tilexr_udma_alltoall_group_device_binary
            DEPENDS "${TILEXR_GROUPED_ALLTOALL_KERNEL_OBJECT}")

        set(TILEXR_GROUPED_ALLTOALL_KERNEL_EMBED_CPP
            "${CMAKE_CURRENT_BINARY_DIR}/tilexr_udma_alltoall_group_kernel_embed.cpp")
        add_custom_command(
            OUTPUT "${TILEXR_GROUPED_ALLTOALL_KERNEL_EMBED_CPP}"
            COMMAND ${CMAKE_COMMAND}
                -DTILEXR_GROUPED_ALLTOALL_KERNEL_OBJECT=${TILEXR_GROUPED_ALLTOALL_KERNEL_OBJECT}
                -DTILEXR_GROUPED_ALLTOALL_KERNEL_EMBED_CPP=${TILEXR_GROUPED_ALLTOALL_KERNEL_EMBED_CPP}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_grouped_alltoall_kernel.cmake"
            DEPENDS tilexr_udma_alltoall_group_device_binary
                "${TILEXR_GROUPED_ALLTOALL_KERNEL_OBJECT}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_grouped_alltoall_kernel.cmake"
            VERBATIM)

        add_library(tilexr_udma_alltoall_group_kernel SHARED
            demo/tilexr_udma_alltoall_group_launcher.cpp
            "${TILEXR_GROUPED_ALLTOALL_KERNEL_EMBED_CPP}")
        add_dependencies(tilexr_udma_alltoall_group_kernel
            tilexr_udma_alltoall_group_device_binary)
        target_link_libraries(tilexr_udma_alltoall_group_kernel PRIVATE runtime)
        set_target_properties(tilexr_udma_alltoall_group_kernel PROPERTIES
            OUTPUT_NAME tilexr_udma_alltoall_group_kernel)

        add_executable(tilexr_udma_demo
            demo/tilexr_udma_demo.cpp
        )
        add_dependencies(tilexr_udma_demo tilexr_udma_demo_kernel)
        target_link_directories(tilexr_udma_demo PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(tilexr_udma_demo
            "${TILEXR_UDMA_DEMO_KERNEL_SO}"
            tilexr_udma_alltoall_group_kernel
            ${TILEXR_LIB}
            ascendcl
            runtime
            ascend_hal
        )
        list(APPEND INSTALL_TARGETS
            tilexr_udma_demo tilexr_udma_alltoall_group_kernel)
        install(FILES "${TILEXR_UDMA_DEMO_KERNEL_SO}" DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        message(STATUS "TileXR UDMA demo enabled with ${BISHENG_EXECUTABLE}, SOC=${TILEXR_UDMA_DEMO_SOC_TYPE}")
    endif()
endif()

# 安装目标
install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
