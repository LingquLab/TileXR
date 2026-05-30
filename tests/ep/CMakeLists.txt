#
# Copyright (c) 2024-2026 TileXR Project
# CMakeLists.txt for TileXR EP tests
#

cmake_minimum_required(VERSION 3.16)
project(TileXR_EP_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

option(BUILD_TILEXR_EP_DEMO "Build TileXR EP demo with Ascend C kernel" ON)

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

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

set(TILEXR_EP_ASCEND_INCLUDE_DIRS)
if(ASCEND_HOME_PATH)
    foreach(dir
        "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc"
        "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime"
        "${ASCEND_HOME_PATH}/${ARCH}-linux/include"
    )
        if(EXISTS "${dir}")
            list(APPEND TILEXR_EP_ASCEND_INCLUDE_DIRS "${dir}")
        endif()
    endforeach()
endif()

add_executable(test_tilexr_ep_layout
    unit/test_tilexr_ep_layout.cpp
    ${TILEXR_ROOT}/src/ep/host/ep_layout.cpp
)

target_include_directories(test_tilexr_ep_layout PRIVATE
    ${TILEXR_ROOT}/src/ep/host
    ${TILEXR_ROOT}/src/ep/common
    ${TILEXR_ROOT}/src/include
    ${TILEXR_EP_ASCEND_INCLUDE_DIRS}
)

add_test(NAME test_tilexr_ep_layout COMMAND test_tilexr_ep_layout)

install(TARGETS
    test_tilexr_ep_layout
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
