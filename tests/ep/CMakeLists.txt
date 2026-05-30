#
# Copyright (c) 2024-2026 TileXR Project
# CMakeLists.txt for TileXR EP tests
#

cmake_minimum_required(VERSION 3.16)
project(TileXR_EP_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)
enable_testing()

option(BUILD_TILEXR_EP_DEMO "Build TileXR EP dispatch hardware demo" OFF)

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(TILEXR_INSTALL_PREFIX "${TILEXR_ROOT}/install" CACHE PATH "TileXR install prefix")
set(TILEXR_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "TileXR install library directory")
set(TILEXR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE STRING "TileXR install include directory")

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

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

set(TILEXR_EP_TEST_INCLUDE_DIRS
    ${TILEXR_ROOT}/src/include
    ${TILEXR_ROOT}/src/ep/common
    ${TILEXR_ROOT}/src/ep/host
)
foreach(dir
    "${ASCEND_DRIVER_PATH}/kernel/inc"
    "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc"
    "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/${ARCH}-linux/include"
)
    if(EXISTS "${dir}")
        list(APPEND TILEXR_EP_TEST_INCLUDE_DIRS "${dir}")
    endif()
endforeach()

find_path(TILEXR_EP_ACL_BASE_INCLUDE_DIR acl/acl_base.h
    PATHS ${TILEXR_EP_TEST_INCLUDE_DIRS}
    NO_DEFAULT_PATH
)
if(NOT TILEXR_EP_ACL_BASE_INCLUDE_DIR)
    set(TILEXR_EP_ACL_STUB_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_acl_stub")
    file(MAKE_DIRECTORY "${TILEXR_EP_ACL_STUB_DIR}/acl")
    file(WRITE "${TILEXR_EP_ACL_STUB_DIR}/acl/acl_base.h"
"#ifndef TILEXR_EP_TEST_ACL_BASE_H\n#define TILEXR_EP_TEST_ACL_BASE_H\ntypedef void *aclrtStream;\n#endif\n")
    list(APPEND TILEXR_EP_TEST_INCLUDE_DIRS "${TILEXR_EP_ACL_STUB_DIR}")
endif()

add_executable(test_tilexr_ep_layout
    unit/test_tilexr_ep_layout.cpp
    ${TILEXR_ROOT}/src/ep/host/ep_layout.cpp
)

add_executable(test_tilexr_ep_api_sources
    unit/test_tilexr_ep_api_sources.cpp
)

add_executable(test_tilexr_ep_host_validation
    unit/test_tilexr_ep_host_validation.cpp
    ${TILEXR_ROOT}/src/ep/host/ep_layout.cpp
    ${TILEXR_ROOT}/src/ep/host/ep_dispatch_host.cpp
)

target_include_directories(test_tilexr_ep_layout PRIVATE
    ${TILEXR_EP_TEST_INCLUDE_DIRS}
)

target_include_directories(test_tilexr_ep_host_validation PRIVATE
    ${TILEXR_EP_TEST_INCLUDE_DIRS}
)

target_compile_definitions(test_tilexr_ep_api_sources PRIVATE TILEXR_SOURCE_ROOT="${TILEXR_ROOT}")

add_test(NAME test_tilexr_ep_layout COMMAND test_tilexr_ep_layout)
add_test(NAME test_tilexr_ep_api_sources COMMAND test_tilexr_ep_api_sources)
add_test(NAME test_tilexr_ep_host_validation COMMAND test_tilexr_ep_host_validation)

install(TARGETS
    test_tilexr_ep_layout
    test_tilexr_ep_api_sources
    test_tilexr_ep_host_validation
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
