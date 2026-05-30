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

set(_tilexr_default_ascend_home_path "$ENV{ASCEND_HOME_PATH}")
if(NOT _tilexr_default_ascend_home_path)
    set(_tilexr_default_ascend_home_path "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(ASCEND_HOME_PATH "${_tilexr_default_ascend_home_path}" CACHE PATH "Ascend toolkit path")

set(_tilexr_default_arch "$ENV{ARCH}")
if(NOT _tilexr_default_arch)
    set(_tilexr_default_arch "$ENV{TILEXR_OS_ARCH}")
endif()
if(NOT _tilexr_default_arch)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE _tilexr_default_arch
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_tilexr_default_arch STREQUAL "arm64")
        set(_tilexr_default_arch "aarch64")
    endif()
endif()
set(ARCH "${_tilexr_default_arch}" CACHE STRING "Target OS architecture")

set(_tilexr_default_ascend_driver_path "$ENV{ASCEND_DRIVER_PATH}")
if(NOT _tilexr_default_ascend_driver_path)
    set(_tilexr_default_ascend_driver_path "/usr/local/Ascend/driver")
endif()
set(ASCEND_DRIVER_PATH "${_tilexr_default_ascend_driver_path}" CACHE PATH "Ascend driver path")

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

add_executable(test_tilexr_ep_kernel_sources unit/test_tilexr_ep_kernel_sources.cpp)

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
target_compile_definitions(test_tilexr_ep_kernel_sources PRIVATE TILEXR_SOURCE_ROOT="${TILEXR_ROOT}")

add_test(NAME test_tilexr_ep_layout COMMAND test_tilexr_ep_layout)
add_test(NAME test_tilexr_ep_api_sources COMMAND test_tilexr_ep_api_sources)
add_test(NAME test_tilexr_ep_kernel_sources COMMAND test_tilexr_ep_kernel_sources)
add_test(NAME test_tilexr_ep_host_validation COMMAND test_tilexr_ep_host_validation)

install(TARGETS
    test_tilexr_ep_layout
    test_tilexr_ep_api_sources
    test_tilexr_ep_kernel_sources
    test_tilexr_ep_host_validation
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_TILEXR_EP_DEMO)
    if(IS_ABSOLUTE "${TILEXR_INSTALL_LIBDIR}")
        set(TILEXR_INSTALL_LIB_SEARCH_DIR "${TILEXR_INSTALL_LIBDIR}")
    else()
        set(TILEXR_INSTALL_LIB_SEARCH_DIR "${TILEXR_INSTALL_PREFIX}/${TILEXR_INSTALL_LIBDIR}")
    endif()
    if(IS_ABSOLUTE "${TILEXR_INSTALL_INCLUDEDIR}")
        set(TILEXR_INSTALL_INCLUDE_SEARCH_DIR "${TILEXR_INSTALL_INCLUDEDIR}")
    else()
        set(TILEXR_INSTALL_INCLUDE_SEARCH_DIR "${TILEXR_INSTALL_PREFIX}/${TILEXR_INSTALL_INCLUDEDIR}")
    endif()

    find_library(TILEXR_EP_LIB tilexr-ep HINTS "${TILEXR_INSTALL_LIB_SEARCH_DIR}" REQUIRED)
    find_library(TILEXR_COMM_LIB tile-comm HINTS "${TILEXR_INSTALL_LIB_SEARCH_DIR}" REQUIRED)

    add_executable(tilexr_ep_dispatch_demo demo/tilexr_ep_dispatch_demo.cpp)

    target_include_directories(tilexr_ep_dispatch_demo PRIVATE
        "${TILEXR_INSTALL_INCLUDE_SEARCH_DIR}"
        "${ASCEND_HOME_PATH}/${ARCH}-linux/include"
        "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc"
        "${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime"
    )

    target_link_directories(tilexr_ep_dispatch_demo PRIVATE
        "${TILEXR_INSTALL_LIB_SEARCH_DIR}"
        "${ASCEND_HOME_PATH}/${ARCH}-linux/lib64"
        "${ASCEND_HOME_PATH}/${ARCH}-linux/devlib"
        "${ASCEND_DRIVER_PATH}/lib64/driver"
    )

    target_link_libraries(tilexr_ep_dispatch_demo PRIVATE
        ${TILEXR_EP_LIB}
        ${TILEXR_COMM_LIB}
        ascendcl
        runtime
        ascend_hal
    )

    install(TARGETS tilexr_ep_dispatch_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
