#
# Copyright (c) 2024-2026 TileXR Project
# CMakeLists.txt for TileXR collectives tests
#

cmake_minimum_required(VERSION 3.16)
project(TileXR_Collectives_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(TILEXR_COLLECTIVES_LIB "" CACHE FILEPATH "Path to libtilexr-collectives.so")
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

add_executable(test_tilexr_collectives_api
    unit/test_tilexr_collectives_api.cpp
)

target_compile_definitions(test_tilexr_collectives_api PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_executable(test_tilexr_collectives_header_compile
    unit/test_tilexr_collectives_header_compile.cpp
)

target_include_directories(test_tilexr_collectives_header_compile PRIVATE
    ${TILEXR_ROOT}/src/include
    ${TILEXR_ROOT}/3rdparty
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
)

if(NOT TILEXR_COLLECTIVES_LIB)
    find_library(TILEXR_COLLECTIVES_LIB_FOUND tilexr-collectives
        HINTS
        "${TILEXR_ROOT}/install/lib"
        "${TILEXR_ROOT}/build/src/collectives"
    )
    if(TILEXR_COLLECTIVES_LIB_FOUND)
        set(TILEXR_COLLECTIVES_LIB "${TILEXR_COLLECTIVES_LIB_FOUND}")
    endif()
endif()

if(NOT TILEXR_COLLECTIVES_LIB)
    message(FATAL_ERROR
        "libtilexr-collectives.so not found. Build/install TileXR with -DTILEXR_BUILD_COLLECTIVES=ON "
        "or pass -DTILEXR_COLLECTIVES_LIB=/path/to/libtilexr-collectives.so."
    )
endif()

find_library(TILEXR_COMM_LIB tile-comm
    HINTS
    "${TILEXR_ROOT}/install/lib"
    "${TILEXR_ROOT}/build/src/comm"
    REQUIRED
)

target_link_libraries(test_tilexr_collectives_header_compile
    ${TILEXR_COLLECTIVES_LIB}
    ${TILEXR_COMM_LIB}
)

install(TARGETS
    test_tilexr_collectives_api
    test_tilexr_collectives_header_compile
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
