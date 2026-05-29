#
# Copyright (c) 2024-2026 TileXR Project
# CMakeLists.txt for TileXR collectives tests
#

cmake_minimum_required(VERSION 3.16)
project(TileXR_Collectives_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)
enable_testing()

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(TILEXR_INSTALL_PREFIX "${TILEXR_ROOT}/install" CACHE PATH "TileXR install prefix for installed collectives headers")
set(TILEXR_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "TileXR install library directory relative to TILEXR_INSTALL_PREFIX")
set(TILEXR_COLLECTIVES_LIB "" CACHE FILEPATH "Path to libtilexr-collectives.so")
set(TILEXR_LIB "" CACHE FILEPATH "Path to libtile-comm.so")
if(IS_ABSOLUTE "${TILEXR_INSTALL_LIBDIR}")
    set(TILEXR_INSTALL_LIB_SEARCH_DIR "${TILEXR_INSTALL_LIBDIR}")
else()
    set(TILEXR_INSTALL_LIB_SEARCH_DIR "${TILEXR_INSTALL_PREFIX}/${TILEXR_INSTALL_LIBDIR}")
endif()
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

add_executable(test_tilexr_collectives_stub_behavior
    unit/test_tilexr_collectives_stub_behavior.cpp
)

if(NOT EXISTS "${TILEXR_INSTALL_PREFIX}/include/tilexr_collectives.h")
    message(FATAL_ERROR
        "Installed tilexr_collectives.h not found at ${TILEXR_INSTALL_PREFIX}/include/tilexr_collectives.h. "
        "Run cmake --install after building TileXR with -DTILEXR_BUILD_COLLECTIVES=ON or set TILEXR_INSTALL_PREFIX."
    )
endif()

if(NOT TILEXR_COLLECTIVES_LIB)
    find_library(TILEXR_COLLECTIVES_LIB_FOUND tilexr-collectives
        HINTS
        "${TILEXR_INSTALL_LIB_SEARCH_DIR}"
        NO_DEFAULT_PATH
    )
    if(TILEXR_COLLECTIVES_LIB_FOUND)
        set(TILEXR_COLLECTIVES_LIB "${TILEXR_COLLECTIVES_LIB_FOUND}")
    endif()
endif()

if(NOT TILEXR_COLLECTIVES_LIB)
    message(FATAL_ERROR
        "libtilexr-collectives.so not found under ${TILEXR_INSTALL_LIB_SEARCH_DIR}. "
        "Build/install TileXR with -DTILEXR_BUILD_COLLECTIVES=ON and set TILEXR_INSTALL_PREFIX, "
        "or pass -DTILEXR_COLLECTIVES_LIB=/path/to/libtilexr-collectives.so."
    )
endif()

if(NOT TILEXR_LIB)
    find_library(TILEXR_LIB_FOUND tile-comm
        HINTS
        "${TILEXR_INSTALL_LIB_SEARCH_DIR}"
        NO_DEFAULT_PATH
    )
    if(TILEXR_LIB_FOUND)
        set(TILEXR_LIB "${TILEXR_LIB_FOUND}")
    endif()
endif()

if(NOT TILEXR_LIB)
    message(FATAL_ERROR
        "libtile-comm.so not found under ${TILEXR_INSTALL_LIB_SEARCH_DIR}. "
        "Build/install TileXR and set TILEXR_INSTALL_PREFIX, "
        "or pass -DTILEXR_LIB=/path/to/libtile-comm.so."
    )
endif()

set(TILEXR_INSTALLED_INCLUDE_DIRS
    ${TILEXR_INSTALL_PREFIX}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
)

add_library(tilexr-comm-installed SHARED IMPORTED)
set_target_properties(tilexr-comm-installed PROPERTIES
    IMPORTED_LOCATION "${TILEXR_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${TILEXR_INSTALLED_INCLUDE_DIRS}"
)

add_library(tilexr-collectives-installed SHARED IMPORTED)
set_target_properties(tilexr-collectives-installed PROPERTIES
    IMPORTED_LOCATION "${TILEXR_COLLECTIVES_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${TILEXR_INSTALLED_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES tilexr-comm-installed
)

target_link_libraries(test_tilexr_collectives_header_compile
    tilexr-collectives-installed
)

target_link_libraries(test_tilexr_collectives_stub_behavior
    tilexr-collectives-installed
)

add_test(NAME test_tilexr_collectives_api COMMAND test_tilexr_collectives_api)
add_test(NAME test_tilexr_collectives_header_compile COMMAND test_tilexr_collectives_header_compile)
add_test(NAME test_tilexr_collectives_stub_behavior COMMAND test_tilexr_collectives_stub_behavior)

install(TARGETS
    test_tilexr_collectives_api
    test_tilexr_collectives_header_compile
    test_tilexr_collectives_stub_behavior
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
