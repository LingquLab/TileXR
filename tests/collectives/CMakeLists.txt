#
# Copyright (c) 2024-2026 TileXR Project
# CMakeLists.txt for TileXR collectives tests
#

cmake_minimum_required(VERSION 3.16)
project(TileXR_Collectives_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_executable(test_tilexr_collectives_api
    unit/test_tilexr_collectives_api.cpp
)

target_compile_definitions(test_tilexr_collectives_api PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

install(TARGETS
    test_tilexr_collectives_api
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
