cmake_minimum_required(VERSION 3.16)
project(TileXR_SDMA_Tests)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ARCH $ENV{ARCH})
if(NOT ARCH)
    set(ARCH $ENV{TILEXR_OS_ARCH})
endif()
if(NOT ARCH)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(ARCH STREQUAL "arm64")
        set(ARCH "aarch64")
    endif()
endif()

set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
if(NOT ASCEND_DRIVER_PATH)
    set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver")
endif()

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

find_library(TILEXR_LIB tile-comm
    HINTS "${TILEXR_ROOT}/install/lib" "${TILEXR_ROOT}/build/src/comm"
    REQUIRED)

link_directories(
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
    ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

add_executable(test_tilexr_sdma_metadata
    unit/test_tilexr_sdma_metadata.cpp
)
target_include_directories(test_tilexr_sdma_metadata PRIVATE
    ${TILEXR_ROOT}/install/include
    ${TILEXR_ROOT}/src/include
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

add_executable(test_tilexr_sdma_api_invalid
    unit/test_tilexr_sdma_api_invalid.cpp
)
target_include_directories(test_tilexr_sdma_api_invalid PRIVATE
    ${TILEXR_ROOT}/install/include
    ${TILEXR_ROOT}/src/include
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_DRIVER_PATH}/kernel/inc
)
target_link_libraries(test_tilexr_sdma_api_invalid
    ${TILEXR_LIB}
    ascendcl
    runtime
    ascend_hal
)

add_executable(test_tilexr_sdma_transport_disabled
    unit/test_tilexr_sdma_transport_disabled.cpp
    ${TILEXR_ROOT}/src/comm/sdma/tilexr_sdma_transport.cpp
)
target_include_directories(test_tilexr_sdma_transport_disabled PRIVATE
    ${TILEXR_ROOT}/src/comm
    ${TILEXR_ROOT}/src/include
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

add_executable(test_tilexr_sdma_comm_wiring
    unit/test_tilexr_sdma_comm_wiring.cpp
)
target_compile_definitions(test_tilexr_sdma_comm_wiring PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_executable(test_tilexr_sdma_source_guard
    unit/test_tilexr_sdma_source_guard.cpp
)
target_compile_definitions(test_tilexr_sdma_source_guard PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}"
)

add_executable(test_tilexr_sdma_disabled_comm
    integration/test_tilexr_sdma_disabled_comm.cpp
)
target_include_directories(test_tilexr_sdma_disabled_comm PRIVATE
    ${TILEXR_ROOT}/install/include
    ${TILEXR_ROOT}/src/include
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_DRIVER_PATH}/kernel/inc
)
target_link_libraries(test_tilexr_sdma_disabled_comm
    ${TILEXR_LIB}
    ascendcl
    runtime
)
set(INSTALL_TARGETS
    test_tilexr_sdma_metadata
    test_tilexr_sdma_api_invalid
    test_tilexr_sdma_transport_disabled
    test_tilexr_sdma_comm_wiring
    test_tilexr_sdma_source_guard
    test_tilexr_sdma_disabled_comm
)

install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
