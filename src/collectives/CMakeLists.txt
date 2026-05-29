#
# Copyright (c) 2024-2026 TileXR Project
# CMakeLists.txt for optional TileXR collectives library
#

set(TILEXR_COLLECTIVES_SOURCE_FILE
        host/tilexr_collectives.cpp
)

add_library(tilexr-collectives SHARED ${TILEXR_COLLECTIVES_SOURCE_FILE})

target_include_directories(tilexr-collectives
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/host
        ${ASCEND_DRIVER_PATH}/kernel/inc
)

target_link_directories(tilexr-collectives
        PRIVATE
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

target_link_libraries(tilexr-collectives
        PRIVATE
        tile-comm
        ascendcl
        runtime
        profapi
        ascend_hal
        dl
)

install(TARGETS tilexr-collectives LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
