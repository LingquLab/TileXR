#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

include(GNUInstallDirs)

set(TILEXR_SOURCE_FILE tilexr_comm.cpp
        comm_wrap.cpp
        tilexr_internal.cpp
        tilexr_internal.h
        tilexr_log.h
        tools/socket/tilexr_sock_exchange.h
        tools/socket/tilexr_sock_exchange.cpp
        udma/tilexr_hccp_defs.h
        udma/tilexr_hccp_loader.h
        udma/tilexr_hccp_loader.cpp
        udma/tilexr_udma_layout.h
        udma/tilexr_udma_layout.cpp
        udma/tilexr_udma_transport.h
        udma/tilexr_udma_transport.cpp
)

add_library(tile-comm SHARED ${TILEXR_SOURCE_FILE})

target_include_directories(tile-comm
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${ASCEND_DRIVER_PATH}/kernel/inc
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/spdlog/include
)
target_link_directories(tile-comm
        PRIVATE
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

target_link_libraries(tile-comm
        PRIVATE
        ascendcl
        runtime
        profapi
        ascend_hal
        dl
)

install(TARGETS tile-comm LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_api.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/comm_args.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_sync.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_udma.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_udma_reg.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_udma_types.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(CODE "string(CONCAT _tilexr_optional_header \"tilexr_\" \"collectives.h\")
if(IS_ABSOLUTE \"${CMAKE_INSTALL_INCLUDEDIR}\")
    set(_tilexr_optional_header_path \"\$ENV{DESTDIR}${CMAKE_INSTALL_INCLUDEDIR}/\${_tilexr_optional_header}\")
else()
    set(_tilexr_optional_header_path \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/\${_tilexr_optional_header}\")
endif()
file(REMOVE \"\${_tilexr_optional_header_path}\")")
