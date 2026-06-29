include(GNUInstallDirs)

find_program(BISHENG_EXECUTABLE bisheng)
if(NOT BISHENG_EXECUTABLE)
    message(FATAL_ERROR "bisheng not found; source scripts/common_env.sh before configuring with -DTILEXR_BUILD_EP=ON")
endif()

set(TILEXR_EP_DEFAULT_SOC_TYPE "$ENV{TILEXR_SOC_NAME}")
if(NOT TILEXR_EP_DEFAULT_SOC_TYPE)
    set(TILEXR_EP_DEFAULT_SOC_TYPE "Ascend910B")
endif()
set(TILEXR_EP_SOC_TYPE "${TILEXR_EP_DEFAULT_SOC_TYPE}" CACHE STRING "TileXR EP kernel SOC type")
string(TOLOWER "${TILEXR_EP_SOC_TYPE}" TILEXR_EP_SOC_TYPE_LOWER)

if(TILEXR_EP_SOC_TYPE_LOWER MATCHES "ascend950|ascend910_9|ascend910-9|a5")
    set(TILEXR_EP_NPU_ARCH "dav-3510")
    set(TILEXR_EP_AICORE_ARCH "--cce-aicore-arch=dav-c310-vec")
    set(TILEXR_EP_CATLASS_ARCH "3510")
elseif(TILEXR_EP_SOC_TYPE_LOWER MATCHES "ascend910b|910b")
    set(TILEXR_EP_NPU_ARCH "dav-2201")
    set(TILEXR_EP_AICORE_ARCH "--cce-aicore-arch=dav-c220-vec")
    set(TILEXR_EP_CATLASS_ARCH "2201")
else()
    set(TILEXR_EP_NPU_ARCH "dav-2201")
    set(TILEXR_EP_AICORE_ARCH "--cce-aicore-arch=dav-c220-vec")
    set(TILEXR_EP_CATLASS_ARCH "2201")
endif()

set(TILEXR_EP_KERNEL_COMPILE_OPTIONS
    -xcce
    -Xhost-start -ftrapv -Xhost-end
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    ${TILEXR_EP_AICORE_ARCH}
)
set(TILEXR_EP_KERNEL_LINK_OPTIONS ${TILEXR_EP_AICORE_ARCH})

set(TILEXR_EP_DISPATCH_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_ep_dispatch_kernel.cpp")
set(TILEXR_EP_COMBINE_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_ep_combine_kernel.cpp")
set(TILEXR_EP_DISPATCH_KERNEL_SO "${CMAKE_CURRENT_BINARY_DIR}/libtilexr_ep_dispatch_kernel.so")
set(TILEXR_EP_COMBINE_KERNEL_SO "${CMAKE_CURRENT_BINARY_DIR}/libtilexr_ep_combine_kernel.so")
set(TILEXR_EP_KERNEL_INCLUDES
    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/runtime/include
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/include
    -I${ASCEND_DRIVER_PATH}/kernel/inc
    -I${CMAKE_SOURCE_DIR}/3rdparty
    -I${CMAKE_SOURCE_DIR}/src/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_custom_command(
    OUTPUT "${TILEXR_EP_DISPATCH_KERNEL_SO}"
    COMMAND ${BISHENG_EXECUTABLE}
        ${TILEXR_EP_KERNEL_COMPILE_OPTIONS}
        -std=gnu++17
        -fPIC
        -shared
        ${TILEXR_EP_KERNEL_LINK_OPTIONS}
        -DCATLASS_ARCH=${TILEXR_EP_CATLASS_ARCH}
        ${TILEXR_EP_KERNEL_INCLUDES}
        "${TILEXR_EP_DISPATCH_KERNEL_SOURCE}"
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
        -o "${TILEXR_EP_DISPATCH_KERNEL_SO}"
    DEPENDS
        "${TILEXR_EP_DISPATCH_KERNEL_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_ep_dispatch_helpers.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_ep_combine_helpers.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/ep_window.h"
        "${CMAKE_SOURCE_DIR}/src/include/comm_args.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_sync.h"
    VERBATIM
    COMMENT "Building TileXR EP dispatch kernel with bisheng"
)
add_custom_target(tilexr_ep_dispatch_kernel ALL DEPENDS "${TILEXR_EP_DISPATCH_KERNEL_SO}")

add_custom_command(
    OUTPUT "${TILEXR_EP_COMBINE_KERNEL_SO}"
    COMMAND ${BISHENG_EXECUTABLE}
        ${TILEXR_EP_KERNEL_COMPILE_OPTIONS}
        -std=gnu++17
        -fPIC
        -shared
        ${TILEXR_EP_KERNEL_LINK_OPTIONS}
        -DCATLASS_ARCH=${TILEXR_EP_CATLASS_ARCH}
        ${TILEXR_EP_KERNEL_INCLUDES}
        "${TILEXR_EP_COMBINE_KERNEL_SOURCE}"
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
        -o "${TILEXR_EP_COMBINE_KERNEL_SO}"
    DEPENDS
        "${TILEXR_EP_COMBINE_KERNEL_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_ep_kernel_common.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/ep_window.h"
        "${CMAKE_SOURCE_DIR}/src/include/comm_args.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_sync.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_udma.h"
    VERBATIM
    COMMENT "Building TileXR EP combine kernel with bisheng"
)
add_custom_target(tilexr_ep_combine_kernel ALL DEPENDS "${TILEXR_EP_COMBINE_KERNEL_SO}")

add_library(tilexr-ep SHARED
    host/ep_layout.cpp
    host/ep_dispatch_host.cpp
    host/ep_launch_context.cpp
    host/ep_kernel_launch.cpp
    host/tilexr_ep_dispatch.cpp
)

add_dependencies(tilexr-ep tilexr_ep_dispatch_kernel tilexr_ep_combine_kernel)

target_include_directories(tilexr-ep
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/host
)

target_link_directories(tilexr-ep
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)

target_link_libraries(tilexr-ep
    PRIVATE
        -l:libtilexr_ep_dispatch_kernel.so
        -l:libtilexr_ep_combine_kernel.so
        tile-comm
        ascendcl
        runtime
        ascend_hal
)

set_target_properties(tilexr-ep PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS tilexr-ep LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    "${TILEXR_EP_DISPATCH_KERNEL_SO}"
    "${TILEXR_EP_COMBINE_KERNEL_SO}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_ep.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
