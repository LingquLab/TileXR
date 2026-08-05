include(GNUInstallDirs)

find_program(BISHENG_EXECUTABLE bisheng)
if(NOT BISHENG_EXECUTABLE)
    message(FATAL_ERROR "bisheng not found; source the CANN environment before configuring MoonEP ReduceGrad")
endif()

set(TILEXR_MOONEP_REDUCE_GRAD_DEFAULT_SOC_TYPE "$ENV{TILEXR_SOC_NAME}")
if(NOT TILEXR_MOONEP_REDUCE_GRAD_DEFAULT_SOC_TYPE)
    set(TILEXR_MOONEP_REDUCE_GRAD_DEFAULT_SOC_TYPE "Ascend950")
endif()
set(TILEXR_MOONEP_REDUCE_GRAD_SOC_TYPE "${TILEXR_MOONEP_REDUCE_GRAD_DEFAULT_SOC_TYPE}"
    CACHE STRING "TileXR MoonEP ReduceGrad kernel SOC type")
string(TOLOWER "${TILEXR_MOONEP_REDUCE_GRAD_SOC_TYPE}"
    TILEXR_MOONEP_REDUCE_GRAD_SOC_TYPE_LOWER)
if(NOT TILEXR_MOONEP_REDUCE_GRAD_SOC_TYPE_LOWER MATCHES "ascend950|ascend910a5|910a5")
    message(FATAL_ERROR
        "TileXR MoonEP ReduceGrad is A5-only; got TILEXR_MOONEP_REDUCE_GRAD_SOC_TYPE=${TILEXR_MOONEP_REDUCE_GRAD_SOC_TYPE}")
endif()

set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_moonep_reduce_grad_kernel.cpp")
set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SO
    "${CMAKE_CURRENT_BINARY_DIR}/libtilexr_moonep_reduce_grad_kernel.so")

set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_OPTIONS
    -xcce
    -Xhost-start -ftrapv -Xhost-end
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    --cce-aicore-arch=dav-c310-vec
)

set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_INCLUDES
    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/asc/include
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/ascendc/include
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/include
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
    -I${ASCEND_DRIVER_PATH}/kernel/inc
    -I${CMAKE_SOURCE_DIR}/src/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_custom_command(
    OUTPUT "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SO}"
    COMMAND ${BISHENG_EXECUTABLE}
        ${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_OPTIONS}
        -std=gnu++17
        -fPIC
        -shared
        ${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_INCLUDES}
        "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SOURCE}"
        -L${ASCEND_DRIVER_PATH}/lib64/driver
        -L${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        -L${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
        -lruntime
        -lascendcl
        -lstdc++
        -lm
        -lc_sec
        -ldl
        -lpthread
        -o "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SO}"
    DEPENDS
        "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/reduce_grad_common.h"
        "${CMAKE_SOURCE_DIR}/src/include/comm_args.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_data_as_flag.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_udma.h"
    VERBATIM
    COMMENT "Building A5 TileXR MoonEP ReduceGrad kernel with bisheng"
)
add_custom_target(tilexr_moonep_reduce_grad_kernel ALL
    DEPENDS "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SO}")

set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SO "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SO}"
    PARENT_SCOPE)

install(FILES "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SO}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
