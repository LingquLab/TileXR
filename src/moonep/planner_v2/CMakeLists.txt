include(GNUInstallDirs)

find_program(BISHENG_EXECUTABLE bisheng)
if(NOT BISHENG_EXECUTABLE)
    message(FATAL_ERROR "bisheng not found; source the CANN environment before configuring the MoonEP planner")
endif()

set(TILEXR_MOONEP_PLANNER_DEFAULT_SOC_TYPE "$ENV{TILEXR_SOC_NAME}")
if(NOT TILEXR_MOONEP_PLANNER_DEFAULT_SOC_TYPE)
    set(TILEXR_MOONEP_PLANNER_DEFAULT_SOC_TYPE "Ascend950")
endif()
set(TILEXR_MOONEP_PLANNER_SOC_TYPE "${TILEXR_MOONEP_PLANNER_DEFAULT_SOC_TYPE}"
    CACHE STRING "TileXR MoonEP planner kernel SOC type")
string(TOLOWER "${TILEXR_MOONEP_PLANNER_SOC_TYPE}" TILEXR_MOONEP_PLANNER_SOC_TYPE_LOWER)

if(NOT TILEXR_MOONEP_PLANNER_SOC_TYPE_LOWER MATCHES "ascend950|ascend910a5|910a5")
    message(FATAL_ERROR
        "TileXR MoonEP planner is A5-only; got TILEXR_MOONEP_PLANNER_SOC_TYPE=${TILEXR_MOONEP_PLANNER_SOC_TYPE}")
endif()

set(TILEXR_MOONEP_PLANNER_KERNEL_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_moonep_planner_kernel.cpp")
set(TILEXR_MOONEP_PLANNER_KERNEL_LAUNCH_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/host/planner_kernel_launch.cpp")
set(TILEXR_MOONEP_PLANNER_KERNEL_SO
    "${CMAKE_CURRENT_BINARY_DIR}/libtilexr_moonep_planner_kernel.so")

set(TILEXR_MOONEP_PLANNER_KERNEL_OPTIONS
    -xcce
    -Xhost-start -ftrapv -Xhost-end
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    --cce-aicore-arch=dav-c310-vec
)

set(TILEXR_MOONEP_PLANNER_KERNEL_INCLUDES
    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/ascendc/include
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/include
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
    -I${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
    -I${ASCEND_DRIVER_PATH}/kernel/inc
    -I${CMAKE_SOURCE_DIR}/src/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_custom_command(
    OUTPUT "${TILEXR_MOONEP_PLANNER_KERNEL_SO}"
    COMMAND ${BISHENG_EXECUTABLE}
        ${TILEXR_MOONEP_PLANNER_KERNEL_OPTIONS}
        -std=gnu++17
        -fPIC
        -shared
        ${TILEXR_MOONEP_PLANNER_KERNEL_INCLUDES}
        "${TILEXR_MOONEP_PLANNER_KERNEL_SOURCE}"
        "${TILEXR_MOONEP_PLANNER_KERNEL_LAUNCH_SOURCE}"
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
        -o "${TILEXR_MOONEP_PLANNER_KERNEL_SO}"
    DEPENDS
        "${TILEXR_MOONEP_PLANNER_KERNEL_SOURCE}"
        "${TILEXR_MOONEP_PLANNER_KERNEL_LAUNCH_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/ep_plan_algorithm.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/ep_plan_algorithm_impl.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/ep_plan_types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/ep_plan_peer_mailbox.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/planner_common.h"
        "${CMAKE_SOURCE_DIR}/src/include/comm_args.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_ep_plan.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_sync.h"
    VERBATIM
    COMMENT "Building optimized A5 TileXR MoonEP planner_v2 kernel with bisheng"
)
add_custom_target(tilexr_moonep_planner_kernel ALL
    DEPENDS "${TILEXR_MOONEP_PLANNER_KERNEL_SO}")

add_library(tilexr-moonep-planner SHARED
    common/ep_plan_layout.cpp
    common/ep_plan_downstream.cpp
    host/ep_plan_host.cpp
    host/planner_layout.cpp
    host/planner_host.cpp
    host/planner_launch.cpp
    host/tilexr_moonep_planner.cpp
)
add_dependencies(tilexr-moonep-planner tilexr_moonep_planner_kernel)
target_compile_features(tilexr-moonep-planner PRIVATE cxx_std_14)

target_include_directories(tilexr-moonep-planner
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/host
)
target_link_directories(tilexr-moonep-planner PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
    ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)
target_link_libraries(tilexr-moonep-planner PRIVATE
    -l:libtilexr_moonep_planner_kernel.so
    tile-comm
    ascendcl
    runtime
)
set_target_properties(tilexr-moonep-planner PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
    INSTALL_RPATH "$ORIGIN"
    VERSION 2.0.0
    SOVERSION 2
)

install(TARGETS tilexr-moonep-planner LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "${TILEXR_MOONEP_PLANNER_KERNEL_SO}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    ${CMAKE_SOURCE_DIR}/src/include/tilexr_moonep_planner.h
    ${CMAKE_SOURCE_DIR}/src/include/tilexr_ep_plan.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
