include(GNUInstallDirs)

if(NOT TILEXR_BUILD_MOONEP AND NOT TILEXR_BUILD_MOONEP_PLANNER)
    return()
endif()

if(NOT TARGET tilexr-moonep-planner)
    add_subdirectory(planner)
endif()

if(NOT TILEXR_BUILD_MOONEP)
    return()
endif()

find_program(BISHENG_EXECUTABLE bisheng)
if(NOT BISHENG_EXECUTABLE)
    message(FATAL_ERROR "bisheng not found; source the CANN environment before configuring MoonEP")
endif()

set(TILEXR_MOONEP_PREFETCH_KERNEL_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_moonep_prefetch_weight_kernel.cpp")
set(TILEXR_MOONEP_PREFETCH_KERNEL_SO
    "${CMAKE_CURRENT_BINARY_DIR}/libtilexr_moonep_prefetch_weight_kernel.so")
set(TILEXR_MOONEP_PREFETCH_KERNEL_OPTIONS
    -xcce
    -Xhost-start -ftrapv -Xhost-end
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    --cce-aicore-arch=dav-c310-vec
)
set(TILEXR_MOONEP_PREFETCH_KERNEL_INCLUDES
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
)
add_custom_command(
    OUTPUT "${TILEXR_MOONEP_PREFETCH_KERNEL_SO}"
    COMMAND ${BISHENG_EXECUTABLE}
        ${TILEXR_MOONEP_PREFETCH_KERNEL_OPTIONS}
        -std=gnu++17 -fPIC -shared
        ${TILEXR_MOONEP_PREFETCH_KERNEL_INCLUDES}
        "${TILEXR_MOONEP_PREFETCH_KERNEL_SOURCE}"
        -L${ASCEND_DRIVER_PATH}/lib64/driver
        -L${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
        -lruntime -lascendcl -lstdc++ -lm -lc_sec -ldl -lpthread
        -o "${TILEXR_MOONEP_PREFETCH_KERNEL_SO}"
    DEPENDS
        "${TILEXR_MOONEP_PREFETCH_KERNEL_SOURCE}"
        "${CMAKE_SOURCE_DIR}/src/include/comm_args.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_udma.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_udma_reg.h"
        "${CMAKE_SOURCE_DIR}/src/include/tilexr_udma_types.h"
    VERBATIM
    COMMENT "Building fused A5 MoonEP PrefetchWeight kernel with bisheng"
)
add_custom_target(tilexr_moonep_prefetch_weight_kernel ALL
    DEPENDS "${TILEXR_MOONEP_PREFETCH_KERNEL_SO}")

add_library(tilexr-moonep SHARED
    host/prefetch_weight_layout.cpp
    host/prefetch_weight_launch.cpp
    host/tilexr_moonep.cpp
)
add_dependencies(tilexr-moonep tilexr_moonep_prefetch_weight_kernel)

target_compile_features(tilexr-moonep PRIVATE cxx_std_14)
target_include_directories(tilexr-moonep
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/host
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include
)
target_link_directories(tilexr-moonep PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
)
target_link_libraries(tilexr-moonep PRIVATE
    -l:libtilexr_moonep_prefetch_weight_kernel.so
    tilexr-moonep-planner
    tile-comm
    ascendcl
    runtime
)

set_target_properties(tilexr-moonep PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
    VERSION 2.0.0
    SOVERSION 2
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS tilexr-moonep LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "${TILEXR_MOONEP_PREFETCH_KERNEL_SO}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_moonep.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if((BUILD_TESTING OR TILEXR_BUILD_TESTS) AND NOT TARGET test_tilexr_moonep_host)
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/moonep
        ${CMAKE_BINARY_DIR}/tests/moonep)
endif()
