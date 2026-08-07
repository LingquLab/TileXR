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
set(TILEXR_MOONEP_PLANNER_KERNEL_BINARY
    "${CMAKE_CURRENT_BINARY_DIR}/tilexr_moonep_planner_kernel.o")
set(TILEXR_MOONEP_PLANNER_KERNEL_EMBED_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/tilexr_moonep_planner_kernel_embed.cpp")

set(TILEXR_MOONEP_PLANNER_KERNEL_OPTIONS
    -O2
    -std=gnu++17
    --cce-aicore-only
    -Wno-deprecated-declarations
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set_source_files_properties("${TILEXR_MOONEP_PLANNER_KERNEL_SOURCE}"
    PROPERTIES LANGUAGE CCE)
add_library(tilexr_moonep_planner_kernel_obj OBJECT
    "${TILEXR_MOONEP_PLANNER_KERNEL_SOURCE}")
target_compile_options(tilexr_moonep_planner_kernel_obj PRIVATE
    ${TILEXR_MOONEP_PLANNER_KERNEL_OPTIONS}
    --cce-aicore-arch=dav-c310-vec)
target_include_directories(tilexr_moonep_planner_kernel_obj PRIVATE
    ${ASCEND_HOME_PATH}/compiler/tikcpp
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/${ARCH}-linux/ascendc/include
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(TILEXR_MOONEP_AICORE_LINKER
    "${ASCEND_HOME_PATH}/${ARCH}-linux/bin/ld.lld")
add_custom_command(
    OUTPUT "${TILEXR_MOONEP_PLANNER_KERNEL_BINARY}"
    COMMAND ${TILEXR_MOONEP_AICORE_LINKER} -m aicorelinux -Ttext=0 --static
        -o "${TILEXR_MOONEP_PLANNER_KERNEL_BINARY}"
        $<TARGET_OBJECTS:tilexr_moonep_planner_kernel_obj>
    DEPENDS tilexr_moonep_planner_kernel_obj
        $<TARGET_OBJECTS:tilexr_moonep_planner_kernel_obj>
    VERBATIM
    COMMENT "Linking TileXR MoonEP planner AICore binary")
add_custom_target(tilexr_moonep_planner_kernel ALL
    DEPENDS "${TILEXR_MOONEP_PLANNER_KERNEL_BINARY}")

add_custom_command(
    OUTPUT "${TILEXR_MOONEP_PLANNER_KERNEL_EMBED_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DTILEXR_MOONEP_KERNEL_BINARY=${TILEXR_MOONEP_PLANNER_KERNEL_BINARY}
        -DTILEXR_MOONEP_KERNEL_EMBED_CPP=${TILEXR_MOONEP_PLANNER_KERNEL_EMBED_CPP}
        -DTILEXR_MOONEP_KERNEL_SYMBOL=TileXRMoonEpPlannerKernelBinary
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/embed_moonep_kernel.cmake
    DEPENDS tilexr_moonep_planner_kernel
        "${TILEXR_MOONEP_PLANNER_KERNEL_BINARY}"
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/embed_moonep_kernel.cmake
    VERBATIM)

add_library(tilexr-moonep-planner SHARED
    host/planner_layout.cpp
    host/planner_host.cpp
    host/planner_launch.cpp
    host/tilexr_moonep_planner.cpp
    ${TILEXR_MOONEP_PLANNER_KERNEL_EMBED_CPP}
)
add_dependencies(tilexr-moonep-planner tilexr_moonep_planner_kernel)
set_source_files_properties(${TILEXR_MOONEP_PLANNER_KERNEL_EMBED_CPP}
    PROPERTIES
        GENERATED TRUE
        OBJECT_DEPENDS "${TILEXR_MOONEP_PLANNER_KERNEL_BINARY}")

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
)
target_link_libraries(tilexr-moonep-planner PRIVATE
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
install(FILES ${CMAKE_SOURCE_DIR}/src/include/tilexr_moonep_planner.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
