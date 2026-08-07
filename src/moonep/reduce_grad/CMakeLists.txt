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
set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_BINARY
    "${CMAKE_CURRENT_BINARY_DIR}/tilexr_moonep_reduce_grad_kernel.o")
set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_EMBED_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/tilexr_moonep_reduce_grad_kernel_embed.cpp")

set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_OPTIONS
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

set_source_files_properties("${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SOURCE}"
    PROPERTIES LANGUAGE CCE)
add_library(tilexr_moonep_reduce_grad_kernel_obj OBJECT
    "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_SOURCE}")
target_compile_options(tilexr_moonep_reduce_grad_kernel_obj PRIVATE
    ${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_OPTIONS}
    --cce-aicore-arch=dav-c310-vec)
target_include_directories(tilexr_moonep_reduce_grad_kernel_obj PRIVATE
    ${ASCEND_HOME_PATH}/compiler/tikcpp
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/${ARCH}-linux/asc/include
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
    OUTPUT "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_BINARY}"
    COMMAND ${TILEXR_MOONEP_AICORE_LINKER} -m aicorelinux -Ttext=0 --static
        -o "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_BINARY}"
        $<TARGET_OBJECTS:tilexr_moonep_reduce_grad_kernel_obj>
    DEPENDS tilexr_moonep_reduce_grad_kernel_obj
        $<TARGET_OBJECTS:tilexr_moonep_reduce_grad_kernel_obj>
    VERBATIM
    COMMENT "Linking TileXR MoonEP ReduceGrad AICore binary")
add_custom_target(tilexr_moonep_reduce_grad_kernel ALL
    DEPENDS "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_BINARY}")

add_custom_command(
    OUTPUT "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_EMBED_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DTILEXR_MOONEP_KERNEL_BINARY=${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_BINARY}
        -DTILEXR_MOONEP_KERNEL_EMBED_CPP=${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_EMBED_CPP}
        -DTILEXR_MOONEP_KERNEL_SYMBOL=TileXRMoonEpReduceGradKernelBinary
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/embed_moonep_kernel.cmake
    DEPENDS tilexr_moonep_reduce_grad_kernel
        "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_BINARY}"
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/embed_moonep_kernel.cmake
    VERBATIM)
add_custom_target(tilexr_moonep_reduce_grad_kernel_embed
    DEPENDS "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_EMBED_CPP}")

set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_BINARY
    "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_BINARY}" PARENT_SCOPE)
set(TILEXR_MOONEP_REDUCE_GRAD_KERNEL_EMBED_CPP
    "${TILEXR_MOONEP_REDUCE_GRAD_KERNEL_EMBED_CPP}" PARENT_SCOPE)
