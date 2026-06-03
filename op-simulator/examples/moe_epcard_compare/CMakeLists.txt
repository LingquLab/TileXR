cmake_minimum_required(VERSION 3.12)
project(MoeEpCardCompare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH aarch64)
else()
    set(ARCH x86_64)
endif()

if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

list(APPEND CMAKE_MODULE_PATH
    "${ASCEND_HOME_PATH}/${ARCH}-linux/tikcpp/ascendc_kernel_cmake/asc_modules")

enable_language(ASC)

set(ASC_COMPILE_OPTION
    -O2
    -std=c++17
    --cce-aicore-only
    -Wno-deprecated-declarations
    "SHELL:-mllvm -cce-aicore-long-call"
    "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm --cce-aicore-jump-expand=true"
)

set(CCE_INCLUDE_BASE ${ASCEND_HOME_PATH}/${ARCH}-linux)
include_directories(
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/interface/
    ${CCE_INCLUDE_BASE}/tikcpp/tikcfw/impl/
    ${CCE_INCLUDE_BASE}/pkg_inc/
    ${CCE_INCLUDE_BASE}/pkg_inc/runtime/
    ${CCE_INCLUDE_BASE}/pkg_inc/runtime/runtime/
    ${CMAKE_SOURCE_DIR}/src
)

set(KERNEL_SRC src/moe_epcard_compare.cpp)
set_source_files_properties(${KERNEL_SRC} PROPERTIES LANGUAGE ASC)

set(CCE_LLD "${ASCEND_HOME_PATH}/aarch64-linux/bin/ld.lld")

function(add_moe_case target_name kernel_name case_ep case_bs use_final)
    add_library(${target_name} OBJECT ${KERNEL_SRC})
    target_compile_definitions(${target_name} PRIVATE
        KERNEL_NAME=${kernel_name}
        CASE_EP=${case_ep}
        CASE_BS=${case_bs}
        USE_FINAL=${use_final}
    )
    target_compile_options(${target_name} PRIVATE
        ${ASC_COMPILE_OPTION}
        --cce-aicore-arch=dav-c310-vec
        -Wno-c99-designator
        -Wno-unused-command-line-argument
    )

    set(out_dir ${CMAKE_SOURCE_DIR}/op/${kernel_name})
    add_custom_command(
        OUTPUT ${out_dir}/my_kernel.o
        DEPENDS ${target_name}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${out_dir}"
        COMMAND ${CCE_LLD} -m aicorelinux -Ttext 0 -static -o ${out_dir}/my_kernel.o $<TARGET_OBJECTS:${target_name}>
        VERBATIM
    )
    add_custom_target(generate_${target_name} ALL DEPENDS ${out_dir}/my_kernel.o)
endfunction()

foreach(ep 32 64)
    foreach(bs 32 128 256)
        add_moe_case(kernel_ep${ep}_bs${bs}_baseline MoeEp${ep}Bs${bs}Baseline ${ep} ${bs} 0)
        add_moe_case(kernel_ep${ep}_bs${bs}_final MoeEp${ep}Bs${bs}Final ${ep} ${bs} 1)
    endforeach()
endforeach()
