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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/src/collectives/kernels/cmake")
enable_language(CCE)

set(TILEXR_MOONEP_PREFETCH_KERNEL_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilexr_moonep_prefetch_weight_kernel.cpp")
set(TILEXR_MOONEP_PREFETCH_KERNEL_OP
    "${CMAKE_CURRENT_BINARY_DIR}/tilexr_moonep_prefetch_weight_kernel.o")
set(TILEXR_MOONEP_PREFETCH_KERNEL_EMBED_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/tilexr_moonep_prefetch_weight_kernel_embed.cpp")
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

set_source_files_properties("${TILEXR_MOONEP_PREFETCH_KERNEL_SOURCE}" PROPERTIES LANGUAGE CCE)
set_source_files_properties("${TILEXR_MOONEP_PREFETCH_KERNEL_SOURCE}" PROPERTIES
    OBJECT_DEPENDS "${CMAKE_SOURCE_DIR}/src/include/comm_args.h;${CMAKE_SOURCE_DIR}/src/include/tilexr_udma.h;${CMAKE_SOURCE_DIR}/src/include/tilexr_udma_reg.h;${CMAKE_SOURCE_DIR}/src/include/tilexr_udma_types.h")
add_library(tilexr_moonep_prefetch_weight_kernel_tmp OBJECT
    "${TILEXR_MOONEP_PREFETCH_KERNEL_SOURCE}")
target_compile_options(tilexr_moonep_prefetch_weight_kernel_tmp PRIVATE
    -O2
    -std=gnu++17
    --cce-aicore-only
    --cce-auto-sync
    -Wno-deprecated-declarations
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    --cce-aicore-arch=dav-c310-vec
    ${TILEXR_MOONEP_PREFETCH_KERNEL_INCLUDES})

set(TILEXR_MOONEP_PREFETCH_KERNEL_TMP_OBJECT
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/tilexr_moonep_prefetch_weight_kernel_tmp.dir/kernels/tilexr_moonep_prefetch_weight_kernel.cpp.o")
add_custom_command(
    OUTPUT "${TILEXR_MOONEP_PREFETCH_KERNEL_OP}"
    COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
        "${TILEXR_MOONEP_PREFETCH_KERNEL_TMP_OBJECT}"
        --static -o "${TILEXR_MOONEP_PREFETCH_KERNEL_OP}" --allow-multiple-definition
    COMMAND truncate -c -s 10485760 "${TILEXR_MOONEP_PREFETCH_KERNEL_OP}"
    DEPENDS tilexr_moonep_prefetch_weight_kernel_tmp
        "${TILEXR_MOONEP_PREFETCH_KERNEL_TMP_OBJECT}"
    VERBATIM
    COMMENT "Linking fused A5 MoonEP PrefetchWeight AICore binary")

add_custom_command(
    OUTPUT "${TILEXR_MOONEP_PREFETCH_KERNEL_EMBED_CPP}"
    COMMAND ${CMAKE_COMMAND}
        -DTILEXR_MOONEP_PREFETCH_KERNEL_OP=${TILEXR_MOONEP_PREFETCH_KERNEL_OP}
        -DTILEXR_MOONEP_PREFETCH_KERNEL_EMBED_CPP=${TILEXR_MOONEP_PREFETCH_KERNEL_EMBED_CPP}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_prefetch_weight_kernel.cmake"
    DEPENDS "${TILEXR_MOONEP_PREFETCH_KERNEL_OP}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_prefetch_weight_kernel.cmake"
    VERBATIM)
add_custom_target(tilexr_moonep_prefetch_weight_kernel ALL
    DEPENDS "${TILEXR_MOONEP_PREFETCH_KERNEL_OP}"
        "${TILEXR_MOONEP_PREFETCH_KERNEL_EMBED_CPP}")

add_library(tilexr-moonep SHARED
    host/prefetch_weight_layout.cpp
    host/prefetch_weight_launch.cpp
    host/tilexr_moonep.cpp
    "${TILEXR_MOONEP_PREFETCH_KERNEL_EMBED_CPP}"
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
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
)
target_link_libraries(tilexr-moonep PRIVATE
    tilexr-moonep-planner
    tile-comm
    ascendcl
    runtime
)

set_target_properties(tilexr-moonep PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS tilexr-moonep LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_moonep.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if((BUILD_TESTING OR TILEXR_BUILD_TESTS) AND NOT TARGET test_tilexr_moonep_host)
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/moonep
        ${CMAKE_BINARY_DIR}/tests/moonep)
endif()
