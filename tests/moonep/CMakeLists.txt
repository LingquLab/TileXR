cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(TileXR_MoonEP_Tests LANGUAGES C CXX)
    enable_testing()
else()
    enable_language(C)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TILEXR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(TILEXR_MOONEP_FAKE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fake_include")
file(MAKE_DIRECTORY "${TILEXR_MOONEP_FAKE_INCLUDE_DIR}/acl")

file(WRITE "${TILEXR_MOONEP_FAKE_INCLUDE_DIR}/acl/acl_base.h"
"#ifndef TILEXR_MOONEP_TEST_ACL_BASE_H\n"
"#define TILEXR_MOONEP_TEST_ACL_BASE_H\n"
"typedef void *aclrtStream;\n"
"typedef int aclError;\n"
"#define ACL_SUCCESS 0\n"
"#endif\n")

file(WRITE "${TILEXR_MOONEP_FAKE_INCLUDE_DIR}/acl/acl_rt.h"
"#ifndef TILEXR_MOONEP_TEST_ACL_RT_H\n"
"#define TILEXR_MOONEP_TEST_ACL_RT_H\n"
"#include <stddef.h>\n"
"#include <stdint.h>\n"
"#include \"acl_base.h\"\n"
"typedef enum aclrtMemcpyKind { ACL_MEMCPY_DEVICE_TO_DEVICE = 3 } aclrtMemcpyKind;\n"
"aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream);\n"
"aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream);\n"
"aclError aclrtSynchronizeStream(aclrtStream stream);\n"
"#endif\n")

file(WRITE "${TILEXR_MOONEP_FAKE_INCLUDE_DIR}/tilexr_moonep_planner.h"
"#ifndef TILEXR_MOONEP_TEST_PLANNER_H\n"
"#define TILEXR_MOONEP_TEST_PLANNER_H\n"
"#include <stdint.h>\n"
"#include \"acl/acl_base.h\"\n"
"#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
"typedef void *TileXRCommPtr;\n"
"int TileXRMoonEpPlannerGetWorkspaceSizeV2(TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount, uint64_t *workspaceBytes, int64_t *dispatchedCapacity);\n"
"int TileXRMoonEpPlannerV2(const int32_t *topkExpertIds, const int32_t *tokensPerExpert, TileXRCommPtr comm, int64_t s, int64_t k, int64_t expertCount, void *workspace, uint64_t workspaceBytes, int32_t *dst, int32_t *cuSeqlens, int32_t *expertsToCopy, int32_t *remoteStats, int32_t *plannerStatus, uint64_t waitIterations, aclrtStream stream);\n"
"#ifdef __cplusplus\n}\n#endif\n"
"#endif\n")

file(WRITE "${TILEXR_MOONEP_FAKE_INCLUDE_DIR}/tilexr_api.h"
"#ifndef TILEXR_MOONEP_TEST_TILEXR_API_H\n"
"#define TILEXR_MOONEP_TEST_TILEXR_API_H\n"
"#ifdef __cplusplus\n"
"namespace TileXR { struct CommArgs { int rank; int rankSize; }; }\n"
"extern \"C\" {\n"
"typedef void *TileXRCommPtr;\n"
"int TileXRGetCommArgsHost(TileXRCommPtr comm, TileXR::CommArgs *&commArgs);\n"
"}\n"
"#endif\n"
"#endif\n")

set(TILEXR_MOONEP_TEST_INCLUDE_DIRS
    ${TILEXR_MOONEP_FAKE_INCLUDE_DIR}
    ${TILEXR_ROOT}/src/include
)

add_executable(test_tilexr_moonep_c_header
    unit/test_tilexr_moonep_c_header.c
)
target_include_directories(test_tilexr_moonep_c_header PRIVATE
    ${TILEXR_MOONEP_TEST_INCLUDE_DIRS})

add_executable(test_tilexr_moonep_abi_layout
    unit/test_tilexr_moonep_abi_layout.cpp
)
target_include_directories(test_tilexr_moonep_abi_layout PRIVATE
    ${TILEXR_MOONEP_TEST_INCLUDE_DIRS})

add_executable(test_tilexr_moonep_host
    unit/test_tilexr_moonep_host.cpp
    ${TILEXR_ROOT}/src/moonep/host/tilexr_moonep.cpp
)
target_include_directories(test_tilexr_moonep_host PRIVATE
    ${TILEXR_MOONEP_TEST_INCLUDE_DIRS})

add_executable(test_tilexr_moonep_sources
    unit/test_tilexr_moonep_sources.cpp
)
target_compile_definitions(test_tilexr_moonep_sources PRIVATE
    TILEXR_SOURCE_ROOT="${TILEXR_ROOT}")

add_test(NAME test_tilexr_moonep_c_header COMMAND test_tilexr_moonep_c_header)
add_test(NAME test_tilexr_moonep_abi_layout COMMAND test_tilexr_moonep_abi_layout)
add_test(NAME test_tilexr_moonep_host COMMAND test_tilexr_moonep_host)
add_test(NAME test_tilexr_moonep_sources COMMAND test_tilexr_moonep_sources)

if(TARGET tilexr-moonep)
    add_executable(tilexr_moonep_flow_demo
        demo/tilexr_moonep_flow_demo.cpp
    )
    target_compile_features(tilexr_moonep_flow_demo PRIVATE cxx_std_14)
    target_include_directories(tilexr_moonep_flow_demo PRIVATE
        ${TILEXR_ROOT}/src/include
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
    )
    target_link_directories(tilexr_moonep_flow_demo PRIVATE
        ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
    )
    target_link_libraries(tilexr_moonep_flow_demo PRIVATE
        tilexr-moonep
        tile-comm
        ascendcl
        runtime
    )
    install(TARGETS tilexr_moonep_flow_demo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(PROGRAMS demo/run_a5.sh
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        RENAME tilexr_moonep_flow_run_a5.sh)
endif()
