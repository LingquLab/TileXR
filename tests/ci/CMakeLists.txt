cmake_minimum_required(VERSION 3.16)

project(TileXR_CI_Tests NONE)

enable_testing()

find_program(BASH_EXECUTABLE bash)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(RUBY_EXECUTABLE ruby)

if(NOT BASH_EXECUTABLE)
    message(FATAL_ERROR "bash is required to run TileXR CI tests")
endif()

add_test(
    NAME test_tilexr_ci_common_env_override
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_common_env_override.sh"
)

add_test(
    NAME test_tilexr_ci_provision_dry_run
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_provision_dry_run.sh"
)

add_test(
    NAME test_tilexr_ci_cann_downloader
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_cann_downloader.sh"
)

add_test(
    NAME test_tilexr_ci_cann_tree
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_cann_tree.sh"
)

add_test(
    NAME test_tilexr_ci_cann_lifecycle
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_cann_lifecycle.sh"
)

add_test(
    NAME test_tilexr_ci_cann_local_install_permissions
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/test_cann_local_install_permissions.sh"
)

add_test(
    NAME test_tilexr_ci_npu_state
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_npu_state.py -v
)

add_test(
    NAME test_tilexr_ci_gate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gate.py -v
)

add_test(
    NAME test_tilexr_ci_control_contract
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_control_contract.py -v
)

if(RUBY_EXECUTABLE)
    add_test(
        NAME test_tilexr_ci_workflows
        COMMAND ${RUBY_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_workflows.rb
    )
endif()
