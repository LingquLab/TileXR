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
