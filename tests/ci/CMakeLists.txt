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
