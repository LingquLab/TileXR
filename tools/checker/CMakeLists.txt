add_library(tilexr-checker-core STATIC
  src/status.cpp
  src/event.cpp
  src/memory.cpp
  src/world.cpp
  src/oracle.cpp
  src/shim_runtime.cpp
  src/diagnostics.cpp
  src/report.cpp
  src/executor.cpp
  src/cli.cpp
)

target_include_directories(tilexr-checker-core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/include
)

target_compile_features(tilexr-checker-core PUBLIC cxx_std_14)

add_executable(tilexr_checker
  src/checker_main.cpp
)

target_link_libraries(tilexr_checker PRIVATE tilexr-checker-core)
