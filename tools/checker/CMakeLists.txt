include(GNUInstallDirs)

add_library(tilexr-checker-core STATIC
  src/status.cpp
  src/event.cpp
  src/memory.cpp
  src/world.cpp
  src/oracle.cpp
  src/shim_runtime.cpp
  src/trace_runtime.cpp
  src/trace_adapter_generator.cpp
  src/trace_bundle_verifier.cpp
  src/collective_trace_runner.cpp
  src/collective_trace_runner_hdb.cpp
  src/ep_dispatch_oracle_runner.cpp
  src/diagnostics.cpp
  src/report.cpp
  src/executor.cpp
  src/cli.cpp
)

target_include_directories(tilexr-checker-core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/shim
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/collectives/kernels
)

target_compile_features(tilexr-checker-core PUBLIC cxx_std_17)

add_executable(tilexr_checker
  src/checker_main.cpp
)

target_link_libraries(tilexr_checker PRIVATE tilexr-checker-core)

install(TARGETS tilexr_checker tilexr-checker-core
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
