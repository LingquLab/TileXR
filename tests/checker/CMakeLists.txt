function(tilexr_checker_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE tilexr-checker-core)
  target_include_directories(${name} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/tools/checker/shim
  )
  target_compile_definitions(${name} PRIVATE TILEXR_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

tilexr_checker_test(test_tilexr_checker_sources
  unit/test_checker_sources.cpp
)

tilexr_checker_test(test_tilexr_checker_world
  unit/test_checker_world.cpp
)

tilexr_checker_test(test_tilexr_checker_oracles
  unit/test_checker_oracles.cpp
)

tilexr_checker_test(test_tilexr_checker_shim_events
  unit/test_checker_shim_events.cpp
)

tilexr_checker_test(test_tilexr_checker_diagnostics
  unit/test_checker_diagnostics.cpp
)
