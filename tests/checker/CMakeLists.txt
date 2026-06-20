function(tilexr_checker_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE tilexr-checker-core)
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
  target_compile_definitions(${name} PRIVATE TILEXR_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

tilexr_checker_test(test_tilexr_checker_sources
  unit/test_checker_sources.cpp
)

tilexr_checker_test(test_tilexr_checker_world
  unit/test_checker_world.cpp
)
