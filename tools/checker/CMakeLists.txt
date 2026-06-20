add_library(tilexr-checker-core STATIC
  src/status.cpp
  src/event.cpp
  src/memory.cpp
  src/world.cpp
)

target_include_directories(tilexr-checker-core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/include
)

target_compile_features(tilexr-checker-core PUBLIC cxx_std_14)
