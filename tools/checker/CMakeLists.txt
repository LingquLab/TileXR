add_library(tilexr-checker-core INTERFACE)

target_include_directories(tilexr-checker-core
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/include
)

target_compile_features(tilexr-checker-core INTERFACE cxx_std_14)
