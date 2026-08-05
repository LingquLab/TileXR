include(GNUInstallDirs)

if(NOT TILEXR_BUILD_MOONEP AND NOT TILEXR_BUILD_MOONEP_PLANNER)
    return()
endif()

if(NOT TARGET tilexr-moonep-planner)
    add_subdirectory(planner)
endif()

if(NOT TILEXR_BUILD_MOONEP)
    return()
endif()

add_subdirectory(reduce_grad)

add_library(tilexr-moonep SHARED
    host/tilexr_moonep.cpp
    reduce_grad/host/reduce_grad_layout.cpp
    reduce_grad/host/reduce_grad_host.cpp
    reduce_grad/host/reduce_grad_launch.cpp
)
add_dependencies(tilexr-moonep tilexr_moonep_reduce_grad_kernel)

target_compile_features(tilexr-moonep PRIVATE cxx_std_14)
target_include_directories(tilexr-moonep
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_grad/common
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_grad/host
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
        ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime
        ${ASCEND_HOME_PATH}/${ARCH}-linux/include
)
target_link_directories(tilexr-moonep PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/reduce_grad
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64
)
target_link_libraries(tilexr-moonep PRIVATE
    -l:libtilexr_moonep_reduce_grad_kernel.so
    tilexr-moonep-planner
    tile-comm
    ascendcl
    runtime
)

set_target_properties(tilexr-moonep PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS tilexr-moonep LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_moonep.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if((BUILD_TESTING OR TILEXR_BUILD_TESTS) AND NOT TARGET test_tilexr_moonep_host)
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/moonep
        ${CMAKE_BINARY_DIR}/tests/moonep)
endif()
