include(GNUInstallDirs)

add_library(tilexr-ep SHARED
    host/ep_layout.cpp
    host/ep_dispatch_host.cpp
    host/ep_launch_context.cpp
    host/ep_kernel_launch.cpp
    host/tilexr_ep_dispatch.cpp
)

target_include_directories(tilexr-ep
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/host
)

target_link_libraries(tilexr-ep PRIVATE tile-comm)

install(TARGETS tilexr-ep LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_ep.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
