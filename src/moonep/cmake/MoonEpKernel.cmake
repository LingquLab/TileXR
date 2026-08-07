include(CMakeParseArguments)

find_program(TILEXR_MOONEP_CCE_LINKER
    NAMES ld.lld cce-ld
    HINTS
        "${ASCEND_HOME_PATH}/bin"
        "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin"
)
if(NOT TILEXR_MOONEP_CCE_LINKER)
    message(FATAL_ERROR "CCE linker not found in the selected CANN installation")
endif()

function(tilexr_add_moonep_kernel target)
    set(oneValueArgs SOURCE BINARY EMBED_CPP DATA_SYMBOL SIZE_SYMBOL)
    set(multiValueArgs OPTIONS INCLUDES DEPENDS)
    cmake_parse_arguments(MOONEP "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(required SOURCE BINARY EMBED_CPP DATA_SYMBOL SIZE_SYMBOL)
        if(NOT MOONEP_${required})
            message(FATAL_ERROR "tilexr_add_moonep_kernel requires ${required}")
        endif()
    endforeach()

    set(relocatable "${CMAKE_CURRENT_BINARY_DIR}/${target}_rel.o")
    add_custom_command(
        OUTPUT "${MOONEP_BINARY}"
        COMMAND ${BISHENG_EXECUTABLE}
            ${MOONEP_OPTIONS}
            --cce-aicore-only
            -std=gnu++17
            ${MOONEP_INCLUDES}
            -c "${MOONEP_SOURCE}"
            -o "${relocatable}"
        COMMAND ${TILEXR_MOONEP_CCE_LINKER}
            -m aicorelinux -Ttext=0 "${relocatable}"
            --static --allow-multiple-definition
            -o "${MOONEP_BINARY}"
        DEPENDS "${MOONEP_SOURCE}" ${MOONEP_DEPENDS}
        VERBATIM
        COMMENT "Building pure AICore kernel ${target}"
    )
    add_custom_target(${target} ALL DEPENDS "${MOONEP_BINARY}")

    add_custom_command(
        OUTPUT "${MOONEP_EMBED_CPP}"
        COMMAND ${CMAKE_COMMAND}
            -DTILEXR_MOONEP_KERNEL_BINARY=${MOONEP_BINARY}
            -DTILEXR_MOONEP_KERNEL_EMBED_CPP=${MOONEP_EMBED_CPP}
            -DTILEXR_MOONEP_KERNEL_DATA_SYMBOL=${MOONEP_DATA_SYMBOL}
            -DTILEXR_MOONEP_KERNEL_SIZE_SYMBOL=${MOONEP_SIZE_SYMBOL}
            -P "${CMAKE_SOURCE_DIR}/src/moonep/cmake/embed_moonep_kernel.cmake"
        DEPENDS "${MOONEP_BINARY}"
            "${CMAKE_SOURCE_DIR}/src/moonep/cmake/embed_moonep_kernel.cmake"
        VERBATIM
        COMMENT "Embedding pure AICore kernel ${target}"
    )
endfunction()
