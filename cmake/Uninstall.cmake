#
# Uninstall routine Removes all files in the <build_folder>/install_manifest.txt Uses cmake -E remove <file> to remove
# file in a cross platform way (hopefully)

set(MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt")

if(NOT EXISTS ${MANIFEST})
    message(FATAL_ERROR "Cannot find install manifest: '${MANIFEST}'")
endif()

file(STRINGS ${MANIFEST} files)
foreach(file ${files})
    if(EXISTS ${file})
        message(STATUS "Removing file: '${file}'")

        exec_program(
            ${CMAKE_COMMAND} ARGS
            "-E remove ${file}"
            OUTPUT_VARIABLE stdout
            RETURN_VALUE result
        )

        if(NOT "${result}" STREQUAL 0)
            message(FATAL_ERROR "Failed to remove file: '${file}'.")
        endif()
    else()
        message(STATUS "File '${file}' does not exist.")
    endif()
endforeach(file)
