cmake_minimum_required(VERSION 3.9)

find_package(Doxygen)
find_package(Sphinx)

if(DOXYGEN_FOUND)
    message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE}")
else()
    message(STATUS "Sphinx not found, no documentation building.")
endif()

if(SPHINX_EXECUTABLE)
    message(STATUS "Sphinx found: ${SPHINX_EXECUTABLE}")
else()
    message(STATUS "Doxygen not found, no documentation building.")
endif()


if(DOXYGEN_FOUND AND SPHINX_EXECUTABLE)
    # setup the Doxyfile
    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    # add the doxygen target
    add_custom_target(doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating the Doxygen documentation"
            VERBATIM)


    # add the sphinx target
    add_custom_target(sphinx
            COMMAND ${SPHINX_EXECUTABLE} -b html
            -Dbreathe_projects.elsa=${CMAKE_CURRENT_BINARY_DIR}/xml # breathe config
            ${CMAKE_CURRENT_SOURCE_DIR} # input dir for sphinx
            ${CMAKE_CURRENT_BINARY_DIR}/sphinx # output dir for sphinx
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating the Sphinx documentation")

    add_custom_target(docs
            DEPENDS doxygen sphinx)
endif()
