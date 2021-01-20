# setup macro for component registration, so you can use find_package(elsa COMPONENT name)
# also sets all target names
macro(registerComponent name)
    set(ELSA_REGISTERED_COMPONENTS "${ELSA_REGISTERED_COMPONENTS};${name}" PARENT_SCOPE)
    set(ELSA_REGISTERED_TARGETS "${ELSA_REGISTERED_TARGETS};elsa_${name}" PARENT_SCOPE)
endmacro()


# macro for the unit tests
macro(ELSA_TEST NAME)
    # Parse arguments
    set(options )
    set(oneValueArgs )
    set(multiValueArgs DEPENDENCIES)
    cmake_parse_arguments(MYTEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # create the test executable
    add_executable(test_${NAME} EXCLUDE_FROM_ALL test_${NAME}.cpp test_main.cpp)
    # add catch and the corresponding elsa library plus optional dependencies
    target_link_libraries(test_${NAME} PRIVATE Catch2::Catch2 ${ELSA_MODULE_TARGET_NAME} elsa::test_routines ${MYTEST_DEPENDENCIES})
    # enable C++17
    target_compile_features(test_${NAME} PUBLIC cxx_std_17)
    # include helpers
    target_include_directories(test_${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/elsa/test_routines/)

    # if we use JUnit reporter handle arguments
    if(${ELSA_CREATE_JUNIT_REPORTS})
        file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test_reports)
        set(ELSA_JUNIT_ARGUMENTS "-r junit" "-o ${PROJECT_BINARY_DIR}/test_reports/test_${NAME}.xml")
    endif(${ELSA_CREATE_JUNIT_REPORTS})
    add_dependencies(tests test_${NAME})

    # let Catch discover and register all the test cases
    catch_discover_tests(test_${NAME} TEST_SPEC ${ELSA_JUNIT_ARGUMENTS})
endmacro(ELSA_TEST)

if (ELSA_BUILD_PYTHON_BINDINGS)
    find_package(PythonLibs)
    if(PYTHONLIBS_FOUND AND (TARGET pybind11_generator))
        set(ELSA_PYTHON_BINDINGS_PATH "${CMAKE_BINARY_DIR}/pyelsa" CACHE INTERNAL "Output directory for pybind11 modules")

        # remove __init__.py and logger.py files if present
        file(REMOVE ${ELSA_PYTHON_BINDINGS_PATH}/__init__.py)
        file(REMOVE ${ELSA_PYTHON_BINDINGS_PATH}/logger.py)

        if (CMAKE_CXX_FLAGS)
            string(REPLACE " " ";" CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS})
        endif()

        # always combine bindings in a single module if using libc++
        foreach(COMPILE_OPTION IN LISTS CXX_FLAGS_LIST)
            if (${COMPILE_OPTION} MATCHES "-stdlib=libc\\+\\+.*")
                set(ELSA_BINDINGS_IN_SINGLE_MODULE ON CACHE BOOL "Bindings compiled in single module as libc++ is used" FORCE)
            endif()
        endforeach()

        if (ELSA_BINDINGS_IN_SINGLE_MODULE)
            # bind_elsa.cpp combines all bindings definitions into a single PYBIND11_MODULE
            file(WRITE ${ELSA_PYTHON_BINDINGS_PATH}/bind_elsa.cpp "#include <pybind11/pybind11.h>\n\n")

            pybind11_add_module(pyelsa ${ELSA_PYTHON_BINDINGS_PATH}/bind_elsa.cpp)
            set_target_properties(pyelsa PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${ELSA_PYTHON_BINDINGS_PATH})
            target_include_directories(pyelsa PUBLIC ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints)
            target_compile_definitions(pyelsa PRIVATE ELSA_BINDINGS_IN_SINGLE_MODULE)
            target_link_libraries(pyelsa PUBLIC elsa::all)
            target_compile_features(pyelsa PUBLIC cxx_std_17)
        else()
            add_custom_target(pyelsa)
        endif()

        if (ELSA_TESTING)
            file(COPY ${PROJECT_SOURCE_DIR}/tools/bindings_generation/tests DESTINATION ${PROJECT_BINARY_DIR}/pyelsa)

            add_custom_target(test_pyelsa DEPENDS pyelsa)

            add_custom_command(TARGET test_pyelsa
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_SOURCE_DIR}/tools/bindings_generation/tests/test_pyelsa.py
                    ${PROJECT_BINARY_DIR}/pyelsa/tests
                DEPENDS ${PROJECT_SOURCE_DIR}/tools/bindings_generation/tests/test_pyelsa.py)

            add_dependencies(tests test_pyelsa)

            add_test(NAME test_pyelsa
                COMMAND ${PYTHON_EXECUTABLE} -m unittest pyelsa/tests/test_pyelsa.py
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()
    else()
        message(STATUS "Couldn't find Python.h. Python bindings will not be generated.")
    endif()

    # macro for generation of the corresponding python module for the elsa target TARGET_NAME
    # as a pre-build step the code for the python bindings will be generated and stored in the directory
    # ELSA_PYTHON_BINDINGS_PATH under the name BINDINGS_CODE_FILENAME
    # the sources containing the public interface of TARGET_NAME should be specified as additional arguments
    # you can omit a source file from the list to prevent the generation of bindings for that file
    macro(GENERATE_BINDINGS TARGET_NAME BINDINGS_CODE_FILENAME HINTS_PATH)
        if ((TARGET pyelsa) AND (TARGET pybind11_generator))
            set(HINTS_FILE ${HINTS_PATH})

            # when compiling with libc++ as the STL, the path to the STL headers must be specified
            set(ADDITIONAL_SYSTEM_INCLUDE_PATHS_FIX "")
            get_target_property(TARGET_COMPILE_OPTIONS ${TARGET_NAME} COMPILE_OPTIONS)
            if (CMAKE_CXX_FLAGS)
                string(REPLACE " " ";" CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS})
            endif()

            # search through the global and target specific compile options for the "-stdlib=libc++" flag
            foreach(COMPILE_OPTION IN LISTS CXX_FLAGS_LIST TARGET_COMPILE_OPTIONS)
                if (${COMPILE_OPTION} MATCHES "-stdlib=libc\\+\\+.*")
                    set(ADDITIONAL_SYSTEM_INCLUDE_PATHS_FIX --extra-arg=-isystem --extra-arg=${LIBCXX_INCLUDE_DIR})
                    break()
                endif()
            endforeach(COMPILE_OPTION)

            if (ELSA_BINDINGS_IN_SINGLE_MODULE)
                set(SINGLE_MODULE_FLAGS --extra-arg=-DELSA_BINDINGS_IN_SINGLE_MODULE --no-module)
            endif()

            set(PY_TARGET_NAME "py${TARGET_NAME}")

            if (ELSA_CUDA_VECTOR)
                if (NOT ${HINTS_PATH} STREQUAL "")
                    get_filename_component(HINTS_NAME ${HINTS_PATH} NAME)
                    get_filename_component(HINTS_CU_NAME ${HINTS_PATH} NAME_WE)
                    set(HINTS_CU_NAME "${HINTS_CU_NAME}.cu")
                    set(HINTS_FILE "${ELSA_PYTHON_BINDINGS_PATH}/${HINTS_CU_NAME}")

                    add_custom_command(OUTPUT ${HINTS_FILE}
                        COMMAND ${CMAKE_COMMAND} -E copy ${HINTS_PATH} ${HINTS_FILE}
                        DEPENDS ${HINTS_PATH}
                    )
                endif()
            endif()

            add_custom_command(OUTPUT ${ELSA_PYTHON_BINDINGS_PATH}/${BINDINGS_CODE_FILENAME}
                COMMAND pybind11_generator
                    ${ARGN}
                    ${ADDITIONAL_SYSTEM_INCLUDE_PATHS_FIX}
                    --extra-arg=-isystem
                    --extra-arg=${CLANG_RESOURCE_DIR}/include
                    --extra-arg=-I${PYTHON_INCLUDE_DIRS}
                    --extra-arg=-I${PYBIND11_INCLUDE_DIR}
                    --extra-arg=-I${CMAKE_CURRENT_LIST_DIR}
                    --extra-arg=-I${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints
                    --extra-arg=-std=c++17
                    --extra-arg=-w
                    --extra-arg=--cuda-host-only
                    -p=${CMAKE_BINARY_DIR}
                    --hints=${HINTS_FILE}
                    -o=${ELSA_PYTHON_BINDINGS_PATH}/${BINDINGS_CODE_FILENAME}
                    --pyname=${PY_TARGET_NAME}
                    ${SINGLE_MODULE_FLAGS}
                DEPENDS ${TARGET_NAME} pybind11_generator ${HINTS_FILE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                COMMENT "Generating bindings code for ${TARGET_NAME}"
                VERBATIM)

            if (NOT ELSA_BINDINGS_IN_SINGLE_MODULE)
                pybind11_add_module(${PY_TARGET_NAME} ${ELSA_PYTHON_BINDINGS_PATH}/${BINDINGS_CODE_FILENAME} ${HINTS_FILE})
                set_target_properties(${PY_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${ELSA_PYTHON_BINDINGS_PATH})
                target_include_directories(${PY_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints)
                target_link_libraries(${PY_TARGET_NAME} PUBLIC ${TARGET_NAME})
                target_compile_features(${PY_TARGET_NAME} PUBLIC cxx_std_17)
                add_dependencies(pyelsa ${PY_TARGET_NAME})

                file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/__init__.py "from .${PY_TARGET_NAME} import *\n")
            else()
                add_custom_target(${PY_TARGET_NAME} DEPENDS ${ELSA_PYTHON_BINDINGS_PATH}/${BINDINGS_CODE_FILENAME})
                add_dependencies(pyelsa ${PY_TARGET_NAME})

                file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/bind_elsa.cpp "#include \"${BINDINGS_CODE_FILENAME}\"\n")
            endif()

            set(ELSA_PYTHON_MODULES "${ELSA_PYTHON_MODULES};${PY_TARGET_NAME}" PARENT_SCOPE)
        endif()
    endmacro()
endif()


# add the elsa modules
add_subdirectory(core)
add_subdirectory(logging)
add_subdirectory(io)
add_subdirectory(operators)
add_subdirectory(functionals)
add_subdirectory(problems)
add_subdirectory(proximity_operators)
add_subdirectory(solvers)
add_subdirectory(projectors)
if(ELSA_BUILD_CUDA_PROJECTORS)
    add_subdirectory(projectors_cuda)
endif(ELSA_BUILD_CUDA_PROJECTORS)
add_subdirectory(generators)
add_subdirectory(test_routines)
add_subdirectory(test_allmodules)

if(ELSA_BUILD_PYTHON_BINDINGS AND PYTHONLIBS_FOUND AND (TARGET pybind11_generator))
    if (ELSA_BINDINGS_IN_SINGLE_MODULE)
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/bind_elsa.cpp "\nPYBIND11_MODULE(pyelsa, m)\n{\n")
        list(REMOVE_ITEM ELSA_PYTHON_MODULES "")
        foreach(PYMODULE IN LISTS ELSA_PYTHON_MODULES)
            file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/bind_elsa.cpp "\tadd_definitions_${PYMODULE}(m);\n")
        endforeach()
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/bind_elsa.cpp "}\n")

        file(WRITE ${ELSA_PYTHON_BINDINGS_PATH}/__init__.py "from .pyelsa import *")
    else()
        # when split in multiple modules provide a top level Logger interface
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "from . import *\n\n")
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "class Logger(object):\n")

        # remove the empty item and core module from the list (core doesn't use logging)
        list(REMOVE_ITEM ELSA_PYTHON_MODULES "")
        list(FILTER ELSA_PYTHON_MODULES EXCLUDE REGEX ".*core.*")

        # define setLevel function
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "\tdef setLevel(level: LogLevel):\n")
        foreach(PYMODULE IN LISTS ELSA_PYTHON_MODULES)
            file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "\t\tlogger_${PYMODULE}.setLevel(level)\n")
        endforeach()
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "\n")

        # define enableFileLogging function
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "\tdef enableFileLogging(filename: str):\n")
        foreach(PYMODULE IN LISTS ELSA_PYTHON_MODULES)
            file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "\t\tlogger_${PYMODULE}.enableFileLogging(filename)\n")
        endforeach()
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "\n")

        # define flush function
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "\tdef flush():\n")
        foreach(PYMODULE IN LISTS ELSA_PYTHON_MODULES)
            file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/logger.py "\t\tlogger_${PYMODULE}.flush()\n")
        endforeach()

        # add Logger to __init__.py file
        file(APPEND ${ELSA_PYTHON_BINDINGS_PATH}/__init__.py "\nfrom .logger import Logger\n")
    endif()
endif()

# library to build and add all registered components of the library
add_library(elsa_all INTERFACE)
add_library(elsa::all ALIAS elsa_all)

# link against all modules
target_link_libraries(elsa_all INTERFACE ${ELSA_REGISTERED_TARGETS})
target_include_directories(elsa_all INTERFACE
        $<INSTALL_INTERFACE:include/elsa/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        )

# register the all "module"
registerComponent(all)

# install the all module and the elsa.h file
InstallElsaModule(all elsa_all elsa_allTargets)
install(FILES elsa.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/elsa/)


# if stand-alone and option set, turn on all warnings for all components
if(${ELSA_MASTER_PROJECT} AND ${ELSA_BUILD_WITH_MORE_WARNINGS})
    list(REMOVE_ITEM ELSA_REGISTERED_COMPONENTS "")
    foreach(_component ${ELSA_REGISTERED_COMPONENTS})
        if (NOT(_component MATCHES "projector_kernels"))
            if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
                target_compile_options(elsa_${_component} PUBLIC -Wall -Wextra -Wconversion -pedantic -Wfatal-errors)
            endif()
            if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                target_compile_options(elsa_${_component} PUBLIC /W3 /WX)
            endif()
        endif()
    endforeach(_component ELSA_REGISTERED_COMPONENTS)
endif()
