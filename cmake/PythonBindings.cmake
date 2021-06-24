find_package(PythonLibs)

# macro for generation of the corresponding python module for the elsa target TARGET_NAME as a pre-build step the
# code for the python bindings will be generated and stored in the directory ELSA_PYTHON_BINDINGS_PATH under the
# additional arguments you can omit a source file from the list to prevent the generation of bindings for that file
macro(GENERATE_BINDINGS TARGET_NAME BINDINGS_CODE_FILENAME HINTS_PATH)
    if((TARGET pyelsa) AND (TARGET pybind11_generator))
        set(HINTS_FILE ${HINTS_PATH})
        # when compiling with libc++ as the STL, the path to the STL headers must be specified
        set(ADDITIONAL_SYSTEM_INCLUDE_PATHS_FIX "")
        get_target_property(TARGET_COMPILE_OPTIONS ${TARGET_NAME} COMPILE_OPTIONS)
        if(CMAKE_CXX_FLAGS)
            string(REPLACE " " ";" CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS})
        endif()

        # search through the global and target specific compile options for the "-stdlib=libc++" flag
        foreach(COMPILE_OPTION IN LISTS CXX_FLAGS_LIST TARGET_COMPILE_OPTIONS)
            if(${COMPILE_OPTION} MATCHES "-stdlib=libc\\+\\+.*")
                set(ADDITIONAL_SYSTEM_INCLUDE_PATHS_FIX --extra-arg=-isystem --extra-arg=${LIBCXX_INCLUDE_DIR})
                break()
            endif()
        endforeach(COMPILE_OPTION)

        if(ELSA_BINDINGS_IN_SINGLE_MODULE)
            set(SINGLE_MODULE_FLAGS --extra-arg=-DELSA_BINDINGS_IN_SINGLE_MODULE --no-module)
        endif()

        set(PY_TARGET_NAME "py${TARGET_NAME}")

        if(ELSA_CUDA_VECTOR AND NOT ${HINTS_PATH} STREQUAL "")
            get_filename_component(HINTS_NAME ${HINTS_PATH} NAME)
            get_filename_component(HINTS_CU_NAME ${HINTS_PATH} NAME_WE)
            set(HINTS_CU_NAME "${HINTS_CU_NAME}.cu")
            set(HINTS_FILE "${ELSA_PYTHON_BINDINGS_PATH}/${HINTS_CU_NAME}")

            add_custom_command(
                OUTPUT ${HINTS_FILE} COMMAND ${CMAKE_COMMAND} -E copy ${HINTS_PATH} ${HINTS_FILE}
                DEPENDS ${HINTS_PATH}
            )
        endif()

        add_custom_command(
            OUTPUT ${ELSA_PYTHON_BINDINGS_PATH}/${BINDINGS_CODE_FILENAME}
            COMMAND
                pybind11_generator ${ARGN} ${ADDITIONAL_SYSTEM_INCLUDE_PATHS_FIX} --extra-arg=-isystem
                --extra-arg=${CLANG_RESOURCE_DIR}/include --extra-arg=-std=c++17 --extra-arg=-w
                --extra-arg=--cuda-host-only -p=${CMAKE_BINARY_DIR} --hints=${HINTS_FILE}
                --extra-arg=-DEIGEN_DONT_PARALLELIZE
                -o=${ELSA_PYTHON_BINDINGS_PATH}/${BINDINGS_CODE_FILENAME} --pyname=${PY_TARGET_NAME}
                ${SINGLE_MODULE_FLAGS}
            DEPENDS ${TARGET_NAME} pybind11_generator ${HINTS_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Generating bindings code for ${TARGET_NAME}"
            VERBATIM
        )

        if(NOT ELSA_BINDINGS_IN_SINGLE_MODULE)
            pybind11_add_module(${PY_TARGET_NAME} ${ELSA_PYTHON_BINDINGS_PATH}/${BINDINGS_CODE_FILENAME})
            set_target_properties(
                ${PY_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${ELSA_PYTHON_BINDINGS_PATH}
            )
            target_include_directories(
                ${PY_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints
            )
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
 
function(SetupPythonBindings)
    if(PYTHONLIBS_FOUND AND (TARGET pybind11_generator))
        set(ELSA_PYTHON_BINDINGS_PATH "${CMAKE_BINARY_DIR}/pyelsa" CACHE INTERNAL
                                                                         "Output directory for pybind11 modules"
        )

        # remove __init__.py and logger.py files if present
        file(REMOVE ${ELSA_PYTHON_BINDINGS_PATH}/__init__.py)
        file(REMOVE ${ELSA_PYTHON_BINDINGS_PATH}/logger.py)

        if(CMAKE_CXX_FLAGS)
            string(REPLACE " " ";" CXX_FLAGS_LIST ${CMAKE_CXX_FLAGS})
        endif()

        # always combine bindings in a single module if using libc++
        set (USES_LIBCXX "FALSE")
        if (CMAKE_CXX_COMPILER_ID MATCHES "Apple[Cc]lang")
            set (USES_LIBCXX "TRUE")
            foreach(COMPILE_OPTION IN LISTS CXX_FLAGS_LIST)
                if(${COMPILE_OPTION} MATCHES "-stdlib=libstdc\\+\\+")
                    set(USES_LIBCXX "FALSE")
                    break()
                endif()
            endforeach()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "^[Cc]lang")
            foreach(COMPILE_OPTION IN LISTS CXX_FLAGS_LIST)
                if(${COMPILE_OPTION} MATCHES "-stdlib=libc\\+\\+")
                    set(USES_LIBCXX "TRUE")
                    break()
                endif()
        endforeach()
        endif()

        if (USES_LIBCXX)
            set(ELSA_BINDINGS_IN_SINGLE_MODULE ON CACHE BOOL "Bindings compiled in single module as libc++ is used"
                                                        FORCE
            )
        endif()

        if(ELSA_BINDINGS_IN_SINGLE_MODULE)
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

        if(ELSA_TESTING)
            file(COPY ${PROJECT_SOURCE_DIR}/tools/bindings_generation/tests DESTINATION ${PROJECT_BINARY_DIR}/pyelsa)

            add_custom_target(test_pyelsa DEPENDS pyelsa)

            add_custom_command(
                TARGET test_pyelsa
                PRE_BUILD
                COMMAND
                    ${CMAKE_COMMAND} -E copy_if_different
                    ${PROJECT_SOURCE_DIR}/tools/bindings_generation/tests/test_pyelsa.py
                    ${PROJECT_BINARY_DIR}/pyelsa/tests DEPENDS
                    ${PROJECT_SOURCE_DIR}/tools/bindings_generation/tests/test_pyelsa.py
            )

            add_dependencies(tests test_pyelsa)

            add_test(NAME test_pyelsa COMMAND ${PYTHON_EXECUTABLE} -m unittest pyelsa/tests/test_pyelsa.py
                     WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            )
        endif()
    else()
        message(STATUS "Couldn't find Python.h. Python bindings will not be generated.")
    endif()
endfunction()

# Logger needs special care to make it work correctly 
function(AddLoggerToBindings)
    if(PYTHONLIBS_FOUND AND (TARGET pybind11_generator))
        if(ELSA_BINDINGS_IN_SINGLE_MODULE)
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
endfunction()
