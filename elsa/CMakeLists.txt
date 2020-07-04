cmake_minimum_required(VERSION 3.9)

# setup macro for component registration, so you can use find_package(elsa COMPONENT name)
# also sets all target names
macro(registerComponent name)
    set(ELSA_REGISTERED_COMPONENTS "${ELSA_REGISTERED_COMPONENTS};${name}" PARENT_SCOPE)
    set(ELSA_REGISTERED_TARGETS "${ELSA_REGISTERED_TARGETS};elsa_${name}" PARENT_SCOPE)
endmacro()


# macro for the unit tests
macro(ELSA_TEST NAME)
    # create the test executable
    add_executable(test_${NAME} EXCLUDE_FROM_ALL test_${NAME}.cpp test_main.cpp)
    # add catch and the corresponding elsa library
    target_link_libraries(test_${NAME} PRIVATE Catch2::Catch2 ${ELSA_MODULE_TARGET_NAME} elsa::test_routines)
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
    if(PYTHONLIBS_FOUND)
        add_custom_target(pyelsa)
    else()
        message(STATUS "Couldn't find Python.h. Python bindings will not be generated.")
    endif()

    # macro for generation of the corresponding python module for the elsa target TARGET_NAME
    # as a pre-build step the code for the python bindings will be generated and stored in BINDINGS_CODE_PATH
    # the sources containing the public interface of TARGET_NAME should be specified as additional arguments
    # you can omit a source file from the list to prevent the generation of bindings for that file
    macro(GENERATE_BINDINGS TARGET_NAME BINDINGS_CODE_PATH HINTS_PATH)
        if ((TARGET pyelsa) AND (TARGET pybind11_generator))

            get_filename_component(CMAKE_CURRENT_LIST_DIR_LAST_PART ${CMAKE_CURRENT_LIST_DIR} NAME)
            file(APPEND ${PROJECT_BINARY_DIR}/elsa/__init__.py "from .${CMAKE_CURRENT_LIST_DIR_LAST_PART} import *\n")

            add_custom_command(OUTPUT ${BINDINGS_CODE_PATH}
                COMMAND ${PROJECT_BINARY_DIR}/tools/bindings_generation/pybind11_generator
                    ${ARGN}
                    --extra-arg=-I${PYTHON_INCLUDE_DIRS}
                    --extra-arg=-I${PYBIND11_INCLUDE_DIR}
                    -p=${CMAKE_BINARY_DIR}
                    --hints=${HINTS_PATH}
                    -o=${BINDINGS_CODE_PATH}
                    --name=${TARGET_NAME}
                DEPENDS ${TARGET_NAME} pybind11_generator ${HINTS_PATH}
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                COMMENT "Generating bindings code for ${TARGET_NAME}"
                VERBATIM)

            pybind11_add_module(py${TARGET_NAME} ${BINDINGS_CODE_PATH})
            target_include_directories(py${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/elsa/bindings/hints)
            target_link_libraries(py${TARGET_NAME} PUBLIC ${TARGET_NAME})
            target_compile_features(py${TARGET_NAME} PUBLIC cxx_std_17)
            add_dependencies(pyelsa py${TARGET_NAME})
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
add_subdirectory(solvers)
add_subdirectory(projectors)
if(ELSA_BUILD_CUDA_PROJECTORS)
    add_subdirectory(projectors_cuda)
endif(ELSA_BUILD_CUDA_PROJECTORS)
add_subdirectory(generators)
add_subdirectory(test_routines)


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
