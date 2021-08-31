include(CompilerWarnings) 
 
# setup macro for component registration, so you can use find_package(elsa COMPONENT name) also sets all target names
macro(REGISTER_MODULE name)
    set(ELSA_REGISTERED_COMPONENTS "${ELSA_REGISTERED_COMPONENTS};${name}" PARENT_SCOPE)
    set(ELSA_REGISTERED_TARGETS "${ELSA_REGISTERED_TARGETS};elsa_${name}" PARENT_SCOPE)
endmacro()

# Simple macro to add a module
macro(ADD_ELSA_MODULE elsa_module_name module_headers module_sources)
    # Parse arguments
    set(options INSTALL_DIR NO_TESTS)
    set(oneValueArgs)
    set(multiValueArgs PUBLIC_DEPS PRIVATE_DEPS)
    cmake_parse_arguments(ADD_MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Make name persistent after macro
    set(ELSA_MODULE_NAME ${elsa_module_name})

    # set the name of the module
    set(ELSA_MODULE_TARGET_NAME elsa_${elsa_module_name})
    set(ELSA_MODULE_EXPORT_TARGET elsa_${elsa_module_name}Targets)

    # build the module library
    add_library(${ELSA_MODULE_TARGET_NAME} ${module_headers} ${module_sources})
    add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

    # Add public dependencies
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC "${ADD_MODULE_PUBLIC_DEPS}")
    # Add private dependencies
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE "${ADD_MODULE_PRIVATE_DPES}")
    # require C++17
    target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17)
    # set -fPIC and -fvisibility=hidden
    set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                                CXX_VISIBILITY_PRESET hidden)
    
    target_include_directories(
        ${ELSA_MODULE_TARGET_NAME} PUBLIC $<INSTALL_INTERFACE:include/elsa/${elsa_module_name}>
                                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )

    if(ELSA_MASTER_PROJECT)
        set_target_warnings(${ELSA_MODULE_TARGET_NAME} LEVEL ${ELSA_WARNINGS_LEVEL}) 
    endif()

    # build the tests (if enabled)
    if(ELSA_TESTING AND NOT ADD_MODULE_NO_TESTS)
        add_subdirectory(tests)
    endif()

    # register the module
    REGISTER_MODULE(${elsa_module_name})

    # install the module
    # if INSTALL_DIR arg is set, recurse the whole directory
    if(${ADD_MODULE_INSTALL_DIR})
        install_elsa_module_dir(
            ${elsa_module_name} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/"
        )
    else()
        install_elsa_module(${elsa_module_name} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})
    endif()
endmacro()

# macro for the unit tests
macro(ELSA_DOCTEST name)
    # Parse arguments
    set(options)
    set(oneValueArgs)
    set(multiValueArgs PUBLIC_DEPENDENCIES PRIVATE_DEPENDENCIES)
    cmake_parse_arguments(ELSA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # set the name for the test once
    set(_testname test_${name})
    # create the test executable
    add_executable(${_testname} EXCLUDE_FROM_ALL ${_testname}.cpp test_docmain.cpp)
    # add catch and the corresponding elsa library
    target_link_libraries(
        ${_testname} PRIVATE doctest::doctest ${ELSA_MODULE_TARGET_NAME} elsa::test_routines
                             ${ELSA_TEST_PUBLIC_DEPENDENCIES}
    )
    # enable C++17
    target_compile_features(${_testname} PUBLIC cxx_std_17)
    # include helpers
    target_include_directories(${_testname} PRIVATE ${CMAKE_SOURCE_DIR}/elsa/test_routines/)
    # Improve compile times for tests
    target_compile_definitions(${_testname} PRIVATE DOCTEST_CONFIG_SUPER_FAST_ASSERTS)
    # Set output directory to bin/tests/<module>
    set_target_properties(
        ${_testname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests/${ELSA_MODULE_TARGET_NAME}"
    )

    # Add test to global dependency
    add_dependencies(tests ${_testname})
    add_dependencies(build-tests ${_testname})

    # if we use JUnit reporter handle arguments
    if(${ELSA_CREATE_JUNIT_REPORTS})
        file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test_reports)
        set(ELSA_JUNIT_OUTPUT_DIR "${PROJECT_BINARY_DIR}/test_reports/${_testname}.xml")
    endif(${ELSA_CREATE_JUNIT_REPORTS})

    # let doctest discover the test and add the JUNIT arguments
    doctest_discover_tests(${_testname} JUNIT_OUTPUT_DIR "${ELSA_JUNIT_OUTPUT_DIR}")

    # Add test to the corresponding fine grained dependency
    if(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_core")
        add_dependencies(build-tests-core ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_functionals")
        add_dependencies(build-tests-functionals ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_logging")
        add_dependencies(build-tests-logging ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_io")
        add_dependencies(build-tests-io ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_operators")
        add_dependencies(build-tests-operators ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_problems")
        add_dependencies(build-tests-problems ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_proximity_operators")
        add_dependencies(build-tests-proximity_operators ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_solvers")
        add_dependencies(build-tests-solvers ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_generators")
        add_dependencies(build-tests-generators ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_projectors")
        add_dependencies(build-tests-projectors ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_projectors_cuda")
        add_dependencies(build-tests-projectors_cuda ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_ml")
        add_dependencies(build-tests-ml ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_ml_dnnl_backend")
        add_dependencies(build-tests-ml_dnnl ${_testname})
    elseif(${ELSA_MODULE_TARGET_NAME} STREQUAL "elsa_ml_cudnn_backend")
        add_dependencies(build-tests-ml_cudnn ${_testname})
    else()
        message(
            WARNING "No test build target for module \"${ELSA_MODULE_TARGET_NAME}\" when adding test \"${_testname}\""
        )
    endif()
endmacro()

include(PythonBindings)

if(ELSA_BUILD_PYTHON_BINDINGS)
    SetupPythonBindings()
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
add_subdirectory(ml)
add_subdirectory(tasks)
if(ELSA_BUILD_CUDA_PROJECTORS)
    add_subdirectory(projectors_cuda)
endif(ELSA_BUILD_CUDA_PROJECTORS)
add_subdirectory(generators)
add_subdirectory(test_routines)
add_subdirectory(test_allmodules)

if(ELSA_BUILD_PYTHON_BINDINGS)
    AddLoggerToBindings()
endif()

# library to build and add all registered components of the library
add_library(elsa_all INTERFACE)
add_library(elsa::all ALIAS elsa_all)

# link against all modules
target_link_libraries(elsa_all INTERFACE ${ELSA_REGISTERED_TARGETS})
target_include_directories(
    elsa_all INTERFACE $<INSTALL_INTERFACE:include/elsa/> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
 
write_module_config(all DEPENDENCIES ${ELSA_REGISTERED_TARGETS})

# register the all "module"
REGISTER_MODULE(all)

# install the all module and the elsa.h file
install_elsa_module(all elsa_all elsa_allTargets)
install(FILES elsa.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/elsa/)
