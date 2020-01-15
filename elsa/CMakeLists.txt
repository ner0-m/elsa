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
    target_link_libraries(test_${NAME} PRIVATE Catch2::Catch2 ${ELSA_MODULE_TARGET_NAME})
    # enable C++17
    target_compile_features(test_${NAME} PUBLIC cxx_std_17)

    # if we use JUnit reporter handle arguments
    if(${ELSA_CREATE_JUNIT_REPORTS})
        file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test_reports)
        set(ELSA_JUNIT_ARGUMENTS "-r junit" "-o ${PROJECT_BINARY_DIR}/test_reports/test_${NAME}.xml")
    endif(${ELSA_CREATE_JUNIT_REPORTS})
    add_dependencies(tests test_${NAME})

    # let Catch discover and register all the test cases
    catch_discover_tests(test_${NAME} TEST_SPEC ${ELSA_JUNIT_ARGUMENTS})
endmacro(ELSA_TEST)


# add sanitizers if in debug mode
if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    include(${PROJECT_SOURCE_DIR}/cmake/Sanitizers.cmake)
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
