cmake_minimum_required(VERSION 3.9)

project(elsa VERSION 0.5
        DESCRIPTION "elsa recon toolbox"
        LANGUAGES CXX)


# ------------ elsa options ------------
# ------------
# detect if elsa is being as a submodule, enable/disable some options based on this
if(NOT DEFINED ELSA_MASTER_PROJECT)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(ELSA_MASTER_PROJECT ON)
    else()
        set(ELSA_MASTER_PROJECT OFF)
    endif()
endif ()

option(ELSA_TESTING "Enable building the unit tests" ${ELSA_MASTER_PROJECT})
option(ELSA_CREATE_JUNIT_REPORTS "Enable creating JUnit style reports when running tests" ON)
option(ELSA_COVERAGE "Enable test coverage computation and reporting" OFF)
option(ELSA_BENCHMARKS "Enable elsa benchmark test cases" OFF)

option(GIT_SUBMODULE "Enable updating the submodules during build" ${ELSA_MASTER_PROJECT})
option(ELSA_INSTALL "Enable generating the install targets for make install" ${ELSA_MASTER_PROJECT})
option(ELSA_BUILD_EXAMPLES "Enable building of examples" ${ELSA_MASTER_PROJECT})

option(ELSA_BUILD_CUDA_PROJECTORS "Enable building (or attempting to) the CUDA projectors" ON)
option(ELSA_BUILD_PYTHON_BINDINGS "Attempt to build python bindings if enabled (requires libclang-dev)" ON)
option(ELSA_BINDINGS_IN_SINGLE_MODULE "Combine the bindings for all elsa modules in a single python module" OFF)
option(ELSA_BUILD_WITH_MORE_WARNINGS "Enable all and extra warnings when building (-Wall -Wextra)" ON)

option(ELSA_CUDA_VECTOR "Build elsa with GPU DataContainer support and default" OFF)

set(ELSA_SANITIZER
        "" CACHE STRING
        "Compile with a sanitizer. Options are: Address, Memory, MemoryWithOrigins, Undefined, Thread, Leak, 'Address;Undefined'"
        )


# ------------ general setup -----------
# ------------

# add our cmake modules under cmake/
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# set default built type to "Release" (if no other specified)
set(DEFAULT_BUILD_TYPE "Release")
include(SetDefaultBuildType)
 
# Forbid extensions (such as gnu++) for this project 
set(CMAKE_CXX_EXTENSIONS OFF) 

# ------------ dependencies ------------
# ------------
 
# Setup list of submodules for elsa
list(APPEND ELSA_SUBMODULES Catch2 eigen3 pybind11 spdlog) 
 
if(ELSA_CUDA_VECTOR) 
    list(APPEND ELSA_SUBMODULES quickvec)
endif(ELSA_CUDA_VECTOR)

# only add the dependencies if elsa is stand-alone
if(ELSA_MASTER_PROJECT)
    # include the git submodule update
    include(UpdateGitSubmodules)

    # setup Eigen Library
    set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/eigen3")
    find_package(Eigen3 3.3 REQUIRED)
    message(STATUS "Using eigen3 submodule at ${EIGEN3_INCLUDE_DIR}")

    # setup spdlog Library
    set(Spdlog_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/spdlog")
    find_package(Spdlog REQUIRED)
    message(STATUS "Using spdlog submodule at ${Spdlog_INCLUDE_DIR}")

    # setup Dnnl library
    find_package(Dnnl)

    if(ELSA_CUDA_VECTOR)
        find_package(CUDA REQUIRED)
        find_package(Quickvec REQUIRED)
        message(STATUS "Using Quickvec submodule at ${Quickvec_INCLUDE_DIR}")
    endif()
else()
    message(STATUS "    No dependencies added for elsa, as it is not stand-alone")
endif()


# set where to install the exports/targets
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/elsa)


# include the InstallElsaModule function
include(InstallElsaModule)

add_subdirectory(thirdparty/pybind11)

# ------------ setup VectorCUDA --------
# ------------
if(ELSA_CUDA_VECTOR)
    set(CMAKE_CXX_EXTENSIONS OFF)

    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        enable_language(CUDA)

        message(STATUS "CUDA support enabled")
        message(STATUS "Enabling DataHandlerGPU")
        message(STATUS "The default container type will be on the GPU")

        set(ELSA_BUILD_WITH_QUICKVEC TRUE)
        
        add_definitions(-DELSA_CUDA_VECTOR)
        add_subdirectory(thirdparty/quickvec)
    else()
        set(ELSA_BUILD_WITH_QUICKVEC FALSE)

        message(STATUS "CUDA compiler not found or not compiling with clang!")
        message(STATUS "Compiling without a DataHandlerGPU")
        message(STATUS "The default container type will be CPU")
        set(ELSA_CUDA_VECTOR OFF)
    endif()
endif()

# ------------ Setup Tools -----------
# ------------

# Includes clang-format, clang-tidy, cmake-format, and sanitizers
include(Tools)


# ------------ setup testing -----------
# ------------

# if elsa is used as a submodule, turn testing off
if(NOT ELSA_MASTER_PROJECT)
    set(ELSA_TESTING OFF)
endif(NOT ELSA_MASTER_PROJECT)

if(ELSA_TESTING OR ELSA_BENCHMARKS)
    enable_testing()
    add_subdirectory(thirdparty/Catch2)

    # add the CMake modules for automatic test discovery
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/thirdparty/Catch2/contrib" ${CMAKE_MODULE_PATH})

    if(ELSA_TESTING)
        message(STATUS "elsa testing is enabled")

        # Run ctest to run all unit tests
        # All tests are includes ass dependencies to this target, such that they get build
        # schedule randomly that we don't start depending on some test order
        add_custom_target(tests
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            USES_TERMINAL
            COMMENT "Build and run all the tests.")
        if(ELSA_COVERAGE)
            message(STATUS "elsa test coverage is enabled")

            include(CodeCoverage)
            APPEND_COVERAGE_COMPILER_FLAGS()
            set(COVERAGE_LCOV_EXCLUDES '${PROJECT_SOURCE_DIR}/thirdparty/*' '/usr/*')
            SETUP_TARGET_FOR_COVERAGE_LCOV(NAME test_coverage EXECUTABLE ctest)

        else(ELSA_COVERAGE)
            message(STATUS "elsa test coverage is disabled")
        endif(ELSA_COVERAGE)
    endif(ELSA_TESTING)
else(ELSA_TESTING OR ELSA_BENCHMARKS)
    message(STATUS "    elsa testing is disabled")
endif(ELSA_TESTING OR ELSA_BENCHMARKS)



# ------------ add code/docs -----------
# ------------

# the bindings generator should be added before elsa, as we check whether the target can be built
if (ELSA_BUILD_PYTHON_BINDINGS)
    add_subdirectory(tools/bindings_generation EXCLUDE_FROM_ALL)

    if (ELSA_BINDINGS_IN_SINGLE_MODULE)
        add_definitions(-DELSA_BINDINGS_IN_SINGLE_MODULE)
    endif()
     
    # Warn if python bindings and code coverage is activated 
    if(ELSA_COVERAGE)
        message(WARNING "Python bindings are generated for a coverage build, consider turning one off (set \
        either \"ELSA_BUILD_PYTHON_BINDINGS\" or \"ELSA_COVERAGE\"to OFF via command line") 
    endif()
endif()

# the elsa library
add_subdirectory(elsa)

# the documentation
add_subdirectory(docs EXCLUDE_FROM_ALL)

if(ELSA_BUILD_EXAMPLES)
    # the examples
    add_subdirectory(examples EXCLUDE_FROM_ALL)
endif(ELSA_BUILD_EXAMPLES)
 
# Add benchmarks after elsa code, such that CUDA is detected correctly and we just add it in the benchmarks 
if (ELSA_BENCHMARKS)
    message(STATUS "elsa benchmarks are enabled")
    add_subdirectory(benchmarks)
endif(ELSA_BENCHMARKS)


# ------------ setup installation ------
# ------------

if(ELSA_INSTALL)
    # set up the target/library for make install
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # setup the ElsaConfig*.cmake files
    write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/elsa/elsaConfigVersion.cmake
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion
    )
    configure_package_config_file(
            ${CMAKE_CURRENT_LIST_DIR}/cmake/elsaConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/elsa/elsaConfig.cmake
            INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
    )

    # install the config files
    install(
            FILES
            ${CMAKE_CURRENT_BINARY_DIR}/elsa/elsaConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/elsa/elsaConfigVersion.cmake
            DESTINATION ${INSTALL_CONFIG_DIR}
    )

    # this puts the local build tree into the user package repository, but not the installed version...
    # ...but we rely on the paths in the installed version. sigh. commented out for now.
    #export(PACKAGE elsa)
endif(ELSA_INSTALL)


##
# Add uninstall target
# Remove all headers, library and CMake files, which where copied to the install location

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P "${PROJECT_SOURCE_DIR}/cmake/Uninstall.cmake"
    COMMENT "Uninstalling elsa (Removing all installed files)"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt"
    )
