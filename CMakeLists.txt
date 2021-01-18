cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

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

# Include CPM dependency manager
include(CPM)

# export compile_commands.json for language servers such as ccls
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------ dependencies ------------
# ------------

# only add the dependencies if elsa is stand-alone
if(ELSA_MASTER_PROJECT)
    # setup Eigen Library
    CPMAddPackage(
        NAME eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen
        # Currently we're on a random master branch commit, until Eigen 3.4 will be released we kind of have to
        # stick with it, see also https://gitlab.lrz.de/IP/elsa/-/issues/91 for a little discussion on it
        GIT_TAG c1b0aea65
        DOWNLOAD_ONLY YES  # Eigen's CMakelists are not intended for library use
    )

    # Add the Eigen library target to link against
    if(eigen3_ADDED)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${eigen3_SOURCE_DIR})
    endif()

    # setup spdlog Library
    CPMAddPackage(
        NAME Spdlog
        GITHUB_REPOSITORY gabime/spdlog
        VERSION 1.7.0 # 1.7.0 Fallback if it dosen't work
        OPTIONS "SPDLOG_INSTALL ON"
    )

    # setup Dnnl library
    find_package(Dnnl)

    # Adding quickvec
    if(ELSA_CUDA_VECTOR)
        include(CheckLanguage)
        check_language(CUDA)

        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            CPMAddPackage(
                NAME Quickvec
                GIT_REPOSITORY https://gitlab.lrz.de/IP/quickvec.git
                GIT_TAG master
            )

            if(Quickvec_ADDED)
                message(STATUS "CUDA support enabled")
                message(STATUS "Enabling DataHandlerGPU")
                message(STATUS "The default container type will be on the GPU")

                set(ELSA_BUILD_WITH_QUICKVEC TRUE)
                add_definitions(-DELSA_CUDA_VECTOR)
            else()
                message(STATUS "CUDA compiler not found or not compiling with clang!")
                message(STATUS "Compiling without a DataHandlerGPU")
                message(STATUS "The default container type will be CPU")

                set(ELSA_BUILD_WITH_QUICKVEC FALSE)
                set(ELSA_CUDA_VECTOR OFF)
            endif()
        endif()
    endif()

    if (${ELSA_BUILD_PYTHON_BINDINGS})
        CPMAddPackage(
            NAME pybind11
            GITHUB_REPOSITORY pybind/pybind11
            VERSION 2.6.0
        )
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

    CPMAddPackage(
         NAME Catch2
         GITHUB_REPOSITORY catchorg/Catch2
         VERSION 2.10.1 # <-- Current, update to: 2.13.1
    )

    # add the CMake modules for automatic test discovery
    set(CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/contrib" ${CMAKE_MODULE_PATH})

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
            set(COVERAGE_LCOV_EXCLUDES
                '${CPM_SOURCE_CACHE}/*'
                '${PROJECT_BINARY_DIR}/_deps/'
                '${PROJECT_SOURCE_DIR}/thirdparty/*'
                '/usr/*')
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

# the examples
if(ELSA_BUILD_EXAMPLES)
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
