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
option(ELSA_COVERAGE "Enable test coverage computation and reporting" OFF)

option(ELSA_INSTALL "Enable generating the install targets for make install" ${ELSA_MASTER_PROJECT})
option(ELSA_BUILD_EXAMPLES "Enable building of examples" ${ELSA_MASTER_PROJECT})

option(ELSA_BUILD_CUDA_PROJECTORS "Enable building (or attempting to) the CUDA projectors" ON)
option(ELSA_BUILD_PYTHON_BINDINGS "Attempt to build python bindings if enabled (requires libclang-dev)" ON)
option(ELSA_BINDINGS_IN_SINGLE_MODULE "Combine the bindings for all elsa modules in a single python module" OFF)

option(ELSA_CUDA_VECTOR "Build elsa with GPU DataContainer support and default" OFF)

set(ELSA_WARNING_LEVEL "base" CACHE STRING "Warning level with which elsa is build. Options are: base, all")
set(ELSA_SANITIZER
        "" CACHE STRING
        "Compile with a sanitizer. Options are: Address, Memory, MemoryWithOrigins, Undefined, Thread, Leak, 'Address;Undefined'"
        )

set(ELSA_CUDA_ARCH_TYPE "auto" CACHE STRING "Set CUDA architectures")

option(SYSTEM_EIGEN "Build elsa using the system eigen installation" OFF)
option(SYSTEM_SPDLOG "Build elsa using the system spdlog installation" OFF)
option(SYSTEM_DOCTEST "Build elsa using the system doctest installation" OFF)
option(SYSTEM_CATCH "Build elsa using the system catch installation" OFF)

# ------------ general setup -----------
# ------------

# add our cmake modules under cmake/
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# set default built type to "Release" (if no other specified)
set(DEFAULT_BUILD_TYPE "Release")
include(SetDefaultBuildType)

# Forbid extensions (such as gnu++) for this project
set(CMAKE_CXX_EXTENSIONS OFF)

# export compile_commands.json for lsp language servers (e.g. clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# By default set all output to lib or bin directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set where to install the exports/targets
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/elsa)

# ------------ dependencies ------------
# ------------

if (NOT (SYSTEM_EIGEN AND SYSTEM_SPDLOG AND SYSTEM_DOCTEST AND SYSTEM_CATCH))
    # Include CPM dependency manager for bundling dependencies
    include(CPM)
endif()


# only add the dependencies if elsa is stand-alone
if(ELSA_MASTER_PROJECT)
    if(SYSTEM_EIGEN)
        message(STATUS "Using system-wide Eigen library...")
        find_package(Eigen3 REQUIRED)
    else()
        # setup custom Eigen Library
        # once there's a proper release,
        # the system-eigen has to be a drop-in replacement for the git-version
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
            message(STATUS "Using bundled Eigen version in ${eigen3_SOURCE_DIR}")
            # this target is set-up as a drop-in replacement
            # for a system eigen package.
            set(EIGEN_BUNDLED_INSTALLDIR "elsa/Eigen")
            add_library(Eigen INTERFACE)
            add_library(Eigen3::Eigen ALIAS Eigen)

            target_include_directories(Eigen
                SYSTEM      # include with -isystem so many warnings are swallowed!
                INTERFACE
                $<BUILD_INTERFACE:${eigen3_SOURCE_DIR}>
                $<INSTALL_INTERFACE:${EIGEN_BUNDLED_INSTALLDIR}>
            )

            # register eigen in the build tree
            export(TARGETS Eigen FILE "${CMAKE_CURRENT_BINARY_DIR}/elsa/eigen3Targets.cmake")

            # install custom elsa
            if(ELSA_INSTALL)
                install(TARGETS Eigen EXPORT eigenExport)
                install(EXPORT eigenExport
                    NAMESPACE Eigen3::
                    FILE "eigen3Targets.cmake"
                    DESTINATION ${INSTALL_CONFIG_DIR}
                )

                install(DIRECTORY "${eigen3_SOURCE_DIR}/Eigen/"
                    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${EIGEN_BUNDLED_INSTALLDIR}"
                )
            endif()
        endif()
    endif()

    if(SYSTEM_SPDLOG)
        message(STATUS "Using system-wide spdlog library...")
        find_package(spdlog REQUIRED)
    else()
        # setup bundled spdlog
        # this already reads spdlog's cmake definitions
        CPMAddPackage(
            NAME spdlog
            GITHUB_REPOSITORY gabime/spdlog
            VERSION 1.10.0
            OPTIONS "SPDLOG_INSTALL ON"
        )
        message(STATUS "Using bundled spdlog version in ${spdlog_SOURCE_DIR}")
        # because spdlog didn't do that on its own, we export the target.
        # this is just for the in-buildtree linking, won't be installed.
        # store in the bindir/elsa/ directory, which may be nested
        # in other build directories somehow.
        export(
            TARGETS spdlog_header_only spdlog
            FILE "${CMAKE_CURRENT_BINARY_DIR}/elsa/spdlogTargets.cmake"
        )
        # SPDLOG_INSTALL will manage install and includedir setup
        # to $CMAKE_INSTALL_INCLUDEDIR/spdlog
    endif()

    if (SKBUILD AND ELSA_BUILD_PYTHON_BINDINGS)
        # Scikit-Build does not add your site-packages to the search path
        # automatically, so we need to add it _or_ the pybind11 specific directory
        # here.
        execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE _tmp_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
        list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

        # Now we can find pybind11
        find_package(pybind11 CONFIG REQUIRED)

        add_subdirectory(pyelsa)
    endif()
else()
    message(STATUS "    No dependencies added for elsa, as it is not stand-alone")
endif()

# Enable CUDA support if requested
if(ELSA_CUDA_VECTOR OR ELSA_BUILD_CUDA_PROJECTORS)
    message(STATUS "CUDA support requested...")
    include(CheckLanguage)
    check_language(CUDA)
    find_package(CUDA 11.4)

    if(CMAKE_CUDA_COMPILER AND (CUDA_VERSION VERSION_GREATER 11))
        enable_language(CUDA)
        message(STATUS "CUDA support enabled")

        # set target CUDA architectures
        include(ComputeCapabilityCUDA)
        set_cuda_arch_type(${ELSA_CUDA_ARCH_TYPE})

        foreach(TARGET_GPU ${TARGET_GPUS})
            string(REPLACE "." "" TARGET_GPU ${TARGET_GPU})
            list(APPEND CMAKE_CUDA_ARCHITECTURES ${TARGET_GPU})
        endforeach()

        if (ELSA_CUDA_VECTOR)
            message(STATUS "Enabling DataHandlerGPU")
            message(STATUS "The default container type will be on the GPU")

            set(ELSA_BUILD_WITH_QUICKVEC TRUE)
            add_definitions(-DELSA_CUDA_VECTOR)
        endif()
    elseif(ELSA_CUDA_VECTOR)
        message(FATAL_ERROR "CUDA compiler not found!")
        message(STATUS "Compiling without a DataHandlerGPU")
        message(STATUS "The default container type will be CPU")

        set(ELSA_BUILD_WITH_QUICKVEC FALSE)
        set(ELSA_CUDA_VECTOR OFF)
    endif()
endif()


# set where to install the exports/targets
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/elsa)

# include the install_elsa_module function
include(InstallElsaModule)

# ------------ Setup Tools -----------
# ------------

# Includes sanitizers
include(Sanitizer)


# ------------ setup testing -----------
# ------------

# if elsa is used as a submodule, turn testing off
if(NOT ELSA_MASTER_PROJECT)
    set(ELSA_TESTING OFF)
endif(NOT ELSA_MASTER_PROJECT)

if(ELSA_TESTING)
    enable_testing()

    if(SYSTEM_DOCTEST)
        message(STATUS "Using system-wide doctest library...")
        find_package(doctest REQUIRED)
    else()
        CPMAddPackage(
            NAME doctest
            GITHUB_REPOSITORY onqtam/doctest
            GIT_TAG 2.4.7
        )

        # add the CMake modules for automatic test discovery
        set(CMAKE_MODULE_PATH "${doctest_SOURCE_DIR}/scripts/cmake" ${CMAKE_MODULE_PATH})
    endif()

    message(STATUS "elsa testing is enabled")

    # Run ctest to run all unit tests
    # All tests are includes ass dependencies to this target, such that they get build
    # schedule randomly that we don't start depending on some test order
    add_custom_target(tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
        COMMENT "Build and run all the tests.")

    # Target to only build all tests
    add_custom_target(build-tests)

    if(ELSA_COVERAGE)
        message(STATUS "elsa test coverage is enabled")

        include(CodeCoverage)
        add_code_coverage_all_targets(EXCLUDE
            elsa/test_routines/*
            */tests/*
            ${PROJECT_BINARY_DIR}/_deps/*
            ${CPM_SOURCE_CACHE}/*
            /usr/*)


    else(ELSA_COVERAGE)
        message(STATUS "elsa test coverage is disabled")
    endif(ELSA_COVERAGE)
else(ELSA_TESTING)
    message(STATUS "    elsa testing is disabled")
endif(ELSA_TESTING)

# ------------ add code/docs -----------
# ------------

# the elsa library
add_subdirectory(elsa)

# the documentation
add_subdirectory(docs EXCLUDE_FROM_ALL)

# the examples
if(ELSA_BUILD_EXAMPLES)
    add_subdirectory(examples EXCLUDE_FROM_ALL)
endif(ELSA_BUILD_EXAMPLES)

# ------------ setup installation ------
# ------------

# set up the target/library for make install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# setup the ElsaConfig*.cmake files
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/elsa/elsaConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/elsaConfig.cmake.in
    ${CMAKE_BINARY_DIR}/elsa/elsaConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
)

if(ELSA_MASTER_PROJECT)
    # this puts the local build tree into the user package repository, but not the installed version...
    export(PACKAGE elsa)
endif()

if(ELSA_INSTALL)
    message(STATUS "setting up elsa installation")

    # install the config files
    install(
        FILES
        ${CMAKE_BINARY_DIR}/elsa/elsaConfig.cmake
        ${CMAKE_BINARY_DIR}/elsa/elsaConfigVersion.cmake
        DESTINATION ${INSTALL_CONFIG_DIR}
    )
endif(ELSA_INSTALL)


##
# Add uninstall target
# Remove all headers, library and CMake files, which where copied to the install location

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P "${PROJECT_SOURCE_DIR}/cmake/Uninstall.cmake"
    COMMENT "Uninstalling elsa (Removing all installed files)"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt"
)


# build information overview
string(TOUPPER "CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}" BUILD_TYPE_CXX_FLAGS)
message("
          project | ${PROJECT_NAME}
         compiler | ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}
       build type | ${CMAKE_BUILD_TYPE}
         cxxflags | ${CMAKE_CXX_FLAGS}
 build type flags | ${${BUILD_TYPE_CXX_FLAGS}}
        build dir | ${CMAKE_BINARY_DIR}
   install prefix | ${CMAKE_INSTALL_PREFIX}
")

set_property(GLOBAL PROPERTY TARGET_MESSAGES OFF)
