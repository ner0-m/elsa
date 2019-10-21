cmake_minimum_required(VERSION 3.9)

project(elsa VERSION 0.5
        DESCRIPTION "elsa recon toolbox"
        LANGUAGES CXX)


# ------------ elsa options ------------
# ------------
# detect if elsa is being bundled, disable some options based on this
if (NOT DEFINED ELSA_MASTER_PROJECT)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(ELSA_MASTER_PROJECT ON)
    else()
        set(ELSA_MASTER_PROJECT OFF)
    endif()
endif ()

option(ELSA_TESTING "Enable the unit tests of elsa" ${ELSA_MASTER_PROJECT})
option(ELSA_INSTALL "Generate the install target" ${ELSA_MASTER_PROJECT})
option(GIT_SUBMODULE "Check submodules during build" ${ELSA_MASTER_PROJECT})
option(ELSA_BUILD_EXAMPLES "Build examples" ${ELSA_MASTER_PROJECT})
option(ELSA_COVERAGE "Enables test coverage reporting" OFF)

option(ELSA_BUILD_CUDA_PROJECTORS "Attempts to build CUDA projectors if set" ON)
option(ELSA_BUILD_WITH_MORE_WARNINGS "Reduce noise of warnings, but make chaning fast and easy" OFF)
option(ELSA_BUILD_WITH_SANITIZERS "Build with Sanitizers. Pass ELSA_SANITIZE_THREAD or ELSA_SANITIZE_ADDRESS" ON)

# ------------ general setup -----------
# ------------

# add our cmake modules under cmake/
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# set default built type to "Release" (if no other specified)
set(DEFAULT_BUILD_TYPE "Release")
include(SetDefaultBuildType)

# set where to install the exports/targets
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake)

# ------------ dependencies ------------
# ------------

# only populate the dependencies, iff it's a standalone project!
if(ELSA_MASTER_PROJECT)
    # include the git submodule update
    include(UpdateGitSubmodules)

    # find Eigen Library
    # if you're not using the submodules use set(EIGEN3_INCLUDE_DIR path/to/Eigen3) to point to the Eigen3 directory
    find_package(Eigen3 3.3 REQUIRED)
    message(STATUS "Using eigen3 submodule at ${EIGEN3_INCLUDE_DIR}")

    # find spdlog Library
    # if you're not using the submodules use set(Spdlog_INCLUDE_DIR path/to/spdlog) to point to the spdlog directory
    find_package(Spdlog REQUIRED)
    message(STATUS "Using spdlog submodule at ${Spdlog_INCLUDE_DIR}")
else()
    message(STATUS "    No dependencies added for elsa, as it's a subproject")
endif()

# include the In stallElsaModule function
include(InstallElsaModule)

# ------------ setup testing -----------
# ------------

# reset ELSA_TESTING to only be active, if it's a standlone project
if(NOT ELSA_MASTER_PROJECT)
    set(ELSA_TESTING OFF)
endif(NOT ELSA_MASTER_PROJECT)

if(ELSA_TESTING)
    message(STATUS "elsa testing is enabled")

    enable_testing()
    add_subdirectory(thirdparty/Catch2)

    # add the CMake modules for automatic test discovery
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/thirdparty/Catch2/contrib" ${CMAKE_MODULE_PATH})
else(ELSA_TESTING)
    message(STATUS "    elsa testing is disabled")
endif(ELSA_TESTING)


# ------------ setup coverage test -----------
# ------------

if(ELSA_COVERAGE)
    message(STATUS "elsa coverage is enabled")

    include(CodeCoverage)
    APPEND_COVERAGE_COMPILER_FLAGS()
    set(COVERAGE_LCOV_EXCLUDES '${PROJECT_SOURCE_DIR}/thirdparty/*' '/usr/*')
    SETUP_TARGET_FOR_COVERAGE_LCOV(NAME test_coverage EXECUTABLE ctest)
endif(ELSA_COVERAGE)



# ------------ add code/docs -----------
# ------------

# the elsa library
add_subdirectory(elsa)

# the documentation
add_subdirectory(docs EXCLUDE_FROM_ALL)

if(ELSA_BUILD_EXAMPLES)
    # the examples
    add_subdirectory(examples EXCLUDE_FROM_ALL)
endif(ELSA_BUILD_EXAMPLES)

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
