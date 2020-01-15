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
option(ELSA_BUILD_WITH_MORE_WARNINGS "Enable all and extra warnings when building (-Wall -Wextra)" ON)

option(ELSA_SANITIZE_THREAD "Build elsa with thread sanitizers (TSAN)" OFF)
option(ELSA_SANITIZE_ADDRESS "Build elsa with address and undefined-behavior sanitizers (ASAN and UBSAN)" OFF)


# ------------ general setup -----------
# ------------

# add our cmake modules under cmake/
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# set default built type to "Release" (if no other specified)
set(DEFAULT_BUILD_TYPE "Release")
include(SetDefaultBuildType)


# ------------ dependencies ------------
# ------------

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
else()
    message(STATUS "    No dependencies added for elsa, as it is not stand-alone")
endif()


# set where to install the exports/targets
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/elsa)


# include the InstallElsaModule function
include(InstallElsaModule)


# ------------ setup testing -----------
# ------------

# if elsa is used as a submodule, turn testing off
if(NOT ELSA_MASTER_PROJECT)
    set(ELSA_TESTING OFF)
endif(NOT ELSA_MASTER_PROJECT)

if(ELSA_TESTING)
    message(STATUS "elsa testing is enabled")

    enable_testing()
    add_subdirectory(thirdparty/Catch2)

    add_custom_target(tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Build and run all the tests.")

    # add the CMake modules for automatic test discovery
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/thirdparty/Catch2/contrib" ${CMAKE_MODULE_PATH})

    if(ELSA_COVERAGE)
        message(STATUS "elsa test coverage is enabled")

        include(CodeCoverage)
        APPEND_COVERAGE_COMPILER_FLAGS()
        set(COVERAGE_LCOV_EXCLUDES '${PROJECT_SOURCE_DIR}/thirdparty/*' '/usr/*')
        SETUP_TARGET_FOR_COVERAGE_LCOV(NAME test_coverage EXECUTABLE ctest)

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
