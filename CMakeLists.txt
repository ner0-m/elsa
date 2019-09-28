cmake_minimum_required(VERSION 3.9)

# detect if elsa is being bundled, disable install and tests
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
else() 
  set(SUBPROJECT ON)
  message(STATUS "Including elsa as a subproject")
endif()


project(elsa VERSION 0.5
        DESCRIPTION "elsa recon toolbox"
        LANGUAGES CXX)


# ------------ elsa options ------------
# ------------

option(ELSA_TESTING "Enable the unit tests of elsa" ON)
option(GIT_SUBMODULE "Check submodules during build" ON)
option(BUILD_ELSA_PROJECTORS_CUDA "Attempts to build CUDA projectors if set" ON)


# ------------ general setup -----------
# ------------

# add our cmake modules under cmake/
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# set default built type to "Debug" (if no other specified)
set(DEFAULT_BUILD_TYPE "Release")
include(SetDefaultBuildType)

# set where to install the exports/targets
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake)



# ------------ dependencies ------------
# ------------

# only populate the dependencies, iff it's a standalone project!
if(NOT_SUBPROJECT) 
    # include the git submodule update
    include(UpdateGitSubmodules)

    # setup the Eigen3 library
    find_package(Eigen3 3.3 NO_MODULE QUIET) # first try if there is an installed version
    if(NOT EIGEN3_INCLUDE_DIR) # if not found, try our submodule
        set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/eigen3")
        message(STATUS "Using eigen3 submodule at ${EIGEN3_INCLUDE_DIR}")
        find_package(Eigen3 3.3 REQUIRED)
    endif()
else()
    message(STATUS "    No dependencies added for elsa, as it's a subproject")
endif()


# include the In stallElsaModule function
include(InstallElsaModule)

# ------------ setup testing -----------
# ------------

# reset ELSA_TESTING to only be active, if it's a standlone project
if(SUBPROJECT)
    set(ELSA_TESTING OFF)
endif()

if(ELSA_TESTING)
    message(STATUS "elsa testing is enabled")

    enable_testing()
    add_subdirectory(thirdparty/Catch2)
    # add the CMake modules for automatic test discovery
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/thirdparty/Catch2/contrib" ${CMAKE_MODULE_PATH})
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
add_subdirectory(examples EXCLUDE_FROM_ALL)


# ------------ setup installation ------
# ------------

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
==== BASE ====
