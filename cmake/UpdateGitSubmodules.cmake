# update/init git submodules if necessary
# idea from https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html

# try to perform the git submodule update
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    if(GIT_SUBMODULE) # do this only if the GIT_SUBMODULE option is enabled
        message(STATUS "Git submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMODULE_RESULT)
        if(NOT GIT_SUBMODULE_RESULT EQUAL "0")
            message(WARNING "git submodule update --init failed with ${GIT_SUBMODULE_RESULT}")
        endif()
    endif()
endif()

# check the results
if (NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/Catch2/CMakeLists.txt" OR
    NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/eigen3/CMakeLists.txt" OR
    NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/spdlog/CMakeLists.txt")
    message(FATAL_ERROR "The git submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()