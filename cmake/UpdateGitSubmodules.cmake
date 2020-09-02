# update/init git submodules if necessary
# idea from https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
 
find_package(Git QUIET)
 
# Return early if git is not found, or submodule flag is not set, or if this is not a git project
if(NOT GIT_FOUND OR NOT GIT_SUBMODULE OR NOT EXISTS "${PROJECT_SOURCE_DIR}/.git")
    return()
endif()
 
# Loop over all submodules 
foreach(module ${ELSA_SUBMODULES})
    set(SUBMODULE_GIT_PATH "${PROJECT_SOURCE_DIR}/.git/modules/thirdparty/${module}") 
    set(SUBMODULE_PATH "${PROJECT_SOURCE_DIR}/thirdparty/${module}")
    set(GIT_SUBMODULE_RESULT "0") 
     
    if(NOT EXISTS ${SUBMODULE_GIT_PATH})
        message(STATUS "Init submodule ${module}") 
         
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --force thirdparty/${module}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMODULE_RESULT
                ERROR_QUIET) 
             
        # If init for some reason didn't work, we deinit and init again 
        # This seems to resolve some problems 
        # This usually happens in the pipeline, if we copy over the thirdparty stuff, but not the .git/modules
        # folder
        if(NOT GIT_SUBMODULE_RESULT EQUAL "0")
            message(STATUS "Init failed, trying to deinit it and init again") 
             
            # reset error
            set(GIT_SUBMODULE_RESULT "0") 
             
            # Deinit it 
            execute_process(COMMAND ${GIT_EXECUTABLE} submodule deinit --force thirdparty/${module} 
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_QUIET ERROR_QUIET) 
            # And reinit and update it 
            execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init thirdparty/${module} 
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMODULE_RESULT)
             
            if(NOT GIT_SUBMODULE_RESULT EQUAL "0")
                message(WARNING "Init submodule ${module} failed with ${GIT_SUBMODULE_RESULT}")
            else()
                message(STATUS "Reinit worked")
            endif() 
        endif()
    endif()
     
    if(NOT EXISTS "${SUBMODULE_PATH}/CMakeLists.txt")
        message(STATUS "Update submodule ${module}") 
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --force thirdparty/${module}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMODULE_RESULT)
    endif()
         
    # Check return value for error 
    if(NOT GIT_SUBMODULE_RESULT EQUAL "0")
        message(WARNING "Updating submodule ${module} failed with ${GIT_SUBMODULE_RESULT}")
    endif()
     
    # If we still have no CMakeLists file something is wrong 
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/${module}/CMakeLists.txt")
        message(FATAL_ERROR "The git submodule ${module} is not present! Either GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
    endif()
endforeach()

