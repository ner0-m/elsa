cmake_minimum_required(VERSION 3.9)

# setup macro for component registration
# so you can use find_package(elsa COMPONENT name)
macro(registerComponent name)
    set(ELSA_REGISTERED_COMPONENTS "${ELSA_REGISTERED_COMPONENTS};${name}" PARENT_SCOPE)
endmacro()

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


#---------------------------------------------------------------------------------------
# Turn on compiler warnings and sanitizers for all components on, if wanted
#---------------------------------------------------------------------------------------
if(${ELSA_MASTER_PROJECT})
    foreach (_component ${ELSA_REGISTERED_COMPONENTS})
        # add warnings if enabled
        if(${ELSA_BUILD_WITH_MORE_WARNINGS})
            if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
                target_compile_options( elsa_${_component} PUBLIC -Wall -Wextra -Wconversion -pedantic -Wfatal-errors)
            endif()
            if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                target_compile_options( elsa_${_component} PUBLIC /W3 /WX )
            endif()
        endif(${ELSA_BUILD_WITH_MORE_WARNINGS})
    endforeach (_component ELSA_REGISTERED_COMPONENTS)

    # add sanitizers if enabled
    if(${ELSA_BUILD_WITH_SANITIZERS})
        include(${PROJECT_SOURCE_DIR}/cmake/Sanitizers.cmake)
    endif(${ELSA_BUILD_WITH_SANITIZERS})
endif(${ELSA_MASTER_PROJECT})

# propogate the variable to the parent scope
set(ELSA_REGISTERED_COMPONENTS "${ELSA_REGISTERED_COMPONENTS};" PARENT_SCOPE)
