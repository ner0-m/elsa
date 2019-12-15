cmake_minimum_required(VERSION 3.10)

include(CheckLanguage)

check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA support enabled")

    # build the kernels first
    add_subdirectory(projector_kernels)

    # necessary to be able to include <cuda_runtime.h>
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    # list all the headers of the module
    set(MODULE_HEADERS
            SiddonsMethodCUDA.h
            JosephsMethodCUDA.h)

    # list all the code files of the module
    set(MODULE_SOURCES
            SiddonsMethodCUDA.cpp
            JosephsMethodCUDA.cpp)

    # setup library see elsa/CMakeLists.txt (it's a macro, so we can set variables here)
    ELSA_LIBRARY_SETUP(projectors_cuda elsa_core elsa_logging elsa_projectors elsa_projector_kernels)

    target_include_directories(${ELSA_MODULE_TARGET_NAME}
            PUBLIC 
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # set define to check cuda projectors in projects
    target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_CUDA_PROJECTORS)
else()
    message(STATUS "CUDA compiler NOT found! CUDA support disabled")
endif()