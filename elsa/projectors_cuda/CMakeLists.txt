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

    # set the name of the module
    set(ELSA_MODULE_NAME projectors_cuda)
    set(ELSA_MODULE_TARGET_NAME elsa_${ELSA_MODULE_NAME})
    set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)

    # list all the headers of the module
    set(MODULE_HEADERS
            SiddonsMethodCUDA.h
            JosephsMethodCUDA.h)

    # list all the code files of the module
    set(MODULE_SOURCES
            SiddonsMethodCUDA.cpp
            JosephsMethodCUDA.cpp)

    # build the module library
    add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
    add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

    target_link_libraries(${ELSA_MODULE_TARGET_NAME} elsa_core elsa_logging elsa_projectors elsa_projector_kernels)

    target_include_directories(${ELSA_MODULE_TARGET_NAME}
            PUBLIC
            $<INSTALL_INTERFACE:include/${ELSA_MODULE_NAME}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            )
    target_include_directories(${ELSA_MODULE_TARGET_NAME}
            PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # require C++17
    target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17) 


    # build the tests (if enabled)
    if(ELSA_TESTING)
        add_subdirectory(tests)
    endif(ELSA_TESTING)


    # register the module
    registerComponent(${ELSA_MODULE_NAME})

    # install the module
    InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})
else()
    message(STATUS "CUDA compiler NOT found! CUDA support disabled")
endif()