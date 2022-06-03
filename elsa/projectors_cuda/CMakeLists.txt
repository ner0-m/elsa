if(CMAKE_CUDA_COMPILER AND (CUDA_VERSION VERSION_GREATER 11))
    message(STATUS "Enabling CUDA projectors...")

    # necessary to be able to include <cuda_runtime.h>
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    # list all the headers of the module
    set(MODULE_HEADERS
        projector_kernels/TraverseSiddonsCUDA.cuh
        projector_kernels/TraverseJosephsCUDA.cuh
        CUDAProjector.h
        CUDAWrappers.h
        SiddonsMethodCUDA.h
        JosephsMethodCUDA.h
        SplittingProjector.h
    )

    # list all the code files of the module
    set(MODULE_SOURCES
        projector_kernels/TraverseSiddonsCUDA.cu
        projector_kernels/TraverseJosephsCUDA.cu
        SiddonsMethodCUDA.cpp
        JosephsMethodCUDA.cpp
        SplittingProjector.cpp
    )

    list(
        APPEND
        MODULE_PUBLIC_DEPS
        elsa_core
        elsa_logging
        elsa_projectors
        ${CUDART_LIBRARY}
    )
    list(APPEND MODULE_PRIVATE_DEPS)

    ADD_ELSA_MODULE(
        projectors_cuda "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
        PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
    )

    # add the projector_kernels folder to the include directories
    target_include_directories(
        ${ELSA_MODULE_TARGET_NAME}
        PUBLIC $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}/projector_kernels>
               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/projector_kernels>
               ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    # set CUDA compiler flags to the optimal for projectors
    # fmad can reduce accuracy (some tests would actually fail if used), but only gives a negligible boost to speed
    # use expt-relaxed-constexpr flag to suppress warnings caused by including the Eigen header
    # enable warnings that would indicate possible performance issues
    target_compile_options(
        ${ELSA_MODULE_TARGET_NAME}
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false --expt-relaxed-constexpr --default-stream=per-thread -Wreorder -Xptxas=-warn-spills,-warn-lmem-usage>
    )

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(
            ${ELSA_MODULE_TARGET_NAME}
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --device-debug>
        )
    endif()

    # set define to check cuda projectors in projects
    target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_CUDA_PROJECTORS)

    if(ELSA_BUILD_PYTHON_BINDINGS)
        set(BINDINGS_SOURCES 
            SiddonsMethodCUDA.cpp
            JosephsMethodCUDA.cpp
        )

        GENERATE_BINDINGS(
            ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
            ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${BINDINGS_SOURCES}
        )
    endif()

    write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_projectors)


else()
    message(STATUS "CUDA compiler NOT found! CUDA projector support disabled")
endif()
