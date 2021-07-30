include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "enabling CUDA projectors...")

    # build the kernels first
    add_subdirectory(projector_kernels)

    # necessary to be able to include <cuda_runtime.h>
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    # list all the headers of the module
    set(MODULE_HEADERS SiddonsMethodCUDA.h JosephsMethodCUDA.h)

    # list all the code files of the module
    set(MODULE_SOURCES SiddonsMethodCUDA.cpp JosephsMethodCUDA.cpp)

    list(
        APPEND
        MODULE_PUBLIC_DEPS
        elsa_core
        elsa_logging
        elsa_projectors
        elsa_projector_kernels
        ${CUDART_LIBRARY}
    )
    list(APPEND MODULE_PRIVATE_DEPS)

    ADD_ELSA_MODULE(
        projectors_cuda "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
        PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
    )

    # link Quickvec for DataHandlerGPU
    # Also add the necessary compile definition for quickvec
    if(${ELSA_BUILD_WITH_QUICKVEC})
        set_source_files_properties(${MODULE_SOURCES} PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_ENABLE_CUDA_VECTOR)
        target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC "Quickvec::quickvec")
        target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cuda_std_17)
    endif()

    target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # set define to check cuda projectors in projects
    target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_CUDA_PROJECTORS)

    if(ELSA_BUILD_PYTHON_BINDINGS)
        GENERATE_BINDINGS(
            ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
            ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
        )
    endif()
else()
    message(STATUS "CUDA compiler NOT found! CUDA projector support disabled")
endif()
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging elsa_projectors elsa_projector_kernels)
