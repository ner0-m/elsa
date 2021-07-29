# Path to ml backend implementations
set(ELSA_BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backend")

# Check for available backends

# Dnnl backend
if (Dnnl_FOUND AND ELSA_BUILD_ML_DNNL)
    message(STATUS "Dnnl backend for ml module is available")

    set(ELSA_HAS_DNNL_BACKEND 1)
    set(ELSA_ML_DNNL_BACKEND_DIR ${ELSA_BACKEND_DIR}/Dnnl)
    add_subdirectory(${ELSA_ML_DNNL_BACKEND_DIR})
endif ()

# Cudnn backend
if (Cudnn_FOUND AND Cublas_FOUND AND ELSA_BUILD_ML_CUDNN)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        message(STATUS "Cudnn backend for ml module is available")

        enable_language(CUDA)

        set(ELSA_HAS_CUDNN_BACKEND 1)
        set(ELSA_ML_CUDNN_BACKEND_DIR ${ELSA_BACKEND_DIR}/Cudnn)
        add_subdirectory(${ELSA_ML_CUDNN_BACKEND_DIR})
    endif ()
elseif (Cudnn_FOUND AND ELSA_BUILD_ML_CUDNN)
    message(STATUS "Cudnn backend deactivated, cuDNN not found")
elseif (Cublas_FOUND AND ELSA_BUILD_ML_CUDNN)
    message(STATUS "Cudnn backend deactivated, cublas not found")
endif ()

# list all the headers of the module
set(MODULE_HEADERS
        Common.h
        Initializer.h
        Activation.h
        Conv.h
        Dense.h
        Input.h
        Layer.h
        Merging.h
        Pooling.h
        Reshape.h
        Softmax.h
        Trainable.h
        Loss.h
        Model.h
        Optimizer.cpp
        Optimizer.h
        ProgressBar.h
        State.h
        Utils.h
        Projector.h
        Graph.h
        BackendAdaptor.h
        CudnnDataContainerInterface.h
        models/AutoEncoder.h
        models/UNet.h)

# list all the code files of the module
set(MODULE_SOURCES
        Common.cpp
        Initializer.cpp
        Activation.cpp
        Conv.cpp
        Dense.cpp
        Input.cpp
        Layer.cpp
        Merging.cpp
        Pooling.cpp
        Reshape.cpp
        Softmax.cpp
        Trainable.cpp
        Loss.cpp
        Model.cpp
        Model.cpp
        State.cpp
        )

# elsa modules ml depends on  
list(APPEND ELSA_MODULE_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PUBLIC_DEPS)
list(APPEND MODULE_PRIVATE_DEPS)

# Add OpenMP to the private dependency list
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    list(APPEND MODULE_PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif ()

# Add Dnnl backend to the list of public dependencies
if (ELSA_HAS_DNNL_BACKEND)
    list(APPEND ELSA_MODULE_DEPS elsa_ml_dnnl_backend)
endif ()

# Add Cudnn backend to the list of public dependencies
if (ELSA_HAS_CUDNN_BACKEND)
    list(APPEND ELSA_MODULE_DEPS elsa_ml_cudnn_backend)
endif ()

list(APPEND MODULE_PUBLIC_DEPS ${ELSA_MODULE_DEPS})

ADD_ELSA_MODULE(
        ml "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
        PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

# Link Dnnl if wanted and available
if (ELSA_HAS_DNNL_BACKEND)
    # Include Dnnl library header as system to avoid warnings
    target_include_directories(${ELSA_MODULE_TARGET_NAME} SYSTEM PUBLIC $<BUILD_INTERFACE:${Dnnl_INCLUDE_DIR}>)
    target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ELSA_ML_DNNL_BACKEND_DIR}>)
    target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_HAS_DNNL_BACKEND=1)
endif ()

# Link Cudnn backend
if (ELSA_HAS_CUDNN_BACKEND)
    target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ELSA_ML_DNNL_BACKEND_DIR}>)
    target_compile_definitions(${ELSA_MODULE_TARGET_NAME} PUBLIC ELSA_HAS_CUDNN_BACKEND=1)
endif ()

# We add Cudnn backend tests here because we want to be able to use C++17 in the tests
if (ELSA_TESTING AND ELSA_HAS_CUDNN_BACKEND)
    # set the module name correctly here!
    set(_module_name ${ELSA_MODULE_TARGET_NAME})
    set(ELSA_MODULE_TARGET_NAME elsa_ml_cudnn_backend)

    add_subdirectory(${ELSA_ML_CUDNN_BACKEND_DIR}/tests)

    # and now set it back
    set(ELSA_MODULE_TARGET_NAME ${_module_name})
endif ()

# TODO required for generating the bindings, also required is to change tools/bindings_generation/hints/ml_hints.hpp to
#  tools/bindings_generation/hints/ml_hints.cpp and add it in its CMakeLists.txt file. What else is required for
#  functional ml hints?

#if(ELSA_BUILD_PYTHON_BINDINGS)
#    GENERATE_BINDINGS(
#            ${ELSA_MODULE_TARGET_NAME} bind_${ELSA_MODULE_NAME}.cpp
#            ${PROJECT_SOURCE_DIR}/tools/bindings_generation/hints/${ELSA_MODULE_NAME}_hints.cpp ${MODULE_SOURCES}
#    )
#endif()

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES ${ELSA_MODULE_DEPS})
