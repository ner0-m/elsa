# TODO write this CMakeLists file correctly if a green light is given

# list all the headers of the module
set(MODULE_HEADERS
        AutoEncoder.h
        UNet.h)

# list all the code files of the module
set(MODULE_SOURCES)

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
        ml_models "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
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
