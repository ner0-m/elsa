# list all the headers of the module
set(MODULE_HEADERS
    DnnlActivationLayer.h
    DnnlBackendAdaptor.h
    DnnlConvolution.h
    DnnlDenseLayer.h
    DnnlFlattenLayer.h
    DnnlLayer.h
    DnnlMerging.h
    DnnlNoopLayer.h
    DnnlPoolingLayer.h
    DnnlSoftmaxLayer.h
    DnnlTrainableLayer.h
    DnnlOptimizer.h
)

# list all the code files of the module
set(MODULE_SOURCES
    DnnlActivationLayer.cpp
    DnnlConvolution.cpp
    DnnlDenseLayer.cpp
    DnnlFlattenLayer.cpp
    DnnlLayer.cpp
    DnnlMerging.cpp
    DnnlNoopLayer.cpp
    DnnlPoolingLayer.cpp
    DnnlSoftmaxLayer.cpp
    DnnlTrainableLayer.cpp
    DnnlOptimizer.cpp
)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging ${Dnnl_LIB})
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    ml_dnnl_backend "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

target_include_directories(
    ${ELSA_MODULE_TARGET_NAME}
    PUBLIC $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/../.."
)

# Include Dnnl headers as system headers to avoid warnings
target_include_directories(${ELSA_MODULE_TARGET_NAME} SYSTEM PUBLIC $<BUILD_INTERFACE:${Dnnl_INCLUDE_DIR}>)
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging)
