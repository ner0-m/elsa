cmake_minimum_required(VERSION 3.10)

# set the name of the module
set(ELSA_MODULE_NAME ml_dnnl_backend)
set(ELSA_MODULE_TARGET_NAME elsa_${ELSA_MODULE_NAME})
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)

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

# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC elsa_core elsa_logging ${Dnnl_LIB})

target_include_directories(${ELSA_MODULE_TARGET_NAME}
                           PUBLIC
                           $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/../.."
)

# Include Dnnl headers as system headers to avoid warnings
target_include_directories(${ELSA_MODULE_TARGET_NAME} SYSTEM PUBLIC $<BUILD_INTERFACE:${Dnnl_INCLUDE_DIR}>)

# require C++17
target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17)
# set -fPIC
set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# build the tests (if enabled)
if(ELSA_TESTING)
    add_subdirectory(tests)
endif(ELSA_TESTING)

# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})