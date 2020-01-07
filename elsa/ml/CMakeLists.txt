cmake_minimum_required(VERSION 3.9)

# set the name of the module
set(ELSA_MODULE_NAME ml)
set(ELSA_MODULE_TARGET_NAME elsa_ml)
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)

set(ELSA_BACKEND_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backend)
set(ELSA_DNNL_BACKEND_DIR ${ELSA_BACKEND_DIR}/DNNL)

if(Dnnl_FOUND)
  set(ELSA_HAS_DNNL_BACKEND 1)
endif(Dnnl_FOUND)

if(ELSA_HAS_DNNL_BACKEND)
  set(ELSA_DNNL_BACKEND_HEADERS
      ${ELSA_DNNL_BACKEND_DIR}/DnnlLayer.h
      ${ELSA_DNNL_BACKEND_DIR}/DnnlTrainableLayer.h
      ${ELSA_DNNL_BACKEND_DIR}/DnnlConvLayer.h
      ${ELSA_DNNL_BACKEND_DIR}/DnnlDenseLayer.h
      ${ELSA_DNNL_BACKEND_DIR}/DnnlPoolingLayer.h
      ${ELSA_DNNL_BACKEND_DIR}/DnnlActivationLayer.h
      ${ELSA_DNNL_BACKEND_DIR}/DnnlSoftmaxLayer.h
      ${ELSA_DNNL_BACKEND_DIR}/DnnlSequentialNetwork.h)

  set(ELSA_DNNL_BACKEND_SOURCES
      ${ELSA_DNNL_BACKEND_DIR}/DnnlLayer.cpp
      ${ELSA_DNNL_BACKEND_DIR}/DnnlTrainableLayer.cpp
      ${ELSA_DNNL_BACKEND_DIR}/DnnlConvLayer.cpp
      ${ELSA_DNNL_BACKEND_DIR}/DnnlDenseLayer.cpp
      ${ELSA_DNNL_BACKEND_DIR}/DnnlPoolingLayer.cpp
      ${ELSA_DNNL_BACKEND_DIR}/DnnlActivationLayer.cpp
      ${ELSA_DNNL_BACKEND_DIR}/DnnlSoftmaxLayer.cpp
      ${ELSA_DNNL_BACKEND_DIR}/DnnlSequentialNetwork.cpp)

endif()

# list all the headers of the module
set(MODULE_HEADERS
    Layer.h
    TrainableLayer.h
    ActivationLayer.h
    ConvLayer.h
    DenseLayer.h
    PoolingLayer.h
    RandomInitializer.h
    SoftmaxLayer.h
    SequentialNetwork.h
    ${ELSA_DNNL_BACKEND_HEADERS})

# list all the code files of the module
set(MODULE_SOURCES
    ActivationLayer.cpp
    ConvLayer.cpp
    DenseLayer.cpp
    PoolingLayer.cpp
    RandomInitializer.cpp
    SoftmaxLayer.cpp
    SequentialNetwork.cpp
    ${ELSA_DNNL_BACKEND_SOURCES})

# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
# add_library(elsa::${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ALIAS)

target_link_libraries(${ELSA_MODULE_TARGET_NAME}
                      PUBLIC elsa_core elsa_logging ${Dnnl_LIB})

target_include_directories(
  ${ELSA_MODULE_TARGET_NAME}
  PUBLIC $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${ELSA_DNNL_BACKEND_DIR}>)

# Include Dnnl library header as system to avoid warnings
target_include_directories(${ELSA_MODULE_TARGET_NAME} SYSTEM
                           PUBLIC $<BUILD_INTERFACE:${Dnnl_INCLUDE_DIR}>)

# require C++17
target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17)
# set -fPIC
set_target_properties(${ELSA_MODULE_TARGET_NAME}
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

# build the tests (if enabled)
if(ELSA_TESTING)
  add_subdirectory(tests)
endif(ELSA_TESTING)

# register the module
registercomponent(${ELSA_MODULE_NAME})

# install the module
installelsamodule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME}
                  ${ELSA_MODULE_EXPORT_TARGET})
