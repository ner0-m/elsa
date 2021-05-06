# Set the name of the module
set(ELSA_MODULE_NAME ml)
set(ELSA_MODULE_TARGET_NAME elsa_${ELSA_MODULE_NAME})
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)

# Path to ml backend implementations
set(ELSA_BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backend")

# Check for available backends

# Dnnl backend
if(Dnnl_FOUND AND ELSA_BUILD_ML_DNNL)
  message(STATUS "Dnnl backend for ml module is available")
  set(ELSA_HAS_DNNL_BACKEND 1)
  # Define ELSA_HAS_DNNL_BACKEND 
  add_definitions(-DELSA_HAS_DNNL_BACKEND=1)
  set(ELSA_ML_DNNL_BACKEND_DIR ${ELSA_BACKEND_DIR}/Dnnl)
  add_subdirectory(${ELSA_ML_DNNL_BACKEND_DIR})
endif(Dnnl_FOUND AND ELSA_BUILD_ML_DNNL)

# Cudnn backend
if(Cudnn_FOUND AND Cublas_FOUND AND ELSA_BUILD_ML_CUDNN)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Cudnn backend for ml module is available")
    set(ELSA_HAS_CUDNN_BACKEND 1)
    # Define ELSA_HAS_CUDNN_BACKEND 
    add_definitions(-DELSA_HAS_CUDNN_BACKEND=1)
    set(ELSA_ML_CUDNN_BACKEND_DIR ${ELSA_BACKEND_DIR}/Cudnn)
    add_subdirectory(${ELSA_ML_CUDNN_BACKEND_DIR})
  endif(CMAKE_CUDA_COMPILER)
elseif(Cudnn_FOUND AND ELSA_BUILD_ML_CUDNN)
    message(STATUS "Cudnn backend deactivated, cuDNN not found")
elseif(Cublas_FOUND AND ELSA_BUILD_ML_CUDNN) 
    message(STATUS "Cudnn backend deactivated, cublas not found")
endif()

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
)

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

# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC elsa_core elsa_logging)

# Link Dnnl if wanted and available
if (ELSA_HAS_DNNL_BACKEND)
  target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC ${Dnnl_LIB})
  # Include Dnnl library header as system to avoid warnings
  target_include_directories(${ELSA_MODULE_TARGET_NAME} SYSTEM PUBLIC $<BUILD_INTERFACE:${Dnnl_INCLUDE_DIR}>)
  target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ELSA_DNNL_BACKEND_DIR}>)
endif (ELSA_HAS_DNNL_BACKEND)


# use OpenMP is available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${ELSA_MODULE_TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(${ELSA_MODULE_TARGET_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${ELSA_ML_DNNL_BACKEND_DIR}>
)

# Link Dnnl backend
if(ELSA_HAS_DNNL_BACKEND)
  target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ELSA_ML_DNNL_BACKEND_DIR}>)
  target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC elsa_ml_dnnl_backend)
endif(ELSA_HAS_DNNL_BACKEND)

# Link Cudnn backend
if(ELSA_HAS_CUDNN_BACKEND)
  target_link_libraries(${ELSA_MODULE_TARGET_NAME} PUBLIC elsa_ml_cudnn_backend)
  target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ELSA_ML_CUDNN_BACKEND_DIR}>)
endif(ELSA_HAS_CUDNN_BACKEND)

# require C++17
target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_17)
# set -fPIC
set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# build the tests (if enabled)
if(ELSA_TESTING)
    add_subdirectory(tests)

    # set the module name correctly here! 
    set(_module_name ${ELSA_MODULE_TARGET_NAME}) 
    set(ELSA_MODULE_TARGET_NAME elsa_ml_cudnn_backend) 
    # We add Cudnn backend tests here because we want to be able to use C++17
    # in the tests
    if(ELSA_HAS_CUDNN_BACKEND)
      add_subdirectory(${ELSA_ML_CUDNN_BACKEND_DIR}/tests)
    endif(ELSA_HAS_CUDNN_BACKEND)
     
    # and now set it back 
    set(ELSA_MODULE_TARGET_NAME ${_module_name}) 
endif(ELSA_TESTING)

if (ELSA_BUILD_WITH_QUICKVEC)
  add_definitions(-DELSA_CUDA_VECTOR)
endif(ELSA_BUILD_WITH_QUICKVEC)

# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})
