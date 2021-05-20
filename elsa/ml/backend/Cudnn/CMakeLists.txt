include(ComputeCapabilityCUDA)

# make sure host compiler used by NVCC is the one used for the rest of the project
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
 
# Append compute capabilities to CMAKE_CUDA_ARCHITECTURE 
set_cuda_arch_type(${ELSA_CUDA_ARCH_TYPE})

# fmad can reduce accuracy, but only gives a negligible boost to speed use expt-relaxed-constexpr flag to suppress
# warnings caused by including the Eigen header
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --expt-relaxed-constexpr")

if(ELSA_MASTER_PROJECT)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wreorder -Xptxas=-warn-spills,-warn-lmem-usage")
endif()

set(ELSA_ML_INCLUDE_PATH)
# list all the headers of the module
set(MODULE_HEADERS
    CudnnActivation.h
    CudnnLayer.h
    CudnnDense.h
    CudnnTrainable.h
    CudnnMemory.h
    CudnnContext.h
    CudnnSoftmax.h
    CudnnReshape.h
    CudnnNoop.h
    CudnnMerging.h
    CudnnOptimizer.h
    CudnnPooling.h
    CudnnConvolution.h
)

# list all the code files of the module
set(MODULE_SOURCES
    CudnnMemory.cu
    CudnnActivation.cu
    CudnnContext.cu
    CudnnTrainable.cpp
    CudnnDense.cu
    CudnnSoftmax.cpp
    CudnnReshape.cpp
    CudnnNoop.cpp
    CudnnMerging.cu
    CudnnOptimizer.cu
    CudnnPooling.cu
    CudnnConvolution.cu
    ../../Initializer.cpp
    ../../Optimizer.cpp
)

list(
    APPEND
    MODULE_PUBLIC_DEPS
    ${Cuda_LIB}
    ${Cudnn_LIB}
    ${Cublas_LIB}
    ${Nppc_LIB}
    ${Nppig_LIB}
)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    ml_cudnn_backend "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR NO_TESTS PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

# CUDA doesn't support C++17, require C++14
set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES CUDA_STANDARD 14 POSITION_INDEPENDENT_CODE ON)

# CUDA doesn't support linking against a library built under the C++17 standard, we only need the elsa.h header ->
# manually specify include directories
target_include_directories(
    ${ELSA_MODULE_TARGET_NAME}
    PUBLIC $<TARGET_PROPERTY:elsa_core,INTERFACE_INCLUDE_DIRECTORIES>
           $<TARGET_PROPERTY:elsa_logging,INTERFACE_INCLUDE_DIRECTORIES>
           $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/../.."
)

# Include Cudnn headers as system headers to avoid warnings
target_include_directories(
    ${ELSA_MODULE_TARGET_NAME} SYSTEM
    PUBLIC $<BUILD_INTERFACE:${Cudnn_INCLUDE_DIR}> $<BUILD_INTERFACE:${Cublas_INCLUDE_DIRS}>
           $<BUILD_INTERFACE:${Npp_INCLUDE_DIRS}>
)
 
write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES )
