cmake_minimum_required(VERSION 3.10)

set(CUDA_ARCH_TYPE "30" CACHE STRING "Compute capability of target GPU architecture")

#fmad can reduce accuracy, but only gives a negligible boost to speed
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --verbose")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_${CUDA_ARCH_TYPE},code=[sm_${CUDA_ARCH_TYPE},compute_${CUDA_ARCH_TYPE}]")

# set the name of the module
set(ELSA_MODULE_NAME projector_kernels)
set(ELSA_MODULE_TARGET_NAME elsa_${ELSA_MODULE_NAME})
set(ELSA_MODULE_EXPORT_TARGET elsa_${ELSA_MODULE_NAME}Targets)

# list all the headers of the module
set(MODULE_HEADERS
        TraverseSiddonsCUDA.cuh
        TraverseJosephsCUDA.cuh)

# list all the code files of the module
set(MODULE_SOURCES
        TraverseSiddonsCUDA.cu
        TraverseJosephsCUDA.cu)

# build the module library
add_library(${ELSA_MODULE_TARGET_NAME} ${MODULE_HEADERS} ${MODULE_SOURCES})
add_library(elsa::${ELSA_MODULE_NAME} ALIAS ${ELSA_MODULE_TARGET_NAME})

#CUDA doesn't support C++17, require C++14
target_compile_features(${ELSA_MODULE_TARGET_NAME} PUBLIC cxx_std_14)

#CUDA doesn't support linking against a library built under the C++17 standard,
#we only need the elsa.h header -> manually specify include directories
target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC 
        $<TARGET_PROPERTY:elsa_core,INTERFACE_INCLUDE_DIRECTORIES>)

# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})