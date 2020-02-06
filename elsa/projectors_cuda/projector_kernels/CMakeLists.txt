cmake_minimum_required(VERSION 3.10)

################################################################################################
# Automatic GPU detection is not included with every CMake release (e.g. it is unavailable in 
# the CMake version installed with the default Ubuntu package manager)
# 
# Taken from the official CMake repository, with minor modifications:
# https://gitlab.kitware.com/cmake/cmake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
#
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   CUDA_DETECT_INSTALLED_GPUS(OUT_VARIABLE)
#
function(CUDA_DETECT_INSTALLED_GPUS OUT_VARIABLE)
  set(file "${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cu")

  file(WRITE ${file} ""
    "#include <cuda_runtime.h>\n"
    "#include <cstdio>\n"
    "int main()\n"
    "{\n"
    "  int count = 0;\n"
    "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
    "  if (count == 0) return -1;\n"
    "  for (int device = 0; device < count; ++device)\n"
    "  {\n"
    "    cudaDeviceProp prop;\n"
    "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
    "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
    "  }\n"
    "  return 0;\n"
    "}\n")

  try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file}
            RUN_OUTPUT_VARIABLE compute_capabilities)

  # Filter unrelated content out of the output.
  string(REGEX MATCHALL "[0-9]+\\.[0-9]+" compute_capabilities "${compute_capabilities}")

  if(run_result EQUAL 0)
    string(REPLACE "2.1" "2.1(2.0)" compute_capabilities "${compute_capabilities}")
    set(CUDA_GPU_DETECT_OUTPUT ${compute_capabilities}
      CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
  endif()

  if(NOT CUDA_GPU_DETECT_OUTPUT)
      message(WARNING "Automatic GPU detection failed. Defaulting to 3.0. You can also set CUDA_ARCH_TYPES manually.")
      set(${OUT_VARIABLE} "3.0" PARENT_SCOPE)
      separate_arguments(OUT_VARIABLE)
  else()
    # Filter based on CUDA version supported archs
    message(STATUS "Automatically detected GPU architectures: ${CUDA_GPU_DETECT_OUTPUT}")
    separate_arguments(CUDA_GPU_DETECT_OUTPUT)
    set(${OUT_VARIABLE} ${CUDA_GPU_DETECT_OUTPUT} PARENT_SCOPE)
  endif()

  list(REMOVE_DUPLICATES OUT_VARIABLE)
  list(SORT OUT_VARIABLE)
endfunction()

# make sure host compiler used by NVCC is the one used for the rest of the project
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

set(CUDA_ARCH_TYPES "Auto" CACHE STRING "Space-separated list of target GPU architectures. Defaults to Auto. If Auto attempts to detect and target all installed GPUS.")

if ("${CUDA_ARCH_TYPES}" STREQUAL "Auto")
  CUDA_DETECT_INSTALLED_GPUS(TARGET_GPUS)
else()
  set(TARGET_GPUS CUDA_ARCH_TYPES)
endif()

# fmad can reduce accuracy, but only gives a negligible boost to speed
# use expt-relaxed-constexpr flag to suppress warnings caused by including the Eigen header
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --expt-relaxed-constexpr")

# set target GPU architectures
foreach(TARGET_GPU ${TARGET_GPUS})
  string(REPLACE "." "" TARGET_GPU ${TARGET_GPU})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_${TARGET_GPU},code=[sm_${TARGET_GPU},compute_${TARGET_GPU}]")
endforeach()

if (ELSA_BUILD_WITH_MORE_WARNINGS)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wreorder -Xptxas=-warn-spills,-warn-lmem-usage")
endif()

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
set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES CUDA_STANDARD 14 POSITION_INDEPENDENT_CODE ON)

# CUDA doesn't support linking against a library built under the C++17 standard,
# we only need the elsa.h header -> manually specify include directories
target_include_directories(${ELSA_MODULE_TARGET_NAME} PUBLIC 
        $<TARGET_PROPERTY:elsa_core,INTERFACE_INCLUDE_DIRECTORIES>
        $<INSTALL_INTERFACE:include/elsa/${ELSA_MODULE_NAME}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# register the module
registerComponent(${ELSA_MODULE_NAME})

# install the kernels as a module
InstallElsaModule(${ELSA_MODULE_NAME} ${ELSA_MODULE_TARGET_NAME} ${ELSA_MODULE_EXPORT_TARGET})