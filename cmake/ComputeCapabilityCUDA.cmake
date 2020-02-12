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
