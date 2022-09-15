#pragma once

// enumerate device compilers we know about for both device and host
#define ELSA_HOST_COMPILER_UNKNOWN 0
#define ELSA_HOST_COMPILER_MSVC 1
#define ELSA_HOST_COMPILER_GCC 2
#define ELSA_HOST_COMPILER_CLANG 3
#define ELSA_HOST_COMPILER_INTEL 4

#define ELSA_DEVICE_COMPILER_UNKNOWN 0
#define ELSA_DEVICE_COMPILER_MSVC 1
#define ELSA_DEVICE_COMPILER_GCC 2
#define ELSA_DEVICE_COMPILER_CLANG 3
#define ELSA_DEVICE_COMPILER_NVCC 4

// figure out which host compiler we're using, this is more comprehensive than what we are currently
// using, as this is similar to the logic used in thrust
#if defined(_MSC_VER)
#define ELSA_HOST_COMPILER ELSA_HOST_COMPILER_MSVC
#elif defined(__ICC)
#define ELSA_HOST_COMPILER ELSA_HOST_COMPILER_INTEL
#elif defined(__clang__)
#define ELSA_HOST_COMPILER ELSA_HOST_COMPILER_CLANG
#elif defined(__GNUC__)
#define ELSA_HOST_COMPILER ELSA_HOST_COMPILER_GCC
#else
#define ELSA_HOST_COMPILER ELSA_HOST_COMPILER_UNKNOWN
#endif // ELSA_HOST_COMPILER

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define ELSA_DEVICE_COMPILER ELSA_DEVICE_COMPILER_NVCC
#elif ELSA_HOST_COMPILER == ELSA_HOST_COMPILER_MSVC
#define ELSA_DEVICE_COMPILER ELSA_DEVICE_COMPILER_MSVC
#elif ELSA_HOST_COMPILER == ELSA_HOST_COMPILER_GCC
#define ELSA_DEVICE_COMPILER ELSA_DEVICE_COMPILER_GCC
#elif ELSA_HOST_COMPILER == ELSA_HOST_COMPILER_CLANG
// CUDA-capable clang should behave similar to NVCC.
#if defined(__CUDA__)
#define ELSA_DEVICE_COMPILER ELSA_DEVICE_COMPILER_NVCC
#else
#define ELSA_DEVICE_COMPILER ELSA_DEVICE_COMPILER_CLANG
#endif

#else
#define ELSA_DEVICE_COMPILER ELSA_DEVICE_COMPILER_UNKNOWN
#endif

#if ELSA_DEVICE_COMPILER != ELSA_DEVICE_COMPILER_NVCC

// since __host__ & __device__ might have already be defined, only #define them if not defined
// already

#ifndef __host__
#define __host__
#endif // __host__

#ifndef __device__
#define __device__
#endif // __device__

#endif
