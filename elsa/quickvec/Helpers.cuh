#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * @brief Custom macro to check CUDA API calls for errors with line information
 */
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/**
 * @brief Custom macro to check cuBLAS API calls for errors with line information
 */
#define cublasErrchk(ans)                        \
    {                                            \
        cublasAssert((ans), __FILE__, __LINE__); \
    }
inline void cublasAssert(cublasStatus_t status, const char* file, int line, bool abort = true)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasAssert: %d %s %d\n", status, file, line);
        if (abort)
            exit(status);
    }
}
