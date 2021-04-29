#pragma once

#include <string>
#include <iostream>

#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <npp.h>
#include <nppi.h>

#include "Logger.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            // CUDA: use 512 threads per block
            const int ELSA_CUDA_NUM_THREADS = 512;

            // CUDA: number of blocks for threads.
            inline int ELSA_CUDA_GET_BLOCKS(const int N)
            {
                return (N + ELSA_CUDA_NUM_THREADS - 1) / ELSA_CUDA_NUM_THREADS;
            }

#define ELSA_CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

            template <typename data_t>
            struct TypeToCudnnTypeTag;

            template <>
            struct TypeToCudnnTypeTag<float> {
                static constexpr cudnnDataType_t Value = CUDNN_DATA_FLOAT;
            };

            template <>
            struct TypeToCudnnTypeTag<double> {
                static constexpr cudnnDataType_t Value = CUDNN_DATA_DOUBLE;
            };

            static std::string getCudnnBackendStatusAsString(cudnnStatus_t status)
            {
                switch (status) {
                    case cudnnStatus_t::CUDNN_STATUS_SUCCESS:
                        return "CUDNN_STATUS_SUCCESS";
                    case cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED:
                        return "CUDNN_STATUS_NOT_INITIALIZED";
                    case cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED:
                        return "CUDNN_STATUS_ALLOC_FAILED";
                    case cudnnStatus_t::CUDNN_STATUS_BAD_PARAM:
                        return "CUDNN_STATUS_BAD_PARAM";
                    case cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH:
                        return "CUDNN_STATUS_ARCH_MISMATCH";
                    case cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR:
                        return "CUDNN_STATUS_MAPPING_ERROR";
                    case cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED:
                        return "CUDNN_STATUS_EXECUTION_FAILED";
                    case cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR:
                        return "CUDNN_STATUS_INTERNAL_ERROR";
                    case cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED:
                        return "CUDNN_STATUS_NOT_SUPPORTED";
                    case cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR:
                        return "CUDNN_STATUS_LICENSE_ERROR";
                    case cudnnStatus_t::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
                        return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
                    case cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS:
                        return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
                    case cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
                        return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
                    default:
                        assert(false && "This exectution path of the code should never be reached");
                        return "This exectution path of the code should never be reached";
                }
            }

            static std::string getCudnnBackendStatusAsString(cublasStatus_t status)
            {
                switch (status) {
                    case CUBLAS_STATUS_SUCCESS:
                        return "CUBLAS_STATUS_SUCCESS";
                    case CUBLAS_STATUS_NOT_INITIALIZED:
                        return "CUBLAS_STATUS_NOT_INITIALIZED";
                    case CUBLAS_STATUS_ALLOC_FAILED:
                        return "CUBLAS_STATUS_ALLOC_FAILED";
                    case CUBLAS_STATUS_INVALID_VALUE:
                        return "CUBLAS_STATUS_INVALID_VALUE";
                    case CUBLAS_STATUS_ARCH_MISMATCH:
                        return "CUBLAS_STATUS_ARCH_MISMATCH";
                    case CUBLAS_STATUS_MAPPING_ERROR:
                        return "CUBLAS_STATUS_MAPPING_ERROR";
                    case CUBLAS_STATUS_EXECUTION_FAILED:
                        return "CUBLAS_STATUS_EXECUTION_FAILED";
                    case CUBLAS_STATUS_INTERNAL_ERROR:
                        return "CUBLAS_STATUS_INTERNAL_ERROR";
                    case CUBLAS_STATUS_NOT_SUPPORTED:
                        return "CUBLAS_STATUS_NOT_SUPPORTED";
                    case CUBLAS_STATUS_LICENSE_ERROR:
                        return "CUBLAS_STATUS_LICENSE_ERROR";
                    default:
                        assert(false && "This exectution path of the code should never be reached");
                        return "This exectution path of the code should never be reached";
                }
            }

            static std::string getCudnnBackendStatusAsString(curandStatus_t status)
            {
                switch (status) {
                    case CURAND_STATUS_SUCCESS:
                        return "CURAND_STATUS_SUCCESS";
                    case CURAND_STATUS_VERSION_MISMATCH:
                        return "CURAND_STATUS_VERSION_MISMATCH";
                    case CURAND_STATUS_NOT_INITIALIZED:
                        return "CURAND_STATUS_NOT_INITIALIZED";
                    case CURAND_STATUS_ALLOCATION_FAILED:
                        return "CURAND_STATUS_ALLOCATION_FAILED";
                    case CURAND_STATUS_TYPE_ERROR:
                        return "CURAND_STATUS_TYPE_ERROR";
                    case CURAND_STATUS_OUT_OF_RANGE:
                        return "CURAND_STATUS_OUT_OF_RANGE";
                    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
                    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
                    case CURAND_STATUS_LAUNCH_FAILURE:
                        return "CURAND_STATUS_LAUNCH_FAILURE";
                    case CURAND_STATUS_PREEXISTING_FAILURE:
                        return "CURAND_STATUS_PREEXISTING_FAILURE";
                    case CURAND_STATUS_INITIALIZATION_FAILED:
                        return "CURAND_STATUS_INITIALIZATION_FAILED";
                    case CURAND_STATUS_ARCH_MISMATCH:
                        return "CURAND_STATUS_ARCH_MISMATCH";
                    case CURAND_STATUS_INTERNAL_ERROR:
                        return "CURAND_STATUS_INTERNAL_ERROR";
                    default:
                        assert(false && "This exectution path of the code should never be reached");
                        return "This exectution path of the code should never be reached";
                }
            }

            static std::string getCudnnBackendStatusAsString(NppStatus status)
            {
                switch (status) {
                    case NPP_NOT_SUPPORTED_MODE_ERROR:
                        return "NPP_NOT_SUPPORTED_MODE_ERROR";
                    case NPP_INVALID_HOST_POINTER_ERROR:
                        return "NPP_INVALID_HOST_POINTER_ERROR";
                    case NPP_INVALID_DEVICE_POINTER_ERROR:
                        return "NPP_INVALID_DEVICE_POINTER_ERROR";
                    case NPP_LUT_PALETTE_BITSIZE_ERROR:
                        return "NPP_LUT_PALETTE_BITSIZE_ERROR";
                    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
                        return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
                    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
                        return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
                    case NPP_TEXTURE_BIND_ERROR:
                        return "NPP_TEXTURE_BIND_ERROR";
                    case NPP_WRONG_INTERSECTION_ROI_ERROR:
                        return "NPP_WRONG_INTERSECTION_ROI_ERROR";
                    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
                        return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
                    case NPP_MEMFREE_ERROR:
                        return "NPP_MEMFREE_ERROR";
                    case NPP_MEMSET_ERROR:
                        return "NPP_MEMSET_ERROR";
                    case NPP_MEMCPY_ERROR:
                        return "NPP_MEMCPY_ERROR";
                    case NPP_ALIGNMENT_ERROR:
                        return "NPP_ALIGNMENT_ERROR";
                    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
                        return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
                    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
                        return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
                    case NPP_QUALITY_INDEX_ERROR:
                        return "NPP_QUALITY_INDEX_ERROR";
                    case NPP_RESIZE_NO_OPERATION_ERROR:
                        return "NPP_RESIZE_NO_OPERATION_ERROR";
                    case NPP_OVERFLOW_ERROR:
                        return "NPP_OVERFLOW_ERROR";
                    case NPP_NOT_EVEN_STEP_ERROR:
                        return "NPP_NOT_EVEN_STEP_ERROR";
                    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
                        return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
                    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
                        return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
                    case NPP_CORRUPTED_DATA_ERROR:
                        return "NPP_CORRUPTED_DATA_ERROR";
                    case NPP_CHANNEL_ORDER_ERROR:
                        return "NPP_CHANNEL_ORDER_ERROR";
                    case NPP_ZERO_MASK_VALUE_ERROR:
                        return "NPP_ZERO_MASK_VALUE_ERROR";
                    case NPP_QUADRANGLE_ERROR:
                        return "NPP_QUADRANGLE_ERROR";
                    case NPP_RECTANGLE_ERROR:
                        return "NPP_RECTANGLE_ERROR";
                    case NPP_COEFFICIENT_ERROR:
                        return "NPP_COEFFICIENT_ERROR";
                    case NPP_NUMBER_OF_CHANNELS_ERROR:
                        return "NPP_NUMBER_OF_CHANNELS_ERROR";
                    case NPP_COI_ERROR:
                        return "NPP_COI_ERROR";
                    case NPP_DIVISOR_ERROR:
                        return "NPP_DIVISOR_ERROR";
                    case NPP_CHANNEL_ERROR:
                        return "NPP_CHANNEL_ERROR";
                    case NPP_STRIDE_ERROR:
                        return "NPP_STRIDE_ERROR";
                    case NPP_ANCHOR_ERROR:
                        return "NPP_ANCHOR_ERROR";
                    case NPP_MASK_SIZE_ERROR:
                        return "NPP_MASK_SIZE_ERROR";
                    case NPP_RESIZE_FACTOR_ERROR:
                        return "NPP_RESIZE_FACTOR_ERROR";
                    case NPP_INTERPOLATION_ERROR:
                        return "NPP_INTERPOLATION_ERROR";
                    case NPP_MIRROR_FLIP_ERROR:
                        return "NPP_MIRROR_FLIP_ERROR";
                    case NPP_MOMENT_00_ZERO_ERROR:
                        return "NPP_MOMENT_00_ZERO_ERROR";
                    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
                        return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
                    case NPP_THRESHOLD_ERROR:
                        return "NPP_THRESHOLD_ERROR";
                    case NPP_CONTEXT_MATCH_ERROR:
                        return "NPP_CONTEXT_MATCH_ERROR";
                    case NPP_FFT_FLAG_ERROR:
                        return "NPP_FFT_FLAG_ERROR ";
                    case NPP_FFT_ORDER_ERROR:
                        return "NPP_FFT_ORDER_ERROR";
                    case NPP_STEP_ERROR:
                        return "NPP_STEP_ERROR";
                    case NPP_DATA_TYPE_ERROR:
                        return "NPP_DATA_TYPE_ERROR";
                    case NPP_OUT_OFF_RANGE_ERROR:
                        return "NPP_OUT_OFF_RANGE_ERROR";
                    case NPP_DIVIDE_BY_ZERO_ERROR:
                        return "NPP_DIVIDE_BY_ZERO_ERROR";
                    case NPP_MEMORY_ALLOCATION_ERR:
                        return "NPP_MEMORY_ALLOCATION_ERR";
                    case NPP_NULL_POINTER_ERROR:
                        return "NPP_NULL_POINTER_ERROR";
                    case NPP_RANGE_ERROR:
                        return "NPP_RANGE_ERROR";
                    case NPP_SIZE_ERROR:
                        return "NPP_SIZE_ERROR";
                    case NPP_BAD_ARGUMENT_ERROR:
                        return "NPP_BAD_ARGUMENT_ERROR";
                    case NPP_NO_MEMORY_ERROR:
                        return "NPP_NO_MEMORY_ERROR";
                    case NPP_NOT_IMPLEMENTED_ERROR:
                        return "NPP_NOT_IMPLEMENTED_ERROR";
                    case NPP_ERROR:
                        return "NPP_ERROR";
                    case NPP_ERROR_RESERVED:
                        return "NPP_ERROR_RESERVED";
                    case NPP_SUCCESS:
                        return "NPP_SUCCESS";
                    case NPP_NO_OPERATION_WARNING:
                        return "NPP_NO_OPERATION_WARNING";
                    case NPP_DIVIDE_BY_ZERO_WARNING:
                        return "NPP_DIVIDE_BY_ZERO_WARNING";
                    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
                        return "NPP_AFFINE_QUAD_INCORRECT_WARNING";
                    case NPP_WRONG_INTERSECTION_ROI_WARNING:
                        return "NPP_WRONG_INTERSECTION_ROI_WARNING";
                    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
                        return "NPP_WRONG_INTERSECTION_QUAD_WARNING";
                    case NPP_DOUBLE_SIZE_WARNING:
                        return "NPP_DOUBLE_SIZE_WARNING";
                    case NPP_MISALIGNED_DST_ROI_WARNING:
                        return "NPP_MISALIGNED_DST_ROI_WARNING";
                    default:
                        assert(false && "This exectution path of the code should never be reached");
                        return "This exectution path of the code should never be reached";
                }
            }

            static std::string getCudnnBackendStatusAsString(cudaError_t status)
            {
                return cudaGetErrorString(status);
            }

            template <typename T>
            struct checkCudnnBackendStatusHelper {
                static void run(T status, const std::string& file, int line)
                {
                    Logger::get("CudnnBackend")->error("Unknown error type at {}:{}", file, line);
                }
            };

            template <>
            struct checkCudnnBackendStatusHelper<cudnnStatus_t> {
                static void run(cudnnStatus_t status, const std::string& file, int line)
                {
                    if (status != CUDNN_STATUS_SUCCESS) {
                        Logger::get("CudnnBackend")
                            ->error("Cudnn error at {}:{}: {}", file, line,
                                    getCudnnBackendStatusAsString(status));
                    }
                }
            };

            template <>
            struct checkCudnnBackendStatusHelper<NppStatus> {
                static void run(NppStatus status, const std::string& file, int line)
                {
                    if (status != NPP_SUCCESS) {
                        Logger::get("CudnnBackend")
                            ->error("Npp error at {}:{}: {}", file, line,
                                    getCudnnBackendStatusAsString(status));
                    }
                }
            };

            template <>
            struct checkCudnnBackendStatusHelper<curandStatus_t> {
                static void run(curandStatus_t status, const std::string& file, int line)
                {
                    if (status != CURAND_STATUS_SUCCESS) {
                        Logger::get("CudnnBackend")
                            ->error("Curand error at {}:{}: {}", file, line,
                                    getCudnnBackendStatusAsString(status));
                    }
                }
            };

            template <>
            struct checkCudnnBackendStatusHelper<cudaError_t> {
                static void run(cudaError_t status, const std::string& file, int line)
                {
                    if (status != cudaSuccess) {
                        Logger::get("CudnnBackend")
                            ->error("Cuda error at {}:{}: {}", file, line,
                                    getCudnnBackendStatusAsString(status));
                    }
                }
            };

            template <>
            struct checkCudnnBackendStatusHelper<cublasStatus_t> {
                static void run(cublasStatus_t status, const std::string& file, int line)
                {
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        Logger::get("CudnnBackend")
                            ->error("Cublas error at {}:{}: {}", file, line,
                                    getCudnnBackendStatusAsString(status));
                    }
                }
            };

            template <typename StatusT>
            static void checkCudnnBackendStatus(StatusT status, const std::string& file, int line)
            {
                checkCudnnBackendStatusHelper<StatusT>::run(status, file, line);
            }

#define ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(status) \
    checkCudnnBackendStatus(status, __FILE__, __LINE__)

        } // namespace detail

    } // namespace ml
} // namespace elsa
