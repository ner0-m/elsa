#include "CudnnDense.h"

#include <cublas_v2.h>

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            CudnnDense<data_t>::CudnnDense(const VolumeDescriptor& inputDescriptor,
                                           const VolumeDescriptor& outputDescriptor,
                                           const VolumeDescriptor& weightsDescriptor, bool useBias,
                                           Initializer weigthsInitializer,
                                           Initializer biasInitializer)
                : CudnnTrainable<data_t, /* isConvolution? */ false>(
                    inputDescriptor, outputDescriptor, weightsDescriptor, useBias,
                    weigthsInitializer, biasInitializer),
                  vectorOfOnes_(nullptr)
            {
                index_t batchSize = inputDescriptor.getNumberOfCoefficientsPerDimension()[0];
            }

            template <typename data_t>
            CudnnDense<data_t>::~CudnnDense()
            {
                // Deallocate vector of 1s
                if (vectorOfOnes_) {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudaFree(vectorOfOnes_));
                }
            }

            template <typename data_t>
            __global__ static void initializeVectorOfOnes(data_t* vec, std::size_t length)
            {
                ELSA_CUDA_KERNEL_LOOP(index, length) { vec[index] = data_t(1.f); }
            }

            template <typename data_t>
            void CudnnDense<data_t>::compileForwardStream()
            {
                BaseType::compileForwardStream();

                if (useBias_) {
                    const index_t batchSize = input_.front().getDimensions()[0];
                    // Initialize a vector of 1s which we need during propagation
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                        cudaMalloc((void**) &vectorOfOnes_, sizeof(data_t) * batchSize));
                    initializeVectorOfOnes<<<ELSA_CUDA_GET_BLOCKS(batchSize),
                                             ELSA_CUDA_NUM_THREADS>>>(vectorOfOnes_, batchSize);
                }
            }

            template <typename data_t>
            void CudnnDense<data_t>::forwardPropagate()
            {
                Logger::get("CudnnDense")->trace("Forward propagate");

                BaseType::validateForwardPropagation();

                // Input descriptor is of shape (n, c, h, w).
                const index_t batchSize = input_.front().getDimensions()[0];
                const auto& inputDims = input_.front().getDimensions();
                const index_t inputSize = inputDims[1] * inputDims[2] * inputDims[3];

                // Output descriptor is of size (n, o, 1, 1)
                const index_t outputSize = output_.getDimensions()[1];

                // cublasSgemm performs the operation
                //    C = alpha * op(A) * op(B) + beta * C,
                // where op(X) is
                //    X for the parameter CUBLAS_OP_N,
                //    X^T for the parameter CUBLAS_OP_T.
                // We calculate the forward-step in two takes:
                //
                //  1. output = 1 * input * weights^T + 0 * output
                //  2. output = 1 * bias * vector-of-1s + 1 * output
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cublasSgemm(
                    /* Cublas handle */ cudnnContext_->getCublasHandle(),
                    /* Don't transpose input  */ CUBLAS_OP_T,
                    /* Transpose weights */ CUBLAS_OP_N,
                    /* Rows of input */ outputSize,
                    /* Cols of weights */ batchSize,
                    /* Cols if input */ inputSize,
                    /* constant 1.f */ &CudnnContext::One,
                    /* weights memory */ weights_.deviceMemory->getMemoryHandle(),
                    /* Leading dimension of input */ inputSize,
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* Leading dimension of weights */ inputSize,
                    /* constant zero */ &CudnnContext::Zero,
                    /* output memory */ output_.deviceMemory->getMemoryHandle(),
                    /* Leading dimension of output */ outputSize));

                if (useBias_) {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cublasSgemm(
                        /* Cublas handle */ cudnnContext_->getCublasHandle(),
                        /* Don't transpose bias */ CUBLAS_OP_N,
                        /* Don't transpose vector of 1s */ CUBLAS_OP_N,
                        /* */ outputSize,
                        /* */ batchSize,
                        /* */ 1,
                        /* constant 1.f */ &CudnnContext::One,
                        /* bias memory */ bias_.deviceMemory->getMemoryHandle(),
                        /* */ outputSize,
                        /* */ vectorOfOnes_,
                        /* */ 1,
                        /* constant 1.f */ &CudnnContext::One,
                        /* output memory */ output_.deviceMemory->getMemoryHandle(),
                        /* */ outputSize));
                }
            }

            template <typename data_t>
            void CudnnDense<data_t>::backwardPropagate()
            {
                BaseType::validateBackwardPropagation();

                // Input descriptor is of shape (n, c, h, w).
                const index_t batchSize = input_.front().getDimensions()[0];
                const auto& inputDims = input_.front().getDimensions();
                const index_t inputSize = inputDims[1] * inputDims[2] * inputDims[3];

                // Output descriptor is of size (n, o, 1, 1)
                const index_t outputSize = output_.getDimensions()[1];

                // db = (dy) * d_one_vec
                if (useBias_) {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cublasSgemv(
                        cudnnContext_->getCublasHandle(), CUBLAS_OP_N, outputSize, batchSize,
                        &CudnnContext::One, outputGradient_.front().deviceMemory->getMemoryHandle(),
                        outputSize, vectorOfOnes_, 1, &CudnnContext::Zero,
                        biasGradient_.deviceMemory->getMemoryHandle(), 1));
                }

                // dw = x * (dy)^T
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cublasSgemm(cudnnContext_->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_T,
                                inputSize, outputSize, batchSize, &CudnnContext::One,
                                input_.front().deviceMemory->getMemoryHandle(), inputSize,
                                outputGradient_.front().deviceMemory->getMemoryHandle(), outputSize,
                                &CudnnContext::Zero,
                                weightsGradient_.deviceMemory->getMemoryHandle(), inputSize));

                // dx = W * dy
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cublasSgemm(cudnnContext_->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                inputSize, batchSize, outputSize, &CudnnContext::One,
                                weights_.deviceMemory->getMemoryHandle(), inputSize,
                                outputGradient_.front().deviceMemory->getMemoryHandle(), outputSize,
                                &CudnnContext::Zero,
                                inputGradient_.front().deviceMemory->getMemoryHandle(), inputSize));

                // Accumulate gradients
                this->accumulateGradients();
            }

            template class CudnnDense<float>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa