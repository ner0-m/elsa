#include "CudnnPooling.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            CudnnPooling<data_t>::CudnnPooling(const VolumeDescriptor& inputDescriptor,
                                               const VolumeDescriptor& outputDescriptor,
                                               const IndexVector_t& poolingWindow,
                                               const IndexVector_t& poolingStride,
                                               cudnnPoolingMode_t poolingMode)
                : CudnnLayer<data_t>(inputDescriptor, outputDescriptor, "CudnnPooling",
                                     /* allowed number of inputs */ 1)
            {
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnCreatePoolingDescriptor(&poolingDescriptor_));
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSetPooling2dDescriptor(
                    /* pooling descriptor */ poolingDescriptor_,
                    /* pooling mode */ poolingMode,
                    /* nan propagation */ CUDNN_PROPAGATE_NAN,
                    /* window height */ poolingWindow[0],
                    /* window width */ poolingWindow[1],
                    /* vertical padding */ 0,
                    /* horizontal padding */ 0,
                    /* vertical strides */ poolingStride[0],
                    /* horizontal strides */ poolingStride[1]));
            }

            template <typename data_t>
            CudnnPooling<data_t>::~CudnnPooling()
            {
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnDestroyPoolingDescriptor(poolingDescriptor_));
            }

            template <typename data_t>
            void CudnnPooling<data_t>::forwardPropagate()
            {
                BaseType::validateForwardPropagation();

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnPoolingForward(
                    /* Cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* pooling descriptor */ poolingDescriptor_,
                    /* constant 1.f */ &CudnnContext::One,
                    /* input desc */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* output desc */ output_.getCudnnDescriptor(),
                    /* output memory */ output_.deviceMemory->getMemoryHandle()));
            }

            template <typename data_t>
            void CudnnPooling<data_t>::backwardPropagate()
            {
                BaseType::validateBackwardPropagation();

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnPoolingBackward(
                    /* Cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* pooling descriptor */ poolingDescriptor_,
                    /* constant 1.f */ &CudnnContext::One,
                    /* output desc */ output_.getCudnnDescriptor(),
                    /* output memory */ output_.deviceMemory->getMemoryHandle(),
                    /* output grad desc */ outputGradient_.front().getCudnnDescriptor(),
                    /* output grad memory */
                    outputGradient_.front().deviceMemory->getMemoryHandle(),
                    /* input desc */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* input grad desc */ inputGradient_.front().getCudnnDescriptor(),
                    /* input grad memory */
                    inputGradient_.front().deviceMemory->getMemoryHandle()));
            }

            template <typename data_t>
            CudnnMaxPooling<data_t>::CudnnMaxPooling(const VolumeDescriptor& inputDescriptor,
                                                     const VolumeDescriptor& outputDescriptor,
                                                     const IndexVector_t& poolingWindow,
                                                     const IndexVector_t& poolingStride)
                : CudnnPooling<data_t>(inputDescriptor, outputDescriptor, poolingWindow,
                                       poolingStride, CUDNN_POOLING_MAX)
            {
            }

            template class CudnnPooling<float>;
            template class CudnnMaxPooling<float>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa