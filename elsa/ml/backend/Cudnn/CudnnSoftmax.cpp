#include "CudnnSoftmax.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            CudnnSoftmax<data_t>::CudnnSoftmax(const VolumeDescriptor& inputDescriptor,
                                               const VolumeDescriptor& outputDescriptor)
                : CudnnLayer<data_t>(inputDescriptor, outputDescriptor, "CudnnSoftmax")
            {
            }

            template <typename data_t>
            void CudnnSoftmax<data_t>::forwardPropagate()
            {
                BaseType::validateForwardPropagation();

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSoftmaxForward(
                    /* Cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* Softmax algo */ CUDNN_SOFTMAX_ACCURATE,
                    /* Softmax mode */ CUDNN_SOFTMAX_MODE_CHANNEL,
                    /* constant 1.f */ &CudnnContext::One,
                    /* input descriptor */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* output descriptor */ output_.getCudnnDescriptor(),
                    /* output memory */ output_.deviceMemory->getMemoryHandle()));
            }

            template <typename data_t>
            void CudnnSoftmax<data_t>::backwardPropagate()
            {
                BaseType::validateBackwardPropagation();

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSoftmaxBackward(
                    /* Cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* Softmax algo */ CUDNN_SOFTMAX_ACCURATE,
                    /* Softmax mode */ CUDNN_SOFTMAX_MODE_CHANNEL,
                    /* constant 1.f */ &CudnnContext::One,
                    /* input descriptor */ output_.getCudnnDescriptor(),
                    /* input memory */ output_.deviceMemory->getMemoryHandle(),
                    /* output grad descriptor */ outputGradient_.front().getCudnnDescriptor(),
                    /* output grad memory */
                    outputGradient_.front().deviceMemory->getMemoryHandle(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* input grad descriptor */ inputGradient_.front().getCudnnDescriptor(),
                    /* input grad memory */
                    inputGradient_.front().deviceMemory->getMemoryHandle()));
            }

            template class CudnnSoftmax<float>;

        } // namespace detail
    }     // namespace ml
} // namespace elsa