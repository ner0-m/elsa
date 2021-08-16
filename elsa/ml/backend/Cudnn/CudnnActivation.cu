#include "CudnnActivation.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            CudnnActivation<data_t>::CudnnActivation(const VolumeDescriptor& inputDescriptor,
                                                     float coeff,
                                                     cudnnActivationMode_t activationMode)
                : CudnnLayer<data_t>(inputDescriptor, inputDescriptor, "CudnnActivation"),
                  coeff_(coeff),
                  activationMode_(activationMode)
            {
                // Create activation descriptor
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnCreateActivationDescriptor(&activationDescriptor_));

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSetActivationDescriptor(
                    activationDescriptor_, activationMode_, CUDNN_PROPAGATE_NAN, coeff_));
            }

            template <typename data_t>
            CudnnActivation<data_t>::~CudnnActivation()
            {
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnDestroyActivationDescriptor(activationDescriptor_));
            }

            template <typename data_t>
            void CudnnActivation<data_t>::forwardPropagate()
            {
                BaseType::validateForwardPropagation();

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnActivationForward(
                    /* cudnn-handle */ cudnnContext_->getCudnnHandle(),
                    /* activation descriptor */ activationDescriptor_,
                    /* constant 1.f */ &CudnnContext::One,
                    /* input descriptor */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* output descriptor */ output_.getCudnnDescriptor(),
                    /* output memory */ output_.deviceMemory->getMemoryHandle()));
            }

            template <typename data_t>
            void CudnnActivation<data_t>::backwardPropagate()
            {
                BaseType::validateBackwardPropagation();

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnActivationBackward(
                    /* cudnn-handle */ cudnnContext_->getCudnnHandle(),
                    /* activation descriptor */ activationDescriptor_,
                    /* constant 1.f */ &CudnnContext::One,
                    /* output descriptor */ output_.getCudnnDescriptor(),
                    /* output memory */ output_.deviceMemory->getMemoryHandle(),
                    /* output-grad descriptor */ outputGradient_.front().getCudnnDescriptor(),
                    /* output-grad memory */
                    outputGradient_.front().deviceMemory->getMemoryHandle(),
                    /* input descriptor */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* input-gradient descriptor */ inputGradient_.front().getCudnnDescriptor(),
                    /* input-gradient memory */
                    inputGradient_.front().deviceMemory->getMemoryHandle()));
            }

            template <typename data_t>
            CudnnSigmoid<data_t>::CudnnSigmoid(const VolumeDescriptor& inputDescriptor, float coeff)
                : CudnnActivation<data_t>(inputDescriptor, coeff, CUDNN_ACTIVATION_SIGMOID)
            {
            }

            template <typename data_t>
            CudnnRelu<data_t>::CudnnRelu(const VolumeDescriptor& inputDescriptor, float coeff)
                : CudnnActivation<data_t>(inputDescriptor, coeff, CUDNN_ACTIVATION_RELU)
            {
            }

            template <typename data_t>
            CudnnTanh<data_t>::CudnnTanh(const VolumeDescriptor& inputDescriptor, float coeff)
                : CudnnActivation<data_t>(inputDescriptor, coeff, CUDNN_ACTIVATION_TANH)
            {
            }

            template <typename data_t>
            CudnnClippedRelu<data_t>::CudnnClippedRelu(const VolumeDescriptor& inputDescriptor,
                                                       float coeff)
                : CudnnActivation<data_t>(inputDescriptor, coeff, CUDNN_ACTIVATION_CLIPPED_RELU)
            {
            }

            template <typename data_t>
            CudnnElu<data_t>::CudnnElu(const VolumeDescriptor& inputDescriptor, float coeff)
                : CudnnActivation<data_t>(inputDescriptor, coeff, CUDNN_ACTIVATION_ELU)
            {
            }

            template class CudnnActivation<float>;
            template struct CudnnSigmoid<float>;
            template struct CudnnRelu<float>;
            template struct CudnnTanh<float>;
            template struct CudnnClippedRelu<float>;
            template struct CudnnElu<float>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa