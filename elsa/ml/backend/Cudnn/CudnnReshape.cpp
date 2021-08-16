#include "CudnnReshape.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            /**************************** CudnnReshape ************************/
            template <typename data_t>
            CudnnReshape<data_t>::CudnnReshape(const VolumeDescriptor& inputDescriptor,
                                               const VolumeDescriptor& outputDescriptor)
                : CudnnLayer<data_t>(inputDescriptor, outputDescriptor, "CudnnFlatten")
            {
                assert(inputDescriptor.getNumberOfCoefficients()
                           == outputDescriptor.getNumberOfCoefficients()
                       && "Number of coefficients of input- and output must match");
            }

            template <typename data_t>
            void CudnnReshape<data_t>::compileForwardStream()
            {
                BaseType::compileForwardStream();
                output_.deviceMemory = input_.front().deviceMemory;
            }

            template <typename data_t>
            void CudnnReshape<data_t>::compileBackwardStream()
            {
                BaseType::compileBackwardStream();
                inputGradient_.front().deviceMemory = outputGradient_.front().deviceMemory;
            }

            template class CudnnReshape<float>;

            /**************************** CudnnFlatten ************************/
            template <typename data_t>
            CudnnFlatten<data_t>::CudnnFlatten(const VolumeDescriptor& inputDescriptor)
                : CudnnReshape<data_t>(
                    inputDescriptor,
                    VolumeDescriptor(IndexVector_t(
                        {{/* batch */
                          inputDescriptor.getNumberOfCoefficientsPerDimension()[0],
                          /* all other dimensions */ inputDescriptor
                              .getNumberOfCoefficientsPerDimension()
                              .tail(inputDescriptor.getNumberOfCoefficientsPerDimension().size()
                                    - 1)
                              .prod()}} // namespace ml
                        )))
            {
            }

            template struct CudnnFlatten<float>;

            /**************************** CudnnUpsampling *********************/
            template <typename data_t>
            CudnnUpsampling<data_t>::CudnnUpsampling(const VolumeDescriptor& inputDescriptor,
                                                     const VolumeDescriptor& outputDescriptor,
                                                     Interpolation interpolation)
                : CudnnLayer<data_t>(inputDescriptor, outputDescriptor, "CudnnUpsampling"),
                  interpolation_(interpolation)
            {
                // Set NPP input sizes
                inputSize_.height = input_.front().getDimensions()[2];
                inputSize_.width = input_.front().getDimensions()[2];
                inputROI_.x = 0;
                inputROI_.y = 0;
                inputROI_.height = inputSize_.height;
                inputROI_.width = inputSize_.width;

                // Set NPP output data
                outputSize_.height = output_.getDimensions()[2];
                outputSize_.width = output_.getDimensions()[3];
                outputROI_.x = 0;
                outputROI_.y = 0;
                outputROI_.height = outputSize_.height;
                outputROI_.width = outputSize_.width;
            }

            template <typename data_t>
            void CudnnUpsampling<data_t>::forwardPropagate()
            {
                BaseType::validateForwardPropagation();

                const index_t C = input_.front().getDimensions()[1];

                for (int n = 0; n < this->getBatchSize(); ++n) {
                    for (int c = 0; c < C; ++c) {
                        const data_t* srcPtr = input_.front().deviceMemory->getMemoryHandle()
                                               + n * C * inputSize_.height * inputSize_.width
                                               + c * inputSize_.height * inputSize_.width;
                        data_t* dstPtr = output_.deviceMemory->getMemoryHandle()
                                         + n * C * outputSize_.height * outputSize_.width
                                         + c * outputSize_.height * outputSize_.width;
                        ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(nppiResize_32f_C1R(
                            srcPtr, sizeof(float) * inputSize_.width, inputSize_, inputROI_, dstPtr,
                            sizeof(float) * outputSize_.width, outputSize_, outputROI_,
                            interpolation_ == Interpolation::NearestNeighbour ? NPPI_INTER_NN
                                                                              : NPPI_INTER_LINEAR));
                    }
                }
            }

            template <typename data_t>
            void CudnnUpsampling<data_t>::backwardPropagate()
            {
                BaseType::validateBackwardPropagation();

                const index_t C = input_.front().getDimensions()[1];

                for (int n = 0; n < this->getBatchSize(); ++n) {
                    for (int c = 0; c < C; ++c) {
                        const data_t* srcPtr =
                            outputGradient_.front().deviceMemory->getMemoryHandle()
                            + n * C * outputSize_.height * outputSize_.width
                            + c * outputSize_.height * outputSize_.width;
                        data_t* dstPtr = inputGradient_.front().deviceMemory->getMemoryHandle()
                                         + n * C * inputSize_.height * inputSize_.width
                                         + c * inputSize_.height * inputSize_.width;
                        ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(nppiResize_32f_C1R(
                            srcPtr, sizeof(float) * outputSize_.width, outputSize_, outputROI_,
                            dstPtr, sizeof(float) * inputSize_.width, inputSize_, inputROI_,
                            interpolation_ == Interpolation::NearestNeighbour ? NPPI_INTER_NN
                                                                              : NPPI_INTER_LINEAR));
                    }
                }
            }

            template class CudnnUpsampling<float>;

        } // namespace detail
    }     // namespace ml
} // namespace elsa