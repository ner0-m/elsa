#include "CudnnTrainable.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t, bool isConvolution>
            CudnnTrainable<data_t, isConvolution>::CudnnTrainable(
                const VolumeDescriptor& inputDescriptor, const VolumeDescriptor& outputDescriptor,
                const VolumeDescriptor& weightsDescriptor, bool useBias,
                Initializer weigthsInitializer, Initializer biasInitializer)
                : CudnnLayer<data_t>(inputDescriptor, outputDescriptor, "CudnnTrainable",
                                     /* allowed number of inputs */ 1),
                  weights_(weightsDescriptor),
                  weightsGradient_(weightsDescriptor),
                  weightsGradientAcc_(weightsDescriptor),
                  weigthsInitializer_(weigthsInitializer),
                  biasInitializer_(biasInitializer),
                  useBias_(useBias)
            {
                // Construct bias memory
                if (useBias_) {
                    IndexVector_t biasVec(2);
                    biasVec << 1, weightsDescriptor.getNumberOfCoefficientsPerDimension()[0];
                    VolumeDescriptor biasDesc(biasVec);
                    bias_ = CudnnMemory<data_t>(biasDesc);
                    biasGradient_ = CudnnMemory<data_t>(biasDesc);
                    biasGradientAcc_ = CudnnMemory<data_t>(biasDesc);
                }

                // Set the layer's fan-in and fan-out. This is needed for
                // random initialization of weights and biases
                fanInOut_.first = inputDescriptor.getNumberOfCoefficients();
                fanInOut_.second = outputDescriptor.getNumberOfCoefficients();

                // initialize trainable parameters
                initialize();
            }

            template <typename data_t, bool isConvolution>
            void CudnnTrainable<data_t, isConvolution>::initialize()
            {
                // TODO(tellenbach): Initialize on device (e.g. using Thrust)
                // Initialize weights
                weights_.allocateHostMemory();
                InitializerImpl<data_t>::initialize(weights_.hostMemory->getMemoryHandle(),
                                                    weights_.hostMemory->getSize(),
                                                    weigthsInitializer_, fanInOut_);
                weights_.copyToDevice();

                weightsGradientAcc_.allocateDeviceMemory();
                weightsGradientAcc_.deviceMemory->fill(data_t(0));

                if (useBias_) {
                    // Initialize bias
                    bias_.allocateHostMemory();
                    InitializerImpl<data_t>::initialize(bias_.hostMemory->getMemoryHandle(),
                                                        bias_.hostMemory->getSize(),
                                                        biasInitializer_, fanInOut_);
                    bias_.copyToDevice();
                    biasGradientAcc_.allocateDeviceMemory();
                    biasGradientAcc_.deviceMemory->fill(data_t(0));
                }
            }

            template <typename data_t, bool isConvolution>
            void CudnnTrainable<data_t, isConvolution>::compileForwardStream()
            {
                BaseType::compileForwardStream();

                // Allocate weights- and bias-memory
                weights_.allocateDeviceMemory();
                BaseType::validateDeviceMemory(weights_);

                if (useBias_) {
                    bias_.allocateDeviceMemory();
                    BaseType::validateDeviceMemory(bias_);
                }
            }

            template <typename data_t, bool isConvolution>
            void CudnnTrainable<data_t, isConvolution>::compileBackwardStream()
            {
                BaseType::compileBackwardStream();

                // Allocate weight- and bias-gradients
                weightsGradient_.allocateDeviceMemory();
                BaseType::validateDeviceMemory(weightsGradient_);

                if (useBias_) {
                    biasGradient_.allocateDeviceMemory();
                    BaseType::validateDeviceMemory(biasGradient_);
                }
            }

            template <typename data_t, bool isConvolution>
            void CudnnTrainable<data_t, isConvolution>::updateTrainableParameters()
            {
                BaseType::validateDeviceMemory(weights_);
                BaseType::validateDeviceMemory(weightsGradientAcc_);

                const index_t batchSize =
                    this->inputDescriptor_.front()->getNumberOfCoefficientsPerDimension()[0];

                weightsOptimizer_->updateParameter(
                    weightsGradientAcc_.deviceMemory->getMemoryHandle(), batchSize,
                    weights_.deviceMemory->getMemoryHandle());

                weightsGradientAcc_.deviceMemory->fill(data_t(0));

                if (useBias_) {
                    const index_t size = biasGradientAcc_.deviceMemory->getSize();
                    BaseType::validateDeviceMemory(bias_);
                    BaseType::validateDeviceMemory(biasGradientAcc_);

                    biasOptimizer_->updateParameter(
                        biasGradientAcc_.deviceMemory->getMemoryHandle(), batchSize,
                        bias_.deviceMemory->getMemoryHandle());

                    biasGradientAcc_.deviceMemory->fill(data_t(0));
                }
            }

            template <typename data_t, bool isConvolution>
            void CudnnTrainable<data_t, isConvolution>::accumulateGradients()
            {
                BaseType::validateDeviceMemory(weightsGradient_);
                BaseType::validateDeviceMemory(weightsGradientAcc_);

                // Accumulate weight-gradients
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnAddTensor(
                    /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* contant 1.f */ &CudnnContext::One,
                    /* weights gradient descriptor */ weightsGradient_.getCudnnDescriptor(),
                    /* weights gradient memory */ weightsGradient_.deviceMemory->getMemoryHandle(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* acc weights grad desc */ weightsGradientAcc_.getCudnnDescriptor(),
                    /* acc weights grad mem */
                    weightsGradientAcc_.deviceMemory->getMemoryHandle()));

                if (useBias_) {
                    BaseType::validateDeviceMemory(biasGradient_);
                    BaseType::validateDeviceMemory(biasGradient_);
                    // Accumulate bias-gradients
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnAddTensor(
                        /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                        /* contant 1.f */ &CudnnContext::One,
                        /* weights gradient descriptor */ biasGradient_.getCudnnDescriptor(),
                        /* weights gradient memory */
                        biasGradient_.deviceMemory->getMemoryHandle(),
                        /* constant 0.f */ &CudnnContext::Zero,
                        /* acc weights grad desc */ biasGradientAcc_.getCudnnDescriptor(),
                        /* acc weights grad mem */
                        biasGradientAcc_.deviceMemory->getMemoryHandle()));
                }
            }

            template <typename data_t, bool isConvolution>
            bool CudnnTrainable<data_t, isConvolution>::isTrainable() const
            {
                return true;
            }

            template <typename data_t, bool isConvolution>
            void CudnnTrainable<data_t, isConvolution>::setOptimizer(Optimizer<data_t>* optimizer)
            {
                weightsOptimizer_ = OptimizerFactory<data_t, MlBackend::Cudnn>::run(
                    optimizer, weights_.deviceMemory->getSize(), this->cudnnContext_);

                if (useBias_) {
                    biasOptimizer_ = OptimizerFactory<data_t, MlBackend::Cudnn>::run(
                        optimizer, bias_.deviceMemory->getSize(), this->cudnnContext_);
                }
            }

            template class CudnnTrainable<float, false>;
            // template class CudnnTrainable<float, true>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa