#pragma once

#include "elsaDefines.h"
#include "CudnnTrainable.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class CudnnConvolution : public CudnnTrainable<data_t>
            {
            public:
                CudnnConvolution(const VolumeDescriptor& inputDescriptor,
                                 const VolumeDescriptor& outputDescriptor,
                                 const VolumeDescriptor& weightsDescriptor,
                                 const IndexVector_t& strides, const IndexVector_t& paddingLow,
                                 const IndexVector_t& paddingHigh, bool useBias,
                                 Initializer weightsInitializer, Initializer biasInitializer);

                ~CudnnConvolution();

                void compileForwardStream() override;

                void forwardPropagate() override;

                void backwardPropagate() override;

            private:
                using BaseType = CudnnTrainable<data_t>;

                using BaseType::cudnnContext_;

                using BaseType::input_;
                using BaseType::inputGradient_;

                using BaseType::output_;
                using BaseType::outputGradient_;

                using BaseType::weights_;
                using BaseType::weightsGradient_;

                using BaseType::useBias_;
                using BaseType::bias_;
                using BaseType::biasGradient_;

                std::shared_ptr<DeviceMemory<data_t>> workspaceMemory_;

                cudnnConvolutionDescriptor_t convolutionDescriptor_;
                cudnnFilterDescriptor_t cudnnFilterDescriptor_;

                cudnnConvolutionFwdAlgo_t convolutionForwardAlgorithm_;
                cudnnConvolutionBwdDataAlgo_t convolutionBackwardDataAlgorithm_;
                cudnnConvolutionBwdFilterAlgo_t convolutionBackwardFilterAlgorithm_;
            };

            template <typename data_t>
            class CudnnDeconvolution : public CudnnTrainable<data_t>
            {
            public:
                CudnnDeconvolution(const VolumeDescriptor& inputDescriptor,
                                   const VolumeDescriptor& outputDescriptor,
                                   const VolumeDescriptor& weightsDescriptor,
                                   const IndexVector_t& strides, const IndexVector_t& paddingLow,
                                   const IndexVector_t& paddingHigh, bool useBias,
                                   Initializer weightsInitializer, Initializer biasInitializer);

                ~CudnnDeconvolution();

                void compileForwardStream() override;

                void forwardPropagate() override;

                void backwardPropagate() override;

            private:
                using BaseType = CudnnTrainable<data_t>;

                using BaseType::cudnnContext_;

                using BaseType::input_;
                using BaseType::inputGradient_;

                using BaseType::output_;
                using BaseType::outputGradient_;

                using BaseType::weights_;
                using BaseType::weightsGradient_;

                using BaseType::useBias_;
                using BaseType::bias_;
                using BaseType::biasGradient_;

                std::shared_ptr<DeviceMemory<data_t>> workspaceMemory_;

                cudnnConvolutionDescriptor_t convolutionDescriptor_;
                cudnnFilterDescriptor_t cudnnFilterDescriptor_;

                cudnnConvolutionFwdAlgo_t convolutionForwardAlgorithm_;
                cudnnConvolutionBwdDataAlgo_t convolutionBackwardDataAlgorithm_;
                cudnnConvolutionBwdFilterAlgo_t convolutionBackwardFilterAlgorithm_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa