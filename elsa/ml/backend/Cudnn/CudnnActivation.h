#pragma once

#include "elsaDefines.h"
#include "VolumeDescriptor.h"
#include "CudnnLayer.h"
#include "CudnnContext.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class CudnnActivation : public CudnnLayer<data_t>
            {
            public:
                CudnnActivation(const VolumeDescriptor& inputDescriptor, float coeff,
                                cudnnActivationMode_t activationMode);

                ~CudnnActivation();

                void forwardPropagate() override;

                void backwardPropagate() override;

            private:
                using BaseType = CudnnLayer<data_t>;

                using BaseType::cudnnContext_;

                using BaseType::input_;
                using BaseType::output_;
                using BaseType::inputGradient_;
                using BaseType::outputGradient_;

                cudnnActivationDescriptor_t activationDescriptor_;
                cudnnActivationMode_t activationMode_;
                float coeff_;
            };

            template <typename data_t>
            struct CudnnSigmoid : public CudnnActivation<data_t> {
                CudnnSigmoid(const VolumeDescriptor& inputDescriptor, float coeff = 0.f);
            };

            template <typename data_t>
            struct CudnnRelu : public CudnnActivation<data_t> {
                CudnnRelu(const VolumeDescriptor& inputDescriptor, float coeff = 0.f);
            };

            template <typename data_t>
            struct CudnnTanh : public CudnnActivation<data_t> {
                CudnnTanh(const VolumeDescriptor& inputDescriptor, float coeff = 0.f);
            };

            template <typename data_t>
            struct CudnnClippedRelu : public CudnnActivation<data_t> {
                CudnnClippedRelu(const VolumeDescriptor& inputDescriptor, float coeff = 0.f);
            };

            template <typename data_t>
            struct CudnnElu : public CudnnActivation<data_t> {
                CudnnElu(const VolumeDescriptor& inputDescriptor, float coeff = 0.f);
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa