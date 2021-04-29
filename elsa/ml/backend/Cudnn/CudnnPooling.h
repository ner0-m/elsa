#pragma once

#include "elsaDefines.h"
#include "CudnnLayer.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class CudnnPooling : public CudnnLayer<data_t>
            {
            public:
                CudnnPooling(const VolumeDescriptor& inputDescriptor,
                             const VolumeDescriptor& outputDescriptor,
                             const IndexVector_t& poolingWindow, const IndexVector_t& poolingStride,
                             cudnnPoolingMode_t poolingMode);

                ~CudnnPooling();

                void forwardPropagate() override;

                void backwardPropagate() override;

            protected:
                using BaseType = CudnnLayer<data_t>;

                using BaseType::cudnnContext_;

                using BaseType::input_;
                using BaseType::inputGradient_;

                using BaseType::output_;
                using BaseType::outputGradient_;

                cudnnPoolingMode_t poolingMode_;

                cudnnPoolingDescriptor_t poolingDescriptor_;
            };

            template <typename data_t>
            struct CudnnMaxPooling : public CudnnPooling<data_t> {
                CudnnMaxPooling(const VolumeDescriptor& inputDescriptor,
                                const VolumeDescriptor& outputDescriptor,
                                const IndexVector_t& poolingWindow,
                                const IndexVector_t& poolingStride);
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa