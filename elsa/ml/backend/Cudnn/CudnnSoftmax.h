#pragma once

#include "elsaDefines.h"
#include "CudnnLayer.h"
#include "CudnnCommon.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class CudnnSoftmax : public CudnnLayer<data_t>
            {
            public:
                CudnnSoftmax(const VolumeDescriptor& inputDescriptor,
                             const VolumeDescriptor& outputDescriptor);

                void forwardPropagate() override;

                void backwardPropagate() override;

            private:
                using BaseType = CudnnLayer<data_t>;

                using BaseType::cudnnContext_;

                using BaseType::input_;
                using BaseType::inputGradient_;

                using BaseType::output_;
                using BaseType::outputGradient_;

                index_t softmaxAxis_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa