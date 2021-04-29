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
            class CudnnProjector : public CudnnLayer<data_t>
            {
            public:
                CudnnProjector(const VolumeDescriptor& inputDescriptor,
                               const VolumeDescriptor& outputDescriptor, LinearOperator<data_t>* op)
                    : CudnnLayer<data_t>(inputDescriptor, outputDescriptor, "CudnnProjector"),
                      operator_(op)
                {
                }

                void forwardPropagate() override
                {
                    BaseType::validateForwardPropagation();

                    input_.front().copyToHost();
                    operator_->applyAdjoint();
                }

                void backwardPropagate() override;

            private:
                using BaseType = CudnnLayer<data_t>;

                using BaseType::input_;
                using BaseType::inputGradient_;

                using BaseType::output_;
                using BaseType::outputGradient_;

                LinearOperator<data_t>* operator_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa