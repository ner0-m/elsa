#include "CudnnProjector.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            CudnnProjector<data_t>::CudnnProjector(const VolumeDescriptor& inputDescriptor,
                                                   const VolumeDescriptor& outputDescriptor,
                                                   LinearOperator<data_t>* op)
                : CudnnLayer<data_t>(inputDescriptor, outputDescriptor, "CudnnProjector"),
                  operator_(op)
            {
            }

            template <typename data_t>
            void CudnnProjector<data_t>::forwardPropagate()
            {
                BaseType::validateForwardPropagation();

                input_.front().copyToHost();
                operator_->applyAdjoint();
            }

            template <typename data_t>
            void CudnnProjector<data_t>::backwardPropagate()
            {
                BaseType::validateBackwardPropagation();
                operator_->applyAdjoint();
            }

        } // namespace detail
    }     // namespace ml
} // namespace elsa