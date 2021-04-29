#include "CudnnNoop.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            CudnnNoop<data_t>::CudnnNoop(const VolumeDescriptor& inputDescriptor)
                : CudnnLayer<data_t>(inputDescriptor, inputDescriptor, "CudnnNoop")
            {
            }

            template <typename data_t>
            void CudnnNoop<data_t>::compileForwardStream()
            {
                BaseType::compileForwardStream();
                output_.deviceMemory = input_.front().deviceMemory;
            }

            template <typename data_t>
            void CudnnNoop<data_t>::compileBackwardStream()
            {
                BaseType::compileBackwardStream();
                inputGradient_.front().deviceMemory = outputGradient_.front().deviceMemory;
            }

            template <typename data_t>
            void CudnnNoop<data_t>::forwardPropagate()
            {
            }

            template <typename data_t>
            void CudnnNoop<data_t>::backwardPropagate()
            {
            }

            template class CudnnNoop<float>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa