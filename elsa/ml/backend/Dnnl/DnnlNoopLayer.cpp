#include "DnnlNoopLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlNoopLayer<data_t>::DnnlNoopLayer(const VolumeDescriptor& inputDescriptor)
            : DnnlLayer<data_t>(inputDescriptor, inputDescriptor, "DnnlNoopLayer")
        {
        }

        template <typename data_t>
        void DnnlNoopLayer<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();
            BaseType::validateDnnlMemory(_input.front().effectiveMemory);
            _output.effectiveMemory = _input.front().effectiveMemory;
        }

        template <typename data_t>
        void DnnlNoopLayer<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();
            BaseType::validateDnnlMemory(_outputGradient.front().effectiveMemory);
            _inputGradient.front().effectiveMemory = _outputGradient.front().effectiveMemory;
        }

        template class DnnlNoopLayer<float>;
    } // namespace detail
} // namespace elsa::ml
