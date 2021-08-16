#include "DnnlFlattenLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlFlattenLayer<data_t>::DnnlFlattenLayer(const VolumeDescriptor& inputDescriptor,
                                                   const VolumeDescriptor& outputDescriptor)
            : DnnlLayer<data_t>(inputDescriptor, outputDescriptor, "DnnlFlattenLayer")
        {
            assert(inputDescriptor.getNumberOfCoefficients()
                       == outputDescriptor.getNumberOfCoefficients()
                   && "Cannot flatten if number of coefficients of input- and output-descriptor do "
                      "not match");
        }

        template <typename data_t>
        void DnnlFlattenLayer<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();

            // Set output-descriptor. This is the flattened input-descriptor
            _output.effectiveMemory = std::make_shared<dnnl::memory>(
                dnnl::memory::desc({{_output.dimensions}, _typeTag, _output.formatTag}), *_engine);

            BaseType::validateDnnlMemory(_input.front().effectiveMemory, _output.effectiveMemory);

            _output.effectiveMemory->set_data_handle(
                _input.front().effectiveMemory->get_data_handle());
        }

        template <typename data_t>
        void DnnlFlattenLayer<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();
            // Set output-memory
            _inputGradient.front().effectiveMemory = std::make_shared<dnnl::memory>(
                dnnl::memory::desc({{_inputGradient.front().dimensions},
                                    _typeTag,
                                    _inputGradient.front().formatTag}),
                *_engine);

            BaseType::validateDnnlMemory(_inputGradient.front().effectiveMemory,
                                         _outputGradient.front().effectiveMemory);
            _inputGradient.front().effectiveMemory->set_data_handle(
                _outputGradient.front().effectiveMemory->get_data_handle());
        }

        template class DnnlFlattenLayer<float>;
    } // namespace detail
} // namespace elsa::ml
