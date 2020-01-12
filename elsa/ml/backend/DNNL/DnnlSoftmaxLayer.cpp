#include "DnnlSoftmaxLayer.h"

namespace elsa
{
    template <typename data_t>
    DnnlSoftmaxLayer<data_t>::DnnlSoftmaxLayer(const DataDescriptor& inputDescriptor,
                                               const DataDescriptor& outputDescriptor)
        : DnnlLayer<data_t>(inputDescriptor, outputDescriptor)
    {
        _softmaxAxis = (inputDescriptor.getNumberOfDimensions() == 1) ? 0 : 1;
    }

    template <typename data_t>
    void DnnlSoftmaxLayer<data_t>::compileForwardStream()
    {
        auto desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training,
                                                _input.descriptor, _softmaxAxis);

        _forwardPrimitiveDescriptor = dnnl::softmax_forward::primitive_desc(desc, *_engine);

        // Set forward primitive
        _forwardStream.primitives.push_back(dnnl::softmax_forward(_forwardPrimitiveDescriptor));

        _output.effectiveMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardStream.arguments.push_back(
            {{DNNL_ARG_SRC, *_input.effectiveMemory}, {DNNL_ARG_DST, *_output.effectiveMemory}});
    }

    template class DnnlSoftmaxLayer<float>;
}; // namespace elsa