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
                                                _srcMemory->get_desc(), _softmaxAxis);

        _forwardPrimitiveDescriptor = dnnl::softmax_forward::primitive_desc(desc, *_engine);

        // Set forward primitive
        _forwardPrimitives.push_back(dnnl::softmax_forward(_forwardPrimitiveDescriptor));

        _dstMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardArguments.push_back({{DNNL_ARG_SRC, *_srcMemory}, {DNNL_ARG_DST, *_dstMemory}});
    }

    template class DnnlSoftmaxLayer<float>;
}; // namespace elsa
