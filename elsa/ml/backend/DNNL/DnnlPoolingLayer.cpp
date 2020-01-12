#include "DnnlPoolingLayer.h"

namespace elsa
{
    template <typename data_t>
    DnnlPoolingLayer<data_t>::DnnlPoolingLayer(const DataDescriptor& inputDescriptor,
                                               const DataDescriptor& outputDescriptor,
                                               const IndexVector_t& poolingWindow,
                                               const IndexVector_t& poolingStride)
        : DnnlLayer<data_t>(inputDescriptor, outputDescriptor)
    {
        for (const auto& dim : poolingWindow) {
            _poolingWindow.push_back(dim);
            _poolingPadding.push_back(0);
        }
        for (const auto& dim : poolingStride)
            _poolingStride.push_back(dim);
    }

    template <typename data_t>
    void DnnlPoolingLayer<data_t>::compileForwardStream()
    {
        auto desc = dnnl::pooling_forward::desc(
            /* Propagation kind */ dnnl::prop_kind::forward,
            /* Pooling algorithm */ dnnl::algorithm::pooling_max,
            /* Source memory descriptor */ _input.descriptor,
            /* Destination memory descriptor */ _output.descriptor,
            /* Pooling strides */ _poolingStride,
            /* Pooling window */ _poolingWindow,
            /* Input padding for lower dims */ _poolingPadding,
            /* Input padding for higher dims */ _poolingPadding);

        _forwardPrimitiveDescriptor = dnnl::pooling_forward::primitive_desc(desc, *_engine);

        _workspaceMemory.effectiveMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.workspace_desc(), *_engine);

        _output.effectiveMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        // TODO: Insert reorder
        _forwardStream.primitives.push_back(dnnl::pooling_forward(_forwardPrimitiveDescriptor));
        _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.effectiveMemory},
                                            {DNNL_ARG_WORKSPACE, *_workspaceMemory.effectiveMemory},
                                            {DNNL_ARG_DST, *_output.effectiveMemory}});
    }

    template class DnnlPoolingLayer<float>;

} // namespace elsa