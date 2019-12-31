#include "DnnlPooling.h"

namespace elsa
{
    template <typename data_t>
    DnnlPooling<data_t>::DnnlPooling(const DataDescriptor& inputDescriptor,
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
    void DnnlPooling<data_t>::compile()
    {
        auto desc = dnnl::pooling_forward::desc(
            dnnl::prop_kind::forward, dnnl::algorithm::pooling_max, _srcMemory->get_desc(),
            _dstMemoryDescriptor, _poolingStride, _poolingWindow, _poolingPadding, _poolingPadding);

        _forwardPrimitiveDescriptor = dnnl::pooling_forward::primitive_desc(desc, *_engine);

        _workspaceMemory = dnnl::memory(_forwardPrimitiveDescriptor.workspace_desc(), *_engine);

        _dstMemory = dnnl::memory(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        // TODO: Insert reorder
        _forwardPrimitives.push_back(dnnl::pooling_forward(_forwardPrimitiveDescriptor));
        _forwardArguments.push_back({{DNNL_ARG_SRC, *_srcMemory},
                                     {DNNL_ARG_WORKSPACE, _workspaceMemory},
                                     {DNNL_ARG_DST, _dstMemory}});
    }

    template class DnnlPooling<float>;

} // namespace elsa