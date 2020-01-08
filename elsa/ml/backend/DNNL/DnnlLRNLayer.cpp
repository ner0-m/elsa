#include "DnnlLRNLayer.h"

namespace elsa
{
    template <typename data_t>
    DnnlLRNLayer<data_t>::DnnlLRNLayer(const DataDescriptor& inputDescriptor,
                                       const DataDescriptor& outputDescriptor, index_t localSize,
                                       data_t alpha, data_t beta, data_t k)
        : DnnlLayer<data_t>(inputDescriptor, outputDescriptor), _alpha(alpha), _beta(beta), _k(k)
    {
        _localSize = static_cast<dnnl::memory::dim>(localSize);
    }

    template <typename data_t>
    void DnnlLRNLayer<data_t>::compileForwardStream()
    {
        auto desc = dnnl::lrn_forward::desc(
            /* Propagation kind */ dnnl::prop_kind::forward,
            /* LRN algorithm */ dnnl::algorithm::lrn_across_channels,
            /* Source memory descirptor */ _srcMemory->get_desc(),
            /* Local size to regularize */ _localSize,
            /* Parameters */ _alpha, _beta, _k);

        _forwardPrimitiveDescriptor = dnnl::lrn_forward::primitive_desc(desc, *_engine);

        // Set forward primitive
        _forwardPrimitives.push_back(dnnl::lrn_forward(_forwardPrimitiveDescriptor));

        _workspaceMemory = dnnl::memory(_forwardPrimitiveDescriptor.workspace_desc(), *_engine);

        _dstMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardArguments.push_back({{DNNL_ARG_SRC, *_srcMemory},
                                     {DNNL_ARG_DST, *_dstMemory},
                                     {DNNL_ARG_WORKSPACE, _workspaceMemory}});
    }

    template class DnnlLRNLayer<float>;
}; // namespace elsa
