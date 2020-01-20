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
            /* Input memory descirptor */ _input.descriptor,
            /* Local size to regularize */ _localSize,
            /* Parameters */ _alpha, _beta, _k);

        _forwardPrimitiveDescriptor = dnnl::lrn_forward::primitive_desc(desc, *_engine);

        // Set forward primitive
        _forwardStream.primitives.push_back(dnnl::lrn_forward(_forwardPrimitiveDescriptor));

        _workspace.effectiveMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.workspace_desc(), *_engine);

        _output.effectiveMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.effectiveMemory},
                                            {DNNL_ARG_DST, *_output.effectiveMemory},
                                            {DNNL_ARG_WORKSPACE, *_workspace.effectiveMemory}});

        _forwardStream.isCompiled = true;
    }

    template <typename data_t>
    void DnnlLRNLayer<data_t>::compileBackwardStream()
    {
        auto desc = dnnl::lrn_backward::desc(
            /* LRN algorithm */ dnnl::algorithm::lrn_across_channels,
            /* Input memory descriptor */ _input.descriptor,
            /* Output gradient memory descriptor */ _outputGradient.descriptor,
            /* Local size to regularize */ _localSize,
            /* Parameters */ _alpha, _beta, _k);

        _backwardPrimitiveDescriptor =
            dnnl::lrn_backward::primitive_desc(desc, *_engine, _forwardPrimitiveDescriptor);

        _inputGradient.effectiveMemory =
            std::make_shared<dnnl::memory>(_backwardPrimitiveDescriptor.diff_src_desc(), *_engine);

        this->reorderMemory(_backwardPrimitiveDescriptor.diff_dst_desc(), _outputGradient,
                            _backwardStream);

        // Set forward primitive
        _backwardStream.primitives.push_back(dnnl::lrn_backward(_backwardPrimitiveDescriptor));
        _backwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.effectiveMemory},
                                             {DNNL_ARG_DIFF_DST, *_outputGradient.effectiveMemory},
                                             {DNNL_ARG_DIFF_SRC, *_inputGradient.effectiveMemory},
                                             {DNNL_ARG_WORKSPACE, *_workspace.effectiveMemory}});

        _backwardStream.isCompiled = true;
    }

    template class DnnlLRNLayer<float>;
} // namespace elsa
