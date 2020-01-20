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

    static void validateDnnlMemory(std::shared_ptr<dnnl::memory> mem)
    {
        assert(mem && "Pointer to memory cannot be null");
        assert(mem->get_desc().get_size() && "Memory cannot have size 0,");
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

        validateDnnlMemory(_input.effectiveMemory);
        validateDnnlMemory(_output.effectiveMemory);

        _forwardStream.arguments.push_back(
            {{DNNL_ARG_SRC, *_input.effectiveMemory}, {DNNL_ARG_DST, *_output.effectiveMemory}});

        _forwardStream.isCompiled = true;
    }

    template <typename data_t>
    void DnnlSoftmaxLayer<data_t>::compileBackwardStream()
    {
        auto desc = dnnl::softmax_backward::desc(
            /* Output gradient descriptor */ _outputGradient.descriptor,
            /* Input descriptor */ _input.descriptor,
            /* Softmax axis */ _softmaxAxis);

        _backwardPrimitiveDescriptor =
            dnnl::softmax_backward::primitive_desc(desc, *_engine, _forwardPrimitiveDescriptor);

        // Reorder if necessary
        this->reorderMemory(_backwardPrimitiveDescriptor.diff_dst_desc(), _outputGradient,
                            _backwardStream);

        _inputGradient.effectiveMemory =
            std::make_shared<dnnl::memory>(_backwardPrimitiveDescriptor.diff_src_desc(), *_engine);

        validateDnnlMemory(_input.effectiveMemory);
        validateDnnlMemory(_output.effectiveMemory);
        validateDnnlMemory(_outputGradient.effectiveMemory);
        validateDnnlMemory(_inputGradient.effectiveMemory);

        _backwardStream.primitives.push_back(dnnl::softmax_backward(_backwardPrimitiveDescriptor));
        _backwardStream.arguments.push_back({{DNNL_ARG_DST, *_output.effectiveMemory},
                                             {DNNL_ARG_DIFF_DST, *_outputGradient.effectiveMemory},
                                             {DNNL_ARG_DIFF_SRC, *_inputGradient.effectiveMemory}});
        _backwardStream.isCompiled = true;
    }

    template class DnnlSoftmaxLayer<float>;
} // namespace elsa
