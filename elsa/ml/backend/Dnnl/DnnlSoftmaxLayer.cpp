#include "DnnlSoftmaxLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlSoftmaxLayer<data_t>::DnnlSoftmaxLayer(const VolumeDescriptor& inputDescriptor,
                                                   const VolumeDescriptor& outputDescriptor)
            : DnnlLayer<data_t>(inputDescriptor, outputDescriptor, "DnnlSoftmaxLayer")
        {
            _softmaxAxis = (inputDescriptor.getNumberOfDimensions() == 1) ? 0 : 1;
        }

        template <typename data_t>
        void DnnlSoftmaxLayer<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();

            auto desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training,
                                                    _input.front().descriptor, _softmaxAxis);

            _forwardPrimitiveDescriptor = dnnl::softmax_forward::primitive_desc(desc, *_engine);

            // Set forward primitive
            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                       dnnl::softmax_forward(_forwardPrimitiveDescriptor));

            _output.effectiveMemory =
                std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

            BaseType::validateDnnlMemory(_input.front().effectiveMemory, _output.effectiveMemory);

            _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.front().effectiveMemory},
                                                {DNNL_ARG_DST, *_output.effectiveMemory}});
        }

        template <typename data_t>
        void DnnlSoftmaxLayer<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();

            auto desc = dnnl::softmax_backward::desc(
                /* Output gradient descriptor */ _outputGradient.front().descriptor,
                /* Input descriptor */ _input.front().descriptor,
                /* Softmax axis */ _softmaxAxis);

            _backwardPrimitiveDescriptor =
                dnnl::softmax_backward::primitive_desc(desc, *_engine, _forwardPrimitiveDescriptor);

            // Reorder if necessary
            this->reorderMemory(_backwardPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            _inputGradient.front().effectiveMemory = std::make_shared<dnnl::memory>(
                _backwardPrimitiveDescriptor.diff_src_desc(), *_engine);

            BaseType::validateDnnlMemory(_input.front().effectiveMemory, _output.effectiveMemory,
                                         _outputGradient.front().effectiveMemory,
                                         _inputGradient.front().effectiveMemory);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream,
                                       dnnl::softmax_backward(_backwardPrimitiveDescriptor));
            _backwardStream.arguments.push_back(
                {{DNNL_ARG_DST, *_output.effectiveMemory},
                 {DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory},
                 {DNNL_ARG_DIFF_SRC, *_inputGradient.front().effectiveMemory}});
        }

        template class DnnlSoftmaxLayer<float>;
    } // namespace detail
} // namespace elsa::ml
