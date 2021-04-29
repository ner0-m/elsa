#include "DnnlPoolingLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlPoolingLayer<data_t>::DnnlPoolingLayer(const VolumeDescriptor& inputDescriptor,
                                                   const VolumeDescriptor& outputDescriptor,
                                                   const IndexVector_t& poolingWindow,
                                                   const IndexVector_t& poolingStride)
            : DnnlLayer<data_t>(inputDescriptor, outputDescriptor, "DnnlPoolingLayer")
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
            BaseType::compileForwardStream();
            auto desc = dnnl::pooling_forward::desc(
                /* Propagation kind */ dnnl::prop_kind::forward,
                /* Pooling algorithm */ dnnl::algorithm::pooling_max,
                /* Source memory descriptor */ _input.front().descriptor,
                /* Destination memory descriptor */ _output.descriptor,
                /* Pooling strides */ _poolingStride,
                /* Pooling window */ _poolingWindow,
                /* Input padding for lower dims */ _poolingPadding,
                /* Input padding for higher dims */ _poolingPadding);

            _forwardPrimitiveDescriptor = dnnl::pooling_forward::primitive_desc(desc, *_engine);

            _workspaceMemory.effectiveMemory = std::make_shared<dnnl::memory>(
                _forwardPrimitiveDescriptor.workspace_desc(), *_engine);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                       dnnl::pooling_forward(_forwardPrimitiveDescriptor));
            _forwardStream.arguments.push_back(
                {{DNNL_ARG_SRC, *_input.front().effectiveMemory},
                 {DNNL_ARG_WORKSPACE, *_workspaceMemory.effectiveMemory}});

            auto outDesc = dnnl::memory::desc({_output.dimensions}, _typeTag, _output.formatTag);

            _output.describedMemory = std::make_shared<dnnl::memory>(outDesc, *_engine);

            _output.effectiveMemory = _output.describedMemory;
            if (_forwardPrimitiveDescriptor.dst_desc() != _output.describedMemory->get_desc()) {
                _output.wasReordered = true;
                _output.describedMemory = std::make_shared<dnnl::memory>(
                    _forwardPrimitiveDescriptor.dst_desc(), *_engine);
                _forwardStream.arguments.back().insert({DNNL_ARG_DST, *_output.describedMemory});
                ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream, dnnl::reorder(*_output.describedMemory,
                                                                         *_output.effectiveMemory));
                _forwardStream.arguments.push_back({{DNNL_ARG_FROM, *_output.describedMemory},
                                                    {DNNL_ARG_TO, *_output.effectiveMemory}});
            } else {
                _forwardStream.arguments.back().insert({DNNL_ARG_DST, *_output.effectiveMemory});
            }
        }

        template <typename data_t>
        void DnnlPoolingLayer<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();
            auto desc = dnnl::pooling_backward::desc(
                /* Pooling algorithm */ dnnl::algorithm::pooling_max,
                /* Input gradient descriptor */ _inputGradient.front().descriptor,
                /* Output gradient descriptor */ _outputGradient.front().descriptor,
                /* Strides */ _poolingStride,
                /* Pooling window */ _poolingWindow,
                /* Padding */ _poolingPadding, _poolingPadding);

            _backwardPrimitiveDescriptor =
                dnnl::pooling_backward::primitive_desc(desc, *_engine, _forwardPrimitiveDescriptor);

            this->reorderMemory(_backwardPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            _inputGradient.front().effectiveMemory = std::make_shared<dnnl::memory>(
                _backwardPrimitiveDescriptor.diff_src_desc(), *_engine);

            BaseType::validateDnnlMemory(_outputGradient.front().effectiveMemory,
                                         _inputGradient.front().effectiveMemory,
                                         _workspaceMemory.effectiveMemory);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream,
                                       dnnl::pooling_backward(_backwardPrimitiveDescriptor));
            _backwardStream.arguments.push_back(
                {{DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory},
                 {DNNL_ARG_DIFF_SRC, *_inputGradient.front().effectiveMemory},
                 {DNNL_ARG_WORKSPACE, *_workspaceMemory.effectiveMemory}});
        }

        template class DnnlPoolingLayer<float>;

    } // namespace detail
} // namespace elsa::ml