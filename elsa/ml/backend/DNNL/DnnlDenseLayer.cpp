#include "DnnlDenseLayer.h"

namespace elsa
{
    template <typename data_t>
    DnnlDenseLayer<data_t>::DnnlDenseLayer(const DataDescriptor& inputDescriptor,
                                           const DataDescriptor& outputDescriptor,
                                           const DataDescriptor& weightsDescriptor,
                                           Initializer initializer)
        : DnnlTrainableLayer<data_t>(inputDescriptor, outputDescriptor, weightsDescriptor,
                                     initializer)
    {
    }

    template <typename data_t>
    void DnnlDenseLayer<data_t>::compileForwardStream()
    {
        BaseType::compileForwardStream();
        auto desc = dnnl::inner_product_forward::desc(
            /* Propagation kind */ dnnl::prop_kind::forward_training,
            /* Source descriptor*/ _srcMemoryDescriptor,
            /* Weights memory descriptor*/ _weightsMemoryDescriptor,
            /* Bias memory descriptor*/ _biasMemoryDescriptor,
            /* Destination memorydescriptor*/ _dstMemoryDescriptor);

        _forwardPrimitiveDescriptor = dnnl::inner_product_forward::primitive_desc(desc, *_engine);

        // Do we need to reorder? This is the case of the memory description chosen by the primitive
        // differs from the memory description we provided based on the layer's input descriptor /
        // weights descriptor.
        _reorderedSrcMemory = *_srcMemory;
        if (_forwardPrimitiveDescriptor.src_desc() != _srcMemory->get_desc()) {
            // Remember reordering to allow reverting it during output
            _hasReorderedMemory = true;
            _reorderedSrcMemory = dnnl::memory(_forwardPrimitiveDescriptor.src_desc(), *_engine);
            _forwardPrimitives.push_back(dnnl::reorder(*_srcMemory, _reorderedSrcMemory));
            _forwardArguments.push_back(
                {{DNNL_ARG_FROM, *_srcMemory}, {DNNL_ARG_TO, _reorderedSrcMemory}});
        }

        _reorderedWeightsMemory = _weightsMemory;
        if (_forwardPrimitiveDescriptor.weights_desc() != _weightsMemory.get_desc()) {
            // Remember reordering to allow reverting it during output
            _hasReorderedMemory = true;
            _reorderedWeightsMemory =
                dnnl::memory(_forwardPrimitiveDescriptor.weights_desc(), *_engine);
            _forwardPrimitives.push_back(dnnl::reorder(_weightsMemory, _reorderedWeightsMemory));
            _forwardArguments.push_back(
                {{DNNL_ARG_FROM, _weightsMemory}, {DNNL_ARG_TO, _reorderedWeightsMemory}});
        }

        _dstMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardPrimitives.push_back(dnnl::inner_product_forward(_forwardPrimitiveDescriptor));

        _forwardArguments.push_back({{DNNL_ARG_SRC, _reorderedSrcMemory},
                                     {DNNL_ARG_WEIGHTS, _reorderedWeightsMemory},
                                     {DNNL_ARG_BIAS, _biasMemory},
                                     {DNNL_ARG_DST, *_dstMemory}});
    }

    template class DnnlDenseLayer<float>;
} // namespace elsa
