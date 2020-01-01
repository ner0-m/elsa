#include "DnnlDense.h"

namespace elsa
{
    template <typename data_t>
    DnnlDense<data_t>::DnnlDense(const DataDescriptor& inputDescriptor,
                                 const DataDescriptor& outputDescriptor,
                                 const DataDescriptor& weightsDescriptor)
        : DnnlTrainableLayer<data_t>(inputDescriptor, outputDescriptor, weightsDescriptor)
    {
    }

    template <typename data_t>
    void DnnlDense<data_t>::compile()
    {
        BaseType::compile();

        auto desc = dnnl::inner_product_forward::desc(
            /* Propagation kind */ dnnl::prop_kind::forward_inference,
            /* Source */ _srcMemoryDescriptor,
            /* Weights */ _weightsMemoryDescriptor,
            /* Bias */ _biasMemoryDescriptor,
            /* Destination */ _dstMemoryDescriptor);

        _forwardPrimitiveDescriptor = dnnl::inner_product_forward::primitive_desc(desc, *_engine);

        // Do we need to reorder?
        _reorderedSrcMemory = *_srcMemory;
        if (_forwardPrimitiveDescriptor.src_desc() != _srcMemory->get_desc()) {
            _hasReorderedMemory = true;
            _reorderedSrcMemory = dnnl::memory(_forwardPrimitiveDescriptor.src_desc(), *_engine);
            _forwardPrimitives.push_back(dnnl::reorder(*_srcMemory, _reorderedSrcMemory));
            _forwardArguments.push_back(
                {{DNNL_ARG_FROM, *_srcMemory}, {DNNL_ARG_TO, _reorderedSrcMemory}});
        }

        _reorderedWeightsMemory = _weightsMemory;
        if (_forwardPrimitiveDescriptor.weights_desc() != _weightsMemory.get_desc()) {
            _hasReorderedMemory = true;
            _reorderedWeightsMemory =
                dnnl::memory(_forwardPrimitiveDescriptor.weights_desc(), *_engine);
            _forwardPrimitives.push_back(dnnl::reorder(_weightsMemory, _reorderedWeightsMemory));
            _forwardArguments.push_back(
                {{DNNL_ARG_FROM, _weightsMemory}, {DNNL_ARG_TO, _reorderedWeightsMemory}});
        }

        _dstMemory = dnnl::memory(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardPrimitives.push_back(dnnl::inner_product_forward(_forwardPrimitiveDescriptor));

        _forwardArguments.push_back({{DNNL_ARG_SRC, _reorderedSrcMemory},
                                     {DNNL_ARG_WEIGHTS, _reorderedWeightsMemory},
                                     {DNNL_ARG_BIAS, _biasMemory},
                                     {DNNL_ARG_DST, _dstMemory}});
    }

    template class DnnlDense<float>;
} // namespace elsa
