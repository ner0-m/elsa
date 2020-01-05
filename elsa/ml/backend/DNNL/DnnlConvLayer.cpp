#include "DnnlConvLayer.h"

namespace elsa
{
    template <typename data_t>
    DnnlConvLayer<data_t>::DnnlConvLayer(const DataDescriptor& inputDescriptor,
                                         const DataDescriptor& outputDescriptor,
                                         const DataDescriptor& weightsDescriptor,
                                         const IndexVector_t& strideVector,
                                         const IndexVector_t& paddingVector,
                                         Initializer initializer)
        : DnnlTrainableLayer<data_t>(inputDescriptor, outputDescriptor, weightsDescriptor,
                                     initializer)
    {
        for (const auto& dim : strideVector)
            _strideDimensions.push_back(dim);

        for (const auto& dim : paddingVector)
            _paddingDimensions.push_back(dim);
    }

    template <typename data_t>
    void DnnlConvLayer<data_t>::compile()
    {
        BaseType::compile();
        auto desc = dnnl::convolution_forward::desc(
            /* Propagation kind */ dnnl::prop_kind::forward_training,
            /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
            /* Source */ _srcMemoryDescriptor,
            /* Weights */ _weightsMemoryDescriptor,
            /* Bias */ _biasMemoryDescriptor,
            /* Destination */ _dstMemoryDescriptor,
            /* Strides for spatial dims */ _strideDimensions,
            /* Dilation for spatial dims */ {0, 0},
            /* Lower padding for spatial dims */ _paddingDimensions,
            /* Higher padding for spatial dims */ _paddingDimensions);

        _forwardPrimitiveDescriptor = dnnl::convolution_forward::primitive_desc(desc, *_engine);

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

        _dstMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardPrimitives.push_back(dnnl::convolution_forward(_forwardPrimitiveDescriptor));

        _forwardArguments.push_back({{DNNL_ARG_SRC, _reorderedSrcMemory},
                                     {DNNL_ARG_WEIGHTS, _reorderedWeightsMemory},
                                     {DNNL_ARG_BIAS, _biasMemory},
                                     {DNNL_ARG_DST, *_dstMemory}});
    }

    template class DnnlConvLayer<float>;
} // namespace elsa
