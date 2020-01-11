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
        // for (const auto& dim : strideVector)
        //     _strideDimensions.push_back(dim);

        // for (const auto& dim : paddingVector)
        //     _paddingDimensions.push_back(dim);
    }

    template <typename data_t>
    void DnnlConvLayer<data_t>::compileForwardStream()
    {
        // BaseType::compileForwardStream();
        // auto desc = dnnl::convolution_forward::desc(
        //     /* Propagation kind */ dnnl::prop_kind::forward_training,
        //     /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
        //     /* Source */ _srcMemoryDescriptor,
        //     /* Weights */ _weightsMemoryDescriptor,
        //     /* Bias */ _biasMemoryDescriptor,
        //     /* Destination */ _dstMemoryDescriptor,
        //     /* Strides for spatial dims */ _strideDimensions,
        //     /* Dilation for spatial dims */ {0, 0},
        //     /* Lower padding for spatial dims */ _paddingDimensions,
        //     /* Higher padding for spatial dims */ _paddingDimensions);

        // _forwardPrimitiveDescriptor = dnnl::convolution_forward::primitive_desc(desc, *_engine);

        // // Do we need to reorder?
        // _reorderedSrcMemory = *_srcMemory;
        // if (_forwardPrimitiveDescriptor.src_desc() != _srcMemory->get_desc()) {
        //     _hasReorderedMemory = true;
        //     _reorderedSrcMemory = dnnl::memory(_forwardPrimitiveDescriptor.src_desc(), *_engine);
        //     _forwardPrimitives.push_back(dnnl::reorder(*_srcMemory, _reorderedSrcMemory));
        //     _forwardArguments.push_back(
        //         {{DNNL_ARG_FROM, *_srcMemory}, {DNNL_ARG_TO, _reorderedSrcMemory}});
        // }

        // _reorderedWeightsMemory = _weightsMemory;
        // if (_forwardPrimitiveDescriptor.weights_desc() != _weightsMemory.get_desc()) {
        //     _hasReorderedMemory = true;
        //     _reorderedWeightsMemory =
        //         dnnl::memory(_forwardPrimitiveDescriptor.weights_desc(), *_engine);
        //     _forwardPrimitives.push_back(dnnl::reorder(_weightsMemory, _reorderedWeightsMemory));
        //     _forwardArguments.push_back(
        //         {{DNNL_ARG_FROM, _weightsMemory}, {DNNL_ARG_TO, _reorderedWeightsMemory}});
        // }

        // _dstMemory =
        //     std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        // _forwardPrimitives.push_back(dnnl::convolution_forward(_forwardPrimitiveDescriptor));

        // _forwardArguments.push_back({{DNNL_ARG_SRC, _reorderedSrcMemory},
        //                              {DNNL_ARG_WEIGHTS, _reorderedWeightsMemory},
        //                              {DNNL_ARG_BIAS, _biasMemory},
        //                              {DNNL_ARG_DST, *_dstMemory}});
    }

    template <typename data_t>
    void DnnlConvLayer<data_t>::compileBackwardStream()
    {
        // BaseType::compileBackwardStream();

        // // Back propagate weights

        // // Backward descriptor for weights backprop
        // auto weightsBackDesc = dnnl::convolution_backward_weights::desc(
        //     dnnl::algorithm::convolution_auto, _gradientSrcMemoryDescriptor,
        //     _gradientWeightsMemoryDescriptor, _gradientBiasMemoryDescriptor,
        //     _gradientDstMemoryDescriptor, _strideDimensions, _paddingDimensions,
        //     _paddingDimensions);

        // // Backward primitive descriptor for weights backprop
        // _backwardWeightsPrimitiveDescriptor = dnnl::convolution_backward_weights::primitive_desc(
        //     weightsBackDesc, *_engine, _forwardPrimitiveDescriptor);

        // // Do we need reorder for gradient src memory?
        // _gradientSrcMemory = *_srcMemory;
        // if (_backwardWeightsPrimitiveDescriptor.src_desc() != _gradientSrcMemory.get_desc()) {
        //     _gradientSrcMemory =
        //         dnnl::memory(_backwardWeightsPrimitiveDescriptor.src_desc(), *_engine);
        //     _backwardPrimitives.push_back(dnnl::reorder(*_srcMemory, _gradientSrcMemory));
        //     _backwardArguments.push_back(
        //         {{DNNL_ARG_FROM, *_srcMemory}, {DNNL_ARG_TO, _gradientSrcMemory}});
        // }

        // // Do we need reorder for gradient dst memory?
        // _reorderedGradientDstMemory = *_gradientDstMemory;

        // if (_reorderedGradientDstMemory.get_desc()
        //     != _backwardWeightsPrimitiveDescriptor.diff_dst_desc()) {
        //     _reorderedGradientDstMemory =
        //         dnnl::memory(_backwardWeightsPrimitiveDescriptor.diff_dst_desc(), *_engine);
        //     _backwardPrimitives.push_back(
        //         dnnl::reorder(*_gradientDstMemory, _reorderedGradientDstMemory));
        //     _backwardArguments.push_back(
        //         {{DNNL_ARG_FROM, *_gradientDstMemory}, {DNNL_ARG_TO, _reorderedGradientDstMemory}});
        // }

        // _backwardPrimitives.push_back(
        //     dnnl::convolution_backward_weights(_backwardWeightsPrimitiveDescriptor));
        // _backwardArguments.push_back({{DNNL_ARG_SRC, _gradientSrcMemory},
        //                               {DNNL_ARG_DIFF_DST, _reorderedGradientDstMemory},
        //                               {DNNL_ARG_DIFF_BIAS, _gradientBiasMemory}});

        // // Do we need reorder for gradient weights memory?
        // _reorderedGradientWeightsMemory = _gradientWeightsMemory;
        // if (_backwardWeightsPrimitiveDescriptor.diff_weights_desc()
        //     != _gradientWeightsMemory.get_desc()) {
        //     _reorderedGradientWeightsMemory =
        //         dnnl::memory(_backwardWeightsPrimitiveDescriptor.diff_weights_desc(), *_engine);
        //     _backwardArguments.back().insert(
        //         {DNNL_ARG_DIFF_WEIGHTS, _reorderedGradientWeightsMemory});

        //     _backwardPrimitives.push_back(
        //         dnnl::reorder(_reorderedGradientWeightsMemory, _gradientWeightsMemory));
        //     _backwardArguments.push_back({{DNNL_ARG_FROM, _reorderedGradientWeightsMemory},
        //                                   {DNNL_ARG_TO, _gradientWeightsMemory}});
        // } else {
        //     _backwardArguments.back().insert({DNNL_ARG_DIFF_WEIGHTS, _gradientWeightsMemory});
        // }
    }

    template class DnnlConvLayer<float>;
} // namespace elsa
