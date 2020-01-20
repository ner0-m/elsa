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

    static void validateDnnlMemory(std::shared_ptr<dnnl::memory> mem)
    {
        assert(mem && "Pointer to memory cannot be null");
        assert(mem->get_desc().get_size() && "Memory cannot have size 0,");
    }

    template <typename data_t>
    void DnnlConvLayer<data_t>::compileForwardStream()
    {
        BaseType::compileForwardStream();

        // TODO: Add support for dilated convolution
        auto desc = dnnl::convolution_forward::desc(
            /* Propagation kind */ dnnl::prop_kind::forward_training,
            /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
            /* Input descriptor */ _input.descriptor,
            /* Weights descriptor */ _weights.descriptor,
            /* Bias descriptor*/ _bias.descriptor,
            /* Output descriptor */ _output.descriptor,
            /* Strides for spatial dims */ _strideDimensions,
            /* Dilation for spatial dims */ {0, 0},
            /* Lower padding for spatial dims */ _paddingDimensions,
            /* Higher padding for spatial dims */ _paddingDimensions);

        _forwardPrimitiveDescriptor = dnnl::convolution_forward::primitive_desc(desc, *_engine);

        // Do we need to reorder?
        this->reorderMemory(_forwardPrimitiveDescriptor.src_desc(), _input, _forwardStream);
        this->reorderMemory(_forwardPrimitiveDescriptor.weights_desc(), _weights, _forwardStream);

        _output.effectiveMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        if (_input.wasReordered || _weights.wasReordered)
            _output.wasReordered = true;

        _forwardStream.primitives.push_back(dnnl::convolution_forward(_forwardPrimitiveDescriptor));

        validateDnnlMemory(_input.effectiveMemory);
        validateDnnlMemory(_weights.effectiveMemory);
        validateDnnlMemory(_bias.effectiveMemory);
        validateDnnlMemory(_output.effectiveMemory);

        _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.effectiveMemory},
                                            {DNNL_ARG_WEIGHTS, *_weights.effectiveMemory},
                                            {DNNL_ARG_BIAS, *_bias.effectiveMemory},
                                            {DNNL_ARG_DST, *_output.effectiveMemory}});
        _forwardStream.isCompiled = true;
    }

    template <typename data_t>
    void DnnlConvLayer<data_t>::compileBackwardDataStream()
    {
        auto desc = dnnl::convolution_backward_data::desc(
            /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
            /* Input Gradient descriptor */ _inputGradient.descriptor,
            /* Weights descriptor */ _weights.descriptor,
            /* Output gradient descriptor */ _outputGradient.descriptor,
            /* Strides */ _strideDimensions,
            /* Padding */ _paddingDimensions, _paddingDimensions);

        _backwardDataPrimitiveDescriptor = dnnl::convolution_backward_data::primitive_desc(
            desc, *_engine, _forwardPrimitiveDescriptor);

        // Reorder output gradient of necessary
        this->reorderMemory(_backwardDataPrimitiveDescriptor.diff_dst_desc(), _outputGradient,
                            _backwardStream);

        // Reorder weights if necessary
        this->reorderMemory(_backwardDataPrimitiveDescriptor.weights_desc(), _weights,
                            _backwardStream);

        // Set input gradient memory
        _inputGradient.effectiveMemory = std::make_shared<dnnl::memory>(
            _backwardDataPrimitiveDescriptor.diff_src_desc(), *_engine);

        // Push backward data primitive
        _backwardStream.primitives.push_back(
            dnnl::convolution_backward_data(_backwardDataPrimitiveDescriptor));

        validateDnnlMemory(_inputGradient.effectiveMemory);
        validateDnnlMemory(_weights.effectiveMemory);
        validateDnnlMemory(_outputGradient.effectiveMemory);

        _backwardStream.arguments.push_back(
            {/*  Input gradient */ {DNNL_ARG_DIFF_SRC, *_inputGradient.effectiveMemory},
             /*  Weights */ {DNNL_ARG_WEIGHTS, *_weights.effectiveMemory},
             /*  Output gradient */ {DNNL_ARG_DIFF_DST, *_outputGradient.effectiveMemory}});
    }

    template <typename data_t>
    void DnnlConvLayer<data_t>::compileBackwardWeightsStream()
    {
        // Backward descriptor for weights backprop
        // Backward descriptor for weights backprop
        auto desc = dnnl::convolution_backward_weights::desc(
            /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
            /* Input gradient descriptor */ _input.descriptor,
            /* Weights gradient descriptor */ _weightsGradient.descriptor,
            /* Bias gradient descriptor */ _biasGradient.descriptor,
            /* Output gradient descriptor */ _outputGradient.descriptor,
            /* Strides */ _strideDimensions,
            /* Padding */ _paddingDimensions, _paddingDimensions);

        _backwardWeightsPrimitiveDescriptor = dnnl::convolution_backward_weights::primitive_desc(
            desc, *_engine, _forwardPrimitiveDescriptor);

        // Do we need reorder for gradient src memory?
        this->reorderMemory(_backwardWeightsPrimitiveDescriptor.src_desc(), _input,
                            _backwardStream);

        this->reorderMemory(_backwardWeightsPrimitiveDescriptor.diff_dst_desc(), _outputGradient,
                            _backwardStream);

        validateDnnlMemory(_input.effectiveMemory);
        validateDnnlMemory(_biasGradient.effectiveMemory);
        validateDnnlMemory(_outputGradient.effectiveMemory);

        _backwardStream.primitives.push_back(
            dnnl::convolution_backward_weights(_backwardWeightsPrimitiveDescriptor));
        _backwardStream.arguments.push_back(
            {{DNNL_ARG_SRC, *_input.effectiveMemory},
             {DNNL_ARG_DIFF_BIAS, *_biasGradient.effectiveMemory},
             {DNNL_ARG_DIFF_DST, *_outputGradient.effectiveMemory}});

        _weightsGradient.effectiveMemory = _weightsGradient.describedMemory;
        validateDnnlMemory(_weightsGradient.effectiveMemory);
        if (_weightsGradient.describedMemory->get_desc()
            != _backwardWeightsPrimitiveDescriptor.diff_weights_desc()) {
            _weightsGradient.wasReordered = true;
            _weightsGradient.describedMemory = std::make_shared<dnnl::memory>(
                _backwardWeightsPrimitiveDescriptor.diff_weights_desc(), *_engine);
            _backwardStream.arguments.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, *_weightsGradient.describedMemory});
            _backwardStream.primitives.push_back(dnnl::reorder(*_weightsGradient.describedMemory,
                                                               *_weightsGradient.effectiveMemory));
            _backwardStream.arguments.push_back({{DNNL_ARG_FROM, *_weightsGradient.describedMemory},
                                                 {DNNL_ARG_TO, *_weightsGradient.effectiveMemory}});
        } else {
            _backwardStream.arguments.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, *_weightsGradient.effectiveMemory});
        }
    }

    template <typename data_t>
    void DnnlConvLayer<data_t>::compileBackwardStream()
    {
        BaseType::compileBackwardStream();
        compileBackwardWeightsStream();
        compileBackwardDataStream();
        _backwardStream.isCompiled = true;
    }

    template class DnnlConvLayer<float>;
} // namespace elsa
