#include "DnnlConvolution.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlConvolution<data_t>::DnnlConvolution(const VolumeDescriptor& inputDescriptor,
                                                 const VolumeDescriptor& outputDescriptor,
                                                 const VolumeDescriptor& weightsDescriptor,
                                                 const IndexVector_t& strides,
                                                 const IndexVector_t& paddingLow,
                                                 const IndexVector_t& paddingHigh,
                                                 Initializer initializer)
            : DnnlTrainableLayer<data_t>(inputDescriptor, outputDescriptor, weightsDescriptor,
                                         initializer)
        {
            for (const auto& dim : strides)
                _stridesDimensions.push_back(dim);

            for (const auto& dim : paddingLow)
                _paddingLowDimensions.push_back(dim);

            for (const auto& dim : paddingHigh)
                _paddingHighDimensions.push_back(dim);
        }

        template <typename data_t>
        void DnnlConvolution<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();

            // TODO(tellenbach): Add support for dilated convolution
            auto desc = dnnl::convolution_forward::desc(
                /* Propagation kind */ dnnl::prop_kind::forward_training,
                /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
                /* Input descriptor */ _input.front().descriptor,
                /* Weights descriptor */ _weights.descriptor,
                /* Bias descriptor*/ _bias.descriptor,
                /* Output descriptor */ _output.descriptor,
                /* Strides for spatial dims */ _stridesDimensions,
                /* Lower padding for spatial dims */ _paddingLowDimensions,
                /* Higher padding for spatial dims */ _paddingHighDimensions);

            _forwardPrimitiveDescriptor = dnnl::convolution_forward::primitive_desc(desc, *_engine);

            // Do we need to reorder?
            this->reorderMemory(_forwardPrimitiveDescriptor.src_desc(), _input.front(),
                                _forwardStream);
            this->reorderMemory(_forwardPrimitiveDescriptor.weights_desc(), _weights,
                                _forwardStream);

            _output.describedMemory =
                std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                       dnnl::convolution_forward(_forwardPrimitiveDescriptor));

            BaseType::validateDnnlMemory(_input.front().effectiveMemory, _weights.effectiveMemory,
                                         _bias.effectiveMemory, _output.describedMemory);

            _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.front().effectiveMemory},
                                                {DNNL_ARG_WEIGHTS, *_weights.effectiveMemory},
                                                {DNNL_ARG_BIAS, *_bias.effectiveMemory},
                                                {DNNL_ARG_DST, *_output.describedMemory}});

            // If either the input or weights have been reordered there could potential reordering
            // for output
            _output.effectiveMemory = _output.describedMemory;
            if (_input.front().wasReordered || _weights.wasReordered) {
                _output.wasReordered = true;
                _output.effectiveMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc({{_output.dimensions}, _typeTag, _output.formatTag}),
                    *_engine);
                ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream, dnnl::reorder(*_output.describedMemory,
                                                                         *_output.effectiveMemory));
                _forwardStream.arguments.push_back({{DNNL_ARG_FROM, *_output.describedMemory},
                                                    {DNNL_ARG_TO, *_output.effectiveMemory}});
            }
        }

        template <typename data_t>
        void DnnlConvolution<data_t>::compileBackwardDataStream()
        {
            auto desc = dnnl::convolution_backward_data::desc(
                /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
                /* Input Gradient descriptor */ _inputGradient.front().descriptor,
                /* Weights descriptor */ _weights.descriptor,
                /* Output gradient descriptor */ _outputGradient.front().descriptor,
                /* Strides */ _stridesDimensions,
                /* Padding */ _paddingLowDimensions, _paddingHighDimensions);

            _backwardDataPrimitiveDescriptor = dnnl::convolution_backward_data::primitive_desc(
                desc, *_engine, _forwardPrimitiveDescriptor);

            // Reorder output gradient of necessary
            this->reorderMemory(_backwardDataPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            // Reorder weights if necessary
            this->reorderMemory(_backwardDataPrimitiveDescriptor.weights_desc(), _weights,
                                _backwardStream);

            // Set input gradient memory
            _inputGradient.front().describedMemory = std::make_shared<dnnl::memory>(
                _backwardDataPrimitiveDescriptor.diff_src_desc(), *_engine);

            // Push backward data primitive
            ELSA_ML_ADD_DNNL_PRIMITIVE(
                _backwardStream, dnnl::convolution_backward_data(_backwardDataPrimitiveDescriptor));

            BaseType::validateDnnlMemory(_inputGradient.front().describedMemory,
                                         _weights.effectiveMemory,
                                         _outputGradient.front().effectiveMemory);

            _backwardStream.arguments.push_back(
                {/*  Input gradient */ {DNNL_ARG_DIFF_SRC, *_inputGradient.front().describedMemory},
                 /*  Weights */ {DNNL_ARG_WEIGHTS, *_weights.effectiveMemory},
                 /*  Output gradient */
                 {DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory}});
        }

        template <typename data_t>
        void DnnlConvolution<data_t>::compileBackwardWeightsStream()
        {
            // Backward descriptor for weights backprop
            auto desc = dnnl::convolution_backward_weights::desc(
                /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
                /* Input gradient descriptor */ _input.front().descriptor,
                /* Weights gradient descriptor */ _weightsGradient.descriptor,
                /* Bias gradient descriptor */ _biasGradient.descriptor,
                /* Output gradient descriptor */ _outputGradient.front().descriptor,
                /* Strides */ _stridesDimensions,
                /* Padding */ _paddingLowDimensions, _paddingHighDimensions);

            _backwardWeightsPrimitiveDescriptor =
                dnnl::convolution_backward_weights::primitive_desc(desc, *_engine,
                                                                   _forwardPrimitiveDescriptor);

            // Do we need reorder for gradient src memory?
            this->reorderMemory(_backwardWeightsPrimitiveDescriptor.src_desc(), _input.front(),
                                _backwardStream);

            // Do we need to reorder gradient destination memory?
            this->reorderMemory(_backwardWeightsPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            BaseType::validateDnnlMemory(
                _input.front().effectiveMemory, _biasGradient.effectiveMemory,
                _outputGradient.front().effectiveMemory, _weightsGradient.describedMemory);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream, dnnl::convolution_backward_weights(
                                                            _backwardWeightsPrimitiveDescriptor));

            _backwardStream.arguments.push_back(
                {/* Input */
                 {DNNL_ARG_SRC, *_input.front().effectiveMemory},
                 {DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory},
                 /* Output */
                 {DNNL_ARG_DIFF_WEIGHTS, *_weightsGradient.describedMemory},
                 {DNNL_ARG_DIFF_BIAS, *_biasGradient.effectiveMemory}});

            _weightsGradient.effectiveMemory = _weightsGradient.describedMemory;

            _weightsGradient.wasReordered = true;
            _weightsGradient.effectiveMemory = std::make_shared<dnnl::memory>(
                dnnl::memory::desc(
                    {{_weightsGradient.dimensions}, _typeTag, _weightsGradient.formatTag}),
                *_engine);
            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                       dnnl::reorder(*_weightsGradient.describedMemory,
                                                     *_weightsGradient.effectiveMemory));
            _forwardStream.arguments.push_back({{DNNL_ARG_FROM, *_weightsGradient.describedMemory},
                                                {DNNL_ARG_TO, *_weightsGradient.effectiveMemory}});

            if (_weightsGradient.describedMemory->get_desc()
                != _backwardWeightsPrimitiveDescriptor.diff_weights_desc()) {
                _weightsGradient.wasReordered = true;
                _weightsGradient.describedMemory = std::make_shared<dnnl::memory>(
                    _backwardWeightsPrimitiveDescriptor.diff_weights_desc(), *_engine);
                _backwardStream.arguments.back().insert(
                    {DNNL_ARG_DIFF_WEIGHTS, *_weightsGradient.describedMemory});
                ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream,
                                           dnnl::reorder(*_weightsGradient.describedMemory,
                                                         *_weightsGradient.effectiveMemory));
                _backwardStream.arguments.push_back(
                    {{DNNL_ARG_FROM, *_weightsGradient.describedMemory},
                     {DNNL_ARG_TO, *_weightsGradient.effectiveMemory}});
            } else {
                _backwardStream.arguments.back().insert(
                    {DNNL_ARG_DIFF_WEIGHTS, *_weightsGradient.effectiveMemory});
            }
        }

        template <typename data_t>
        void DnnlConvolution<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();
            compileBackwardWeightsStream();
            compileBackwardDataStream();

            _inputGradient.front().effectiveMemory = _inputGradient.front().describedMemory;
            if (1) {
                _inputGradient.front().wasReordered = true;
                _inputGradient.front().effectiveMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc({{_inputGradient.front().dimensions},
                                        _typeTag,
                                        _inputGradient.front().formatTag}),
                    *_engine);
                ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                           dnnl::reorder(*_inputGradient.front().describedMemory,
                                                         *_inputGradient.front().effectiveMemory));
                _forwardStream.arguments.push_back(
                    {{DNNL_ARG_FROM, *_inputGradient.front().describedMemory},
                     {DNNL_ARG_TO, *_inputGradient.front().effectiveMemory}});
            }
        }

        template class DnnlConvolution<float>;

        template <typename data_t>
        DnnlDeconvolution<data_t>::DnnlDeconvolution(const VolumeDescriptor& inputDescriptor,
                                                     const VolumeDescriptor& outputDescriptor,
                                                     const VolumeDescriptor& weightsDescriptor,
                                                     const IndexVector_t& strides,
                                                     const IndexVector_t& paddingLow,
                                                     const IndexVector_t& paddingHigh,
                                                     Initializer initializer)
            : DnnlTrainableLayer<data_t>(inputDescriptor, outputDescriptor, weightsDescriptor,
                                         initializer)
        {
            for (const auto& dim : strides)
                _stridesDimensions.push_back(dim);

            for (const auto& dim : paddingLow)
                _paddingLowDimensions.push_back(dim);

            for (const auto& dim : paddingHigh)
                _paddingHighDimensions.push_back(dim);
        }

        template <typename data_t>
        void DnnlDeconvolution<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();

            // TODO(todo): Add support for dilated convolution, we currently assume dilation of 0
            auto desc = dnnl::deconvolution_forward::desc(
                /* Propagation kind */ dnnl::prop_kind::forward_training,
                /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
                /* Input descriptor */ _input.descriptor,
                /* Weights descriptor */ _weights.descriptor,
                /* Bias descriptor*/ _bias.descriptor,
                /* Output descriptor */ _output.descriptor,
                /* Strides for spatial dims */ _stridesDimensions,
                /* Dilation for spatial dims */ {0, 0},
                /* Lower padding for spatial dims */ _paddingLowDimensions,
                /* Higher padding for spatial dims */ _paddingHighDimensions);

            _forwardPrimitiveDescriptor =
                dnnl::deconvolution_forward::primitive_desc(desc, *_engine);

            // Do we need to reorder?
            this->reorderMemory(_forwardPrimitiveDescriptor.src_desc(), _input.front(),
                                _forwardStream);
            this->reorderMemory(_forwardPrimitiveDescriptor.weights_desc(), _weights,
                                _forwardStream);

            _output.describedMemory =
                std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                       dnnl::deconvolution_forward(_forwardPrimitiveDescriptor));

            BaseType::validateDnnlMemory(_input.front().effectiveMemory, _weights.effectiveMemory,
                                         _bias.effectiveMemory, _output.describedMemory);

            _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.front().effectiveMemory},
                                                {DNNL_ARG_WEIGHTS, *_weights.effectiveMemory},
                                                {DNNL_ARG_BIAS, *_bias.effectiveMemory},
                                                {DNNL_ARG_DST, *_output.describedMemory}});

            // If either the input or weights have been reordered there could potential reordering
            // for output
            _output.effectiveMemory = _output.describedMemory;
            if (_input.front().wasReordered || _weights.wasReordered) {
                _output.wasReordered = true;
                _output.effectiveMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc({{_output.dimensions}, _typeTag, _output.formatTag}),
                    *_engine);
                ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream, dnnl::reorder(*_output.describedMemory,
                                                                         *_output.effectiveMemory));
                _forwardStream.arguments.push_back({{DNNL_ARG_FROM, *_output.describedMemory},
                                                    {DNNL_ARG_TO, *_output.effectiveMemory}});
            }
        }

        template <typename data_t>
        void DnnlDeconvolution<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();
            compileBackwardWeightsStream();
            compileBackwardDataStream();

            _inputGradient.front().effectiveMemory = _inputGradient.front().describedMemory;
            if (1) {
                _inputGradient.front().wasReordered = true;
                _inputGradient.front().effectiveMemory = std::make_shared<dnnl::memory>(
                    dnnl::memory::desc({{_inputGradient.front().dimensions},
                                        _typeTag,
                                        _inputGradient.front().formatTag}),
                    *_engine);
                ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                           dnnl::reorder(*_inputGradient.front().describedMemory,
                                                         *_inputGradient.front().effectiveMemory));
                _forwardStream.arguments.push_back(
                    {{DNNL_ARG_FROM, *_inputGradient.front().describedMemory},
                     {DNNL_ARG_TO, *_inputGradient.front().effectiveMemory}});
            }
        }

        template <typename data_t>
        void DnnlDeconvolution<data_t>::compileBackwardDataStream()
        {
            auto desc = dnnl::deconvolution_backward_data::desc(
                /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
                /* Input Gradient descriptor */ _inputGradient.front().descriptor,
                /* Weights descriptor */ _weights.descriptor,
                /* Output gradient descriptor */ _outputGradient.front().descriptor,
                /* Strides */ _stridesDimensions,
                /* Padding */ _paddingLowDimensions, _paddingHighDimensions);

            _backwardDataPrimitiveDescriptor = dnnl::deconvolution_backward_data::primitive_desc(
                desc, *_engine, _forwardPrimitiveDescriptor);

            // Reorder output gradient of necessary
            this->reorderMemory(_backwardDataPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            // Reorder weights if necessary
            this->reorderMemory(_backwardDataPrimitiveDescriptor.weights_desc(), _weights,
                                _backwardStream);

            // Set input gradient memory
            _inputGradient.front().describedMemory = std::make_shared<dnnl::memory>(
                _backwardDataPrimitiveDescriptor.diff_src_desc(), *_engine);

            // Push backward data primitive
            ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream, dnnl::deconvolution_backward_data(
                                                            _backwardDataPrimitiveDescriptor));

            BaseType::validateDnnlMemory(_inputGradient.front().describedMemory,
                                         _weights.effectiveMemory,
                                         _outputGradient.front().effectiveMemory);

            _backwardStream.arguments.push_back(
                {/*  Input gradient */ {DNNL_ARG_DIFF_SRC, *_inputGradient.front().describedMemory},
                 /*  Weights */ {DNNL_ARG_WEIGHTS, *_weights.effectiveMemory},
                 /*  Output gradient */
                 {DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory}});
        }

        template <typename data_t>
        void DnnlDeconvolution<data_t>::compileBackwardWeightsStream()
        {
            // Backward descriptor for weights backprop
            auto desc = dnnl::deconvolution_backward_weights::desc(
                /* Convolution algorithm */ dnnl::algorithm::convolution_auto,
                /* Input gradient descriptor */ _input.front().descriptor,
                /* Weights gradient descriptor */ _weightsGradient.descriptor,
                /* Bias gradient descriptor */ _biasGradient.descriptor,
                /* Output gradient descriptor */ _outputGradient.front().descriptor,
                /* Strides */ _stridesDimensions,
                /* Padding */ _paddingLowDimensions, _paddingHighDimensions);

            _backwardWeightsPrimitiveDescriptor =
                dnnl::deconvolution_backward_weights::primitive_desc(desc, *_engine,
                                                                     _forwardPrimitiveDescriptor);

            // Do we need reorder for gradient src memory?
            this->reorderMemory(_backwardWeightsPrimitiveDescriptor.src_desc(), _input.front(),
                                _backwardStream);

            // Do we need to reorder gradient destination memory?
            this->reorderMemory(_backwardWeightsPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            BaseType::validateDnnlMemory(
                _input.front().effectiveMemory, _biasGradient.effectiveMemory,
                _outputGradient.front().effectiveMemory, _weightsGradient.describedMemory);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream, dnnl::deconvolution_backward_weights(
                                                            _backwardWeightsPrimitiveDescriptor));

            _backwardStream.arguments.push_back(
                {/* Input */
                 {DNNL_ARG_SRC, *_input.front().effectiveMemory},
                 {DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory},
                 /* Output */
                 {DNNL_ARG_DIFF_WEIGHTS, *_weightsGradient.describedMemory},
                 {DNNL_ARG_DIFF_BIAS, *_biasGradient.effectiveMemory}});

            _weightsGradient.effectiveMemory = _weightsGradient.describedMemory;

            _weightsGradient.wasReordered = true;
            _weightsGradient.effectiveMemory = std::make_shared<dnnl::memory>(
                dnnl::memory::desc(
                    {{_weightsGradient.dimensions}, _typeTag, _weightsGradient.formatTag}),
                *_engine);
            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                       dnnl::reorder(*_weightsGradient.describedMemory,
                                                     *_weightsGradient.effectiveMemory));
            _forwardStream.arguments.push_back({{DNNL_ARG_FROM, *_weightsGradient.describedMemory},
                                                {DNNL_ARG_TO, *_weightsGradient.effectiveMemory}});

            if (_weightsGradient.describedMemory->get_desc()
                != _backwardWeightsPrimitiveDescriptor.diff_weights_desc()) {
                _weightsGradient.wasReordered = true;
                _weightsGradient.describedMemory = std::make_shared<dnnl::memory>(
                    _backwardWeightsPrimitiveDescriptor.diff_weights_desc(), *_engine);
                _backwardStream.arguments.back().insert(
                    {DNNL_ARG_DIFF_WEIGHTS, *_weightsGradient.describedMemory});
                ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream,
                                           dnnl::reorder(*_weightsGradient.describedMemory,
                                                         *_weightsGradient.effectiveMemory));
                _backwardStream.arguments.push_back(
                    {{DNNL_ARG_FROM, *_weightsGradient.describedMemory},
                     {DNNL_ARG_TO, *_weightsGradient.effectiveMemory}});
            } else {
                _backwardStream.arguments.back().insert(
                    {DNNL_ARG_DIFF_WEIGHTS, *_weightsGradient.effectiveMemory});
            }
        }
    } // namespace detail
} // namespace elsa::ml