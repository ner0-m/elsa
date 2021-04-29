#include "DnnlDenseLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlDenseLayer<data_t>::DnnlDenseLayer(const VolumeDescriptor& inputDescriptor,
                                               const VolumeDescriptor& outputDescriptor,
                                               const VolumeDescriptor& weightsDescriptor,
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
                /* Source descriptor*/ _input.front().descriptor,
                /* Weights memory descriptor*/ _weights.descriptor,
                /* Bias memory descriptor*/ _bias.descriptor,
                /* Destination memorydescriptor*/ _output.descriptor);

            // Create inner-product forward primitive
            _forwardPrimitiveDescriptor =
                dnnl::inner_product_forward::primitive_desc(desc, *_engine);

            // Do we need to reorder? This is the case of the memory description chosen by the
            // primitive differs from the memory description we provided based on the layer's input
            // descriptor / weights descriptor.
            this->reorderMemory(_forwardPrimitiveDescriptor.src_desc(), _input.front(),
                                _forwardStream);
            this->reorderMemory(_forwardPrimitiveDescriptor.weights_desc(), _weights,
                                _forwardStream);

            // Allocate output-demmory
            _output.effectiveMemory =
                std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

            // Add inner-product primitive to forward-stream
            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                       dnnl::inner_product_forward(_forwardPrimitiveDescriptor));

            // Validate memory
            BaseType::validateDnnlMemory(_input.front().effectiveMemory, _weights.effectiveMemory,
                                         _bias.effectiveMemory, _output.effectiveMemory);

            _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.front().effectiveMemory},
                                                {DNNL_ARG_WEIGHTS, *_weights.effectiveMemory},
                                                {DNNL_ARG_BIAS, *_bias.effectiveMemory},
                                                {DNNL_ARG_DST, *_output.effectiveMemory}});
        }

        template <typename data_t>
        void DnnlDenseLayer<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();
            compileBackwardWeightsStream();
            compileBackwardDataStream();
        }

        template <typename data_t>
        void DnnlDenseLayer<data_t>::compileBackwardDataStream()
        {
            auto desc = dnnl::inner_product_backward_data::desc(
                /* Input gradient descriptor (output) */ _inputGradient.front().descriptor,
                /* Weights descriptor */ _weights.descriptor,
                /* Output gradient descriptor */ _outputGradient.front().descriptor);

            _backwardDataPrimitiveDescriptor = dnnl::inner_product_backward_data::primitive_desc(
                desc, *_engine, _forwardPrimitiveDescriptor);

            // Reorder output gradient of necessary
            this->reorderMemory(_backwardDataPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            // Reorder weights if necessary
            this->reorderMemory(_backwardDataPrimitiveDescriptor.weights_desc(), _weights,
                                _backwardStream);

            // Set input gradient memory
            _inputGradient.front().effectiveMemory = std::make_shared<dnnl::memory>(
                _backwardDataPrimitiveDescriptor.diff_src_desc(), *_engine);

            // Push backward data primitive
            ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream, dnnl::inner_product_backward_data(
                                                            _backwardDataPrimitiveDescriptor));

            BaseType::validateDnnlMemory(_inputGradient.front().effectiveMemory,
                                         _weights.effectiveMemory,
                                         _outputGradient.front().effectiveMemory);

            _backwardStream.arguments.push_back(
                {/*  Input gradient */ {DNNL_ARG_DIFF_SRC, *_inputGradient.front().effectiveMemory},
                 /*  Weights */ {DNNL_ARG_WEIGHTS, *_weights.effectiveMemory},
                 /*  Output gradient */
                 {DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory}});
        }

        template <typename data_t>
        void DnnlDenseLayer<data_t>::compileBackwardWeightsStream()
        {
            // Backward descriptor for weights backprop
            auto desc = dnnl::inner_product_backward_weights::desc(
                /* Input descriptor */ _input.front().descriptor,
                /* Weights gradient descriptor */ _weightsGradient.descriptor,
                /* Bias gradient descriptor */ _biasGradient.descriptor,
                /* Output gradient descriptor */ _outputGradient.front().descriptor);

            _backwardWeightsPrimitiveDescriptor =
                dnnl::inner_product_backward_weights::primitive_desc(desc, *_engine,
                                                                     _forwardPrimitiveDescriptor);

            // Do we need reorder for gradient src memory?
            this->reorderMemory(_backwardWeightsPrimitiveDescriptor.src_desc(), _input.front(),
                                _backwardStream);

            this->reorderMemory(_backwardWeightsPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            BaseType::validateDnnlMemory(_input.front().effectiveMemory,
                                         _biasGradient.effectiveMemory,
                                         _outputGradient.front().effectiveMemory);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream, dnnl::inner_product_backward_weights(
                                                            _backwardWeightsPrimitiveDescriptor));
            _backwardStream.arguments.push_back(
                {{DNNL_ARG_SRC, *_input.front().effectiveMemory},
                 {DNNL_ARG_DIFF_BIAS, *_biasGradient.effectiveMemory},
                 {DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory}});

            _weightsGradient.effectiveMemory = _weightsGradient.describedMemory;
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

        template class DnnlDenseLayer<float>;
    } // namespace detail
} // namespace elsa::ml
