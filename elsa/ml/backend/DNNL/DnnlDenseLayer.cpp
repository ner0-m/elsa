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
            /* Source descriptor*/ _input.descriptor,
            /* Weights memory descriptor*/ _weights.descriptor,
            /* Bias memory descriptor*/ _bias.descriptor,
            /* Destination memorydescriptor*/ _output.descriptor);

        _forwardPrimitiveDescriptor = dnnl::inner_product_forward::primitive_desc(desc, *_engine);

        // Do we need to reorder? This is the case of the memory description chosen by the primitive
        // differs from the memory description we provided based on the layer's input descriptor /
        // weights descriptor.
        _input.effectiveMemory = _input.effectiveMemory;
        if (_forwardPrimitiveDescriptor.src_desc() != _input.describedMemory->get_desc()) {
            // Remember reordering to allow reverting it during output
            _input.wasReordered = true;
            _input.effectiveMemory =
                std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.src_desc(), *_engine);
            _forwardStream.primitives.push_back(
                dnnl::reorder(*_input.describedMemory, *_input.effectiveMemory));
            _forwardStream.arguments.push_back(
                {{DNNL_ARG_FROM, *_input.describedMemory}, {DNNL_ARG_TO, *_input.effectiveMemory}});
        }

        _weights.effectiveMemory = _weights.describedMemory;
        if (_forwardPrimitiveDescriptor.weights_desc() != _weights.describedMemory->get_desc()) {
            // Remember reordering to allow reverting it during output
            _weights.wasReordered = true;
            _weights.effectiveMemory = std::make_shared<dnnl::memory>(
                _forwardPrimitiveDescriptor.weights_desc(), *_engine);
            _forwardStream.primitives.push_back(
                dnnl::reorder(*_weights.describedMemory, *_weights.effectiveMemory));
            _forwardStream.arguments.push_back({{DNNL_ARG_FROM, *_weights.describedMemory},
                                                {DNNL_ARG_TO, *_weights.effectiveMemory}});
        }

        _output.effectiveMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardStream.primitives.push_back(
            dnnl::inner_product_forward(_forwardPrimitiveDescriptor));

        _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.effectiveMemory},
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
    }

    template <typename data_t>
    void DnnlDenseLayer<data_t>::compileBackwardWeightsStream()
    {
    }

    template class DnnlDenseLayer<float>;
} // namespace elsa
