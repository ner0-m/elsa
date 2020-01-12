#include "DnnlTrainableLayer.h"

namespace elsa
{

    template <typename data_t>
    DnnlTrainableLayer<data_t>::DnnlTrainableLayer(const DataDescriptor& inputDescriptor,
                                                   const DataDescriptor& outputDescriptor,
                                                   const DataDescriptor& weightsDescriptor,
                                                   Initializer initializer)
        : DnnlLayer<data_t>(inputDescriptor, outputDescriptor),
          _weightsDescriptor(weightsDescriptor.clone()),
          _initializer(initializer)
    {
        _input.canBeReordered = true;

        // Set the layer's fan-in and fan-out. This is needed for random initialization of weights
        // and biases
        _fanInOut.first = inputDescriptor.getNumberOfCoefficients();
        _fanInOut.second = outputDescriptor.getNumberOfCoefficients();

        // Set weights meta information
        for (const auto& dim : weightsDescriptor.getNumberOfCoefficientsPerDimension())
            _weights.dimensions.push_back(dim);

        _weightsGradient.dimensions = _weights.dimensions;

        _weights.formatTag =
            BaseType::dataDescriptorToDnnlMemoryFormatTag(weightsDescriptor,
                                                          /* No input but weights tag */ false);

        _weightsGradient.formatTag = _weights.formatTag;

        _weights.descriptor =
            dnnl::memory::desc({_weights.dimensions}, _typeTag, dnnl::memory::format_tag::any);

        _weightsGradient.descriptor = _weights.descriptor;

        IndexVector_t biasVec(1);
        biasVec << _weights.dimensions[0];

        _biasDescriptor = DataDescriptor(biasVec).clone();

        // Set weights bias information
        _bias.dimensions.push_back(_weights.dimensions[0]);

        _biasGradient.dimensions = _bias.dimensions;

        _bias.descriptor =
            dnnl::memory::desc({_bias.dimensions}, _typeTag, dnnl::memory::format_tag::any);

        _biasGradient.descriptor = _bias.descriptor;

        _bias.formatTag = dnnl::memory::format_tag::x;

        _biasGradient.formatTag = _bias.formatTag;
    }

    template <typename data_t>
    void DnnlTrainableLayer<data_t>::setWeights(const DataContainer<data_t>& weights)
    {
        this->writeToDnnlMemory(weights, *_weights.describedMemory);
    }

    template <typename data_t>
    void DnnlTrainableLayer<data_t>::setBias(const DataContainer<data_t>& bias)
    {
        this->writeToDnnlMemory(bias, *_bias.describedMemory);
    }

    template <typename data_t>
    void DnnlTrainableLayer<data_t>::compileForwardStream()
    {
        // Construct weights memory and initialize it
        auto weightsDesc = dnnl::memory::desc({_weights.dimensions}, _typeTag, _weights.formatTag);
        _weights.describedMemory = std::make_shared<dnnl::memory>(weightsDesc, *_engine);

        RandomInitializer<data_t>::initialize(
            static_cast<data_t*>(_weights.describedMemory->get_data_handle()),
            _weightsDescriptor->getNumberOfCoefficients(), _initializer, _fanInOut);

        // Construct bias memory and initialize it
        auto biasDesc = dnnl::memory::desc({_bias.dimensions}, _typeTag, _bias.formatTag);
        _bias.describedMemory = std::make_shared<dnnl::memory>(biasDesc, *_engine);

        RandomInitializer<data_t>::initialize(
            static_cast<data_t*>(_bias.describedMemory->get_data_handle()), _bias.dimensions[0],
            _initializer, _fanInOut);

        // Bias can never be reordered
        _bias.effectiveMemory = _bias.describedMemory;
    }

    template <typename data_t>
    void DnnlTrainableLayer<data_t>::compileBackwardStream()
    {
        auto weightsDesc =
            dnnl::memory::desc({_weightsGradient.dimensions}, _typeTag, _weightsGradient.formatTag);
        _weightsGradient.describedMemory = std::make_shared<dnnl::memory>(weightsDesc, *_engine);

        auto biasDesc =
            dnnl::memory::desc({_biasGradient.dimensions}, _typeTag, _biasGradient.formatTag);
        _biasGradient.describedMemory = std::make_shared<dnnl::memory>(biasDesc, *_engine);

        // Bias can never be reordered
        _biasGradient.effectiveMemory = _biasGradient.describedMemory;
    }

    template <typename data_t>
    DataContainer<data_t> DnnlTrainableLayer<data_t>::getGradientWeights() const
    {
        DataContainer<data_t> output(*_weightsDescriptor);
        this->readFromDnnlMemory(output, *_weightsGradient.effectiveMemory);
        return output;
    }

    template <typename data_t>
    DataContainer<data_t> DnnlTrainableLayer<data_t>::getGradientBias() const
    {
        DataContainer<data_t> output(*_biasDescriptor);
        this->readFromDnnlMemory(output, *_biasGradient.effectiveMemory);
        return output;
    }

    template class DnnlTrainableLayer<float>;

} // namespace elsa