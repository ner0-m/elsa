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
        BaseType::_mayReorderMemory = true;

        // Set the layer's fan-in and fan-out. This is needed for random initialization of weights
        // and biases
        _fanInOut.first = inputDescriptor.getNumberOfCoefficients();
        _fanInOut.second = outputDescriptor.getNumberOfCoefficients();

        // Set weights meta information
        for (const auto& dim : weightsDescriptor.getNumberOfCoefficientsPerDimension())
            _weightsDimensions.push_back(dim);

        _weightsMemoryFormatTag =
            BaseType::dataDescriptorToDnnlMemoryFormatTag(weightsDescriptor,
                                                          /* No input but weights tag */ false);

        _weightsMemoryDescriptor =
            dnnl::memory::desc({_weightsDimensions}, _typeTag, dnnl::memory::format_tag::any);

        _gradientWeightsMemoryDescriptor = _weightsMemoryDescriptor;

        IndexVector_t biasVec(1);
        biasVec << _weightsDimensions[0];

        _biasDescriptor = DataDescriptor(biasVec).clone();

        // Set weights bias information
        _biasDimensions.push_back(_weightsDimensions[0]);

        _biasMemoryDescriptor =
            dnnl::memory::desc({_biasDimensions}, _typeTag, dnnl::memory::format_tag::any);

        _gradientBiasMemoryDescriptor = _biasMemoryDescriptor;
    }

    template <typename data_t>
    void DnnlTrainableLayer<data_t>::setWeights(const DataContainer<data_t>& weights)
    {
        this->writeToDnnlMemory(weights, _weightsMemory);
    }

    template <typename data_t>
    void DnnlTrainableLayer<data_t>::setBias(const DataContainer<data_t>& bias)
    {
        this->writeToDnnlMemory(bias, _biasMemory);
    }

    template <typename data_t>
    void DnnlTrainableLayer<data_t>::compileForwardStream()
    {
        // Construct weights memory and initialize it
        _weightsMemory =
            dnnl::memory({{_weightsDimensions}, _typeTag, _weightsMemoryFormatTag}, *_engine);

        RandomInitializer<data_t>::initialize(
            static_cast<data_t*>(_weightsMemory.get_data_handle()),
            _weightsDescriptor->getNumberOfCoefficients(), _initializer, _fanInOut);

        // Construct bias memory and initialize it
        _biasMemory =
            dnnl::memory({{_biasDimensions}, _typeTag, dnnl::memory::format_tag::x}, *_engine);

        RandomInitializer<data_t>::initialize(static_cast<data_t*>(_biasMemory.get_data_handle()),
                                              _biasDimensions[0], _initializer, _fanInOut);
    }

    template <typename data_t>
    void DnnlTrainableLayer<data_t>::compileBackwardStream()
    {
        _gradientWeightsMemory =
            dnnl::memory({{_weightsDimensions}, _typeTag, _weightsMemoryFormatTag}, *_engine);

        _gradientBiasMemory =
            dnnl::memory({{_biasDimensions}, _typeTag, dnnl::memory::format_tag::x}, *_engine);
    }

    template <typename data_t>
    DataContainer<data_t> DnnlTrainableLayer<data_t>::getGradientWeights() const
    {
        DataContainer<data_t> output(*_weightsDescriptor);
        this->readFromDnnlMemory(output, _gradientWeightsMemory);
        return output;
    }

    template <typename data_t>
    DataContainer<data_t> DnnlTrainableLayer<data_t>::getGradientBias() const
    {
        DataContainer<data_t> output(*_biasDescriptor);
        this->readFromDnnlMemory(output, _gradientBiasMemory);
        return output;
    }

    template class DnnlTrainableLayer<float>;

} // namespace elsa