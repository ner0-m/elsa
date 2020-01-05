#include "DnnlTrainableLayer.h"
#include <iostream>

namespace elsa
{

    template <typename data_t>
    DnnlTrainableLayer<data_t>::DnnlTrainableLayer(const DataDescriptor& inputDescriptor,
                                                   const DataDescriptor& outputDescriptor,
                                                   const DataDescriptor& weightsDescriptor,
                                                   Initializer initializer)
        : DnnlLayer<data_t>(inputDescriptor, outputDescriptor),
          _initializer(initializer),
          _weightsDescriptor(weightsDescriptor.clone())
    {
        for (const auto& dim : weightsDescriptor.getNumberOfCoefficientsPerDimension())
            _weightsDimensions.push_back(dim);

        _weightsMemoryFormatTag =
            BaseType::dataDescriptorToDnnlMemoryFormatTag(weightsDescriptor,
                                                          /* No input but weights tag */ false);

        _weightsMemoryDescriptor =
            dnnl::memory::desc({_weightsDimensions}, _typeTag, dnnl::memory::format_tag::any);

        _biasDimensions.push_back(_weightsDimensions[0]);

        _biasMemoryDescriptor =
            dnnl::memory::desc({_biasDimensions}, _typeTag, dnnl::memory::format_tag::any);
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
    void DnnlTrainableLayer<data_t>::compile()
    {
        _weightsMemory =
            dnnl::memory({{_weightsDimensions}, _typeTag, _weightsMemoryFormatTag}, *_engine);

        RandomInitializer<data_t>::initialize(
            static_cast<data_t*>(_weightsMemory.get_data_handle()),
            _weightsDescriptor->getNumberOfCoefficients(), _initializer);

        _biasMemory =
            dnnl::memory({{_biasDimensions}, _typeTag, dnnl::memory::format_tag::x}, *_engine);

        RandomInitializer<data_t>::initialize(static_cast<data_t*>(_biasMemory.get_data_handle()),
                                              _biasDimensions[0], _initializer);
    }

    template class DnnlTrainableLayer<float>;

} // namespace elsa