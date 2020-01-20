#include "FixedLayer.h"
#include <iostream>

namespace elsa
{
    template <typename data_t, MlBackend Backend>
    FixedLayer<data_t, Backend>::FixedLayer(const DataDescriptor& inputDescriptor,
                                            const JosephsMethod<data_t>& op)
        : Layer<data_t, Backend>(inputDescriptor)
    {
        // The layer's output descriptor is the operators input descriptor
        _outputDescriptor = op.getDomainDescriptor().clone();

        // The layer's input descriptor must match the operators output descriptor. We could even
        // avoid setting the input descriptor for this layer entirely but keep it to be able to
        // provide a uniform API
        if (inputDescriptor != op.getRangeDescriptor())
            throw std::invalid_argument(
                "Fixed layer's input descriptor must match the linear operator's outpu descriptor");

        _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor, op);
    }

    template <typename data_t, MlBackend Backend>
    bool FixedLayer<data_t, Backend>::isOperator() const
    {
        return true;
    }

    template class FixedLayer<float, MlBackend::Dnnl>;
} // namespace elsa
