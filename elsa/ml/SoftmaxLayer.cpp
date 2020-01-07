#include "SoftmaxLayer.h"

namespace elsa
{
    template <typename data_t, MlBackend BackendTag>
    SoftmaxLayer<data_t, BackendTag>::SoftmaxLayer(const DataDescriptor& inputDescriptor)
        : Layer<data_t, BackendTag>(inputDescriptor)
    {
        _outputDescriptor = inputDescriptor.clone();
        _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor);
    }

    template class SoftmaxLayer<float, MlBackend::Dnnl>;
} // namespace elsa
