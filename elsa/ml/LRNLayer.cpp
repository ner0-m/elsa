#include "LRNLayer.h"

namespace elsa
{
    template <typename data_t, MlBackend BackendTag>
    LRNLayer<data_t, BackendTag>::LRNLayer(const DataDescriptor& inputDescriptor, index_t localSize,
                                           data_t alpha, data_t beta, data_t k)
        : Layer<data_t, BackendTag>(inputDescriptor)
    {
        _outputDescriptor = inputDescriptor.clone();
        _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor,
                                                      localSize, alpha, beta, k);
    }

    template class LRNLayer<float, MlBackend::Dnnl>;
} // namespace elsa
