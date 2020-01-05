#include "PoolingLayer.h"

namespace elsa
{
    template <typename data_t, MlBackend _BackendTag>
    PoolingLayer<data_t, _BackendTag>::PoolingLayer(const DataDescriptor& inputDescriptor,
                                                    const IndexVector_t& poolingWindow,
                                                    const IndexVector_t& poolingStride)
        : Layer<data_t, _BackendTag>(inputDescriptor),
          _poolingWindow(poolingWindow),
          _poolingStride(poolingStride)
    {
        if ((poolingStride.array() <= 0).any())
            throw std::invalid_argument("Pooling stride must not be 0");
        if (_poolingWindow.size() != _poolingStride.size())
            throw std::invalid_argument(
                "Dimensions of pooling stride and pooling window must match");

        // inputDescriptor is in nchw or nchwd format, i.e., the first dimension is the batch
        // dimension, the second is the channel dimension and the other dimensions are spatial
        IndexVector_t outputDims(inputDescriptor.getNumberOfDimensions());
        outputDims <<
            // batch
            inputDescriptor.getNumberOfCoefficientsPerDimension()[0],
            // channels
            inputDescriptor.getNumberOfCoefficientsPerDimension()[1],
            // For each spatial dimension:
            // output = (input - pooling) / stride + 1
            (inputDescriptor.getNumberOfCoefficientsPerDimension()[2] - poolingWindow[0])
                    / poolingStride[0]
                + 1,
            (inputDescriptor.getNumberOfCoefficientsPerDimension()[3] - poolingWindow[1])
                    / poolingStride[1]
                + 1;

        _outputDescriptor = DataDescriptor(outputDims).clone();
        _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor,
                                                      _poolingWindow, _poolingStride);
    }

    template class PoolingLayer<float>;

} // namespace elsa