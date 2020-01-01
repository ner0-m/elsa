#include "Dense.h"

namespace elsa
{
    template <typename data_t, MlBackend _BackendTag>
    Dense<data_t, _BackendTag>::Dense(const DataDescriptor& inputDescriptor, int numNeurons)
        : TrainableLayer<data_t, _BackendTag>(inputDescriptor), _numNeurons(numNeurons)
    {
        IndexVector_t weightsVec(inputDescriptor.getNumberOfDimensions());

        weightsVec[0] = numNeurons;

        for (int idx = 1; idx < inputDescriptor.getNumberOfDimensions(); ++idx)
            weightsVec[idx] = inputDescriptor.getNumberOfCoefficientsPerDimension()[idx];

        _weightsDescriptor = DataDescriptor(weightsVec).clone();

        IndexVector_t biasDims(1);
        biasDims << _weightsDescriptor->getNumberOfCoefficientsPerDimension()[0];
        _biasDescriptor = DataDescriptor(biasDims).clone();

        // The output of any dense layer is a matrix containing an output vector for each batch
        IndexVector_t outputDims(2);

        outputDims
            // Batch
            << inputDescriptor.getNumberOfCoefficientsPerDimension()[0],
            // Number of filters (channels, neurons)
            _weightsDescriptor->getNumberOfCoefficientsPerDimension()[0];

        _outputDescriptor = DataDescriptor(outputDims).clone();
        _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor,
                                                      *_weightsDescriptor);
    }

    template class Dense<float, MlBackend::Dnnl>;
} // namespace elsa