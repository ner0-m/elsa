#include "Conv.h"

namespace elsa
{
    template <typename data_t, MlBackend _BackendTag>
    Conv<data_t, _BackendTag>::Conv(const DataDescriptor& inputDescriptor,
                                    const DataDescriptor& weightsDescriptor,
                                    const IndexVector_t& strideVector,
                                    const IndexVector_t& paddingVector, Initializer initializer)
        : TrainableLayer<data_t, _BackendTag>(inputDescriptor, weightsDescriptor)
    {
        if (inputDescriptor.getNumberOfDimensions() != weightsDescriptor.getNumberOfDimensions())
            throw std::invalid_argument("Number of dimensions of input and weights must match");

        // Number of channels must be equal for input and weights
        if (inputDescriptor.getNumberOfCoefficientsPerDimension()[1]
            != weightsDescriptor.getNumberOfCoefficientsPerDimension()[1])
            throw std::invalid_argument("Number of channels for input and weights must match");

        // inputDescriptor is in nchw or nchwd format, i.e., the first dimension is the batch
        // dimension, the second is the channel dimension and all other dimensions are spatial
        IndexVector_t outputDims(inputDescriptor.getNumberOfDimensions());
        // We assume nchw or nchwd for the input and oihw or oihwd format for the weights
        // nchw case

        outputDims
            // Batch
            << inputDescriptor.getNumberOfCoefficientsPerDimension()[0],
            // Number of filters
            weightsDescriptor.getNumberOfCoefficientsPerDimension()[0],
            // For each spatial dimension:
            // output = (input - kernel + 2 * padding) / stride + 1
            (inputDescriptor.getNumberOfCoefficientsPerDimension()[2]
             - weightsDescriptor.getNumberOfCoefficientsPerDimension()[2]
             + 2 * paddingVector.coeff(0))
                    / strideVector.coeff(0)
                + 1,
            (inputDescriptor.getNumberOfCoefficientsPerDimension()[2]
             - weightsDescriptor.getNumberOfCoefficientsPerDimension()[2]
             + 2 * paddingVector.coeff(1))
                    / strideVector.coeff(1)
                + 1;

        _outputDescriptor = DataDescriptor(outputDims).clone();
        _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor,
                                                      weightsDescriptor, strideVector,
                                                      paddingVector, initializer);
    }

    template class Conv<float, MlBackend::Dnnl>;
} // namespace elsa