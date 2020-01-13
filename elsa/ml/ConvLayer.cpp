#include "ConvLayer.h"

namespace elsa
{
    template <typename data_t, MlBackend _BackendTag>
    ConvLayer<data_t, _BackendTag>::ConvLayer(const DataDescriptor& inputDescriptor,
                                              const DataDescriptor& weightsDescriptor,
                                              const IndexVector_t& strideVector,
                                              const IndexVector_t& paddingVector,
                                              Initializer initializer)
        : TrainableLayer<data_t, _BackendTag>(inputDescriptor, weightsDescriptor)
    {
        if (inputDescriptor.getNumberOfDimensions() != weightsDescriptor.getNumberOfDimensions())
            throw std::invalid_argument("Number of dimensions of input and weights must match");

        // Number of channels must be equal for input and weights
        if (inputDescriptor.getNumberOfCoefficientsPerDimension()[1]
            != weightsDescriptor.getNumberOfCoefficientsPerDimension()[1])
            throw std::invalid_argument("Number of channels for input and weights must match");

        if (inputDescriptor.getNumberOfDimensions() < 3
            || inputDescriptor.getNumberOfDimensions() > 5)
            throw std::invalid_argument(
                "Number of input and weight dimensions must be at least 3 and at most 5");

        // If input dim is 3, we assume ncw format (1D convolution) and need exactly
        // one stride and one padding dimension
        if (inputDescriptor.getNumberOfDimensions() == 3
            && (strideVector.size() != 1 || paddingVector.size() != 1))
            throw std::invalid_argument(
                "Input dimension 3 requires exactly one stride value and one padding value");

        // If input dim is 4, we assume nchw format (2D convolution) and need exactly
        // two strides and two padding dimension
        if (inputDescriptor.getNumberOfDimensions() == 4
            && (strideVector.size() != 2 || paddingVector.size() != 2))
            throw std::invalid_argument(
                "Input dimension 4 requires exactly two stride values and two padding values");

        // If input dim is 4, we assume ncdhw format (3D convolution) and need exactly
        // three strides and three padding dimension
        if (inputDescriptor.getNumberOfDimensions() == 5
            && (strideVector.size() != 3 || paddingVector.size() != 3))
            throw std::invalid_argument(
                "Input dimension 5 requires exactly three stride values and three padding values");

        IndexVector_t outputDims(inputDescriptor.getNumberOfDimensions());

        // Batch dimension
        outputDims[0] = inputDescriptor.getNumberOfCoefficientsPerDimension()[0];

        // Input channels
        outputDims[1] = weightsDescriptor.getNumberOfCoefficientsPerDimension()[0];

        // For each spatial dimension:
        // output = (input - kernel + 2 * padding) / stride + 1
        for (index_t i = 2; i < inputDescriptor.getNumberOfDimensions(); ++i) {
            outputDims[i] = (inputDescriptor.getNumberOfCoefficientsPerDimension()[i]
                             - weightsDescriptor.getNumberOfCoefficientsPerDimension()[i]
                             + 2 * paddingVector.coeff(i - 2))
                                / strideVector.coeff(i - 2)
                            + 1;
        }

        _outputDescriptor = DataDescriptor(outputDims).clone();
        _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor,
                                                      weightsDescriptor, strideVector,
                                                      paddingVector, initializer);
    }

    template <typename data_t, MlBackend _BackendTag>
    ConvLayer<data_t, _BackendTag>::ConvLayer(const DataDescriptor& inputDescriptor,
                                              index_t numFilters,
                                              const IndexVector_t& spatialFilterVector,
                                              const IndexVector_t& strideVector,
                                              const IndexVector_t& paddingVector,
                                              Initializer initializer)
    {
        index_t numSpatialDims = spatialFilterVector.size();
        if (strideVector.size() != numSpatialDims || paddingVector.size() != numSpatialDims)
            throw std::invalid_argument(
                "Spatial dimensions of filters, strides and padding must match");

        IndexVector_t weightsVec(inputDescriptor.getNumberOfDimensions());

        // Num filters
        weightsVec[0] = numFilters;

        // Num input channels
        weightsVec[1] = inputDescriptor.getNumberOfCoefficientsPerDimension()[1];

        // Spatial dims
        for (index_t i = 0; i < spatialFilterVector.size(); ++i)
            weightsVec[i + 2] = spatialFilterVector[i];

        DataDescriptor weightsDesc(weightsVec);
        *this = ConvLayer<data_t, _BackendTag>(inputDescriptor, weightsDesc, strideVector,
                                               paddingVector, initializer);
    }

    template <typename data_t, MlBackend _BackendTag>
    ConvLayer<data_t, _BackendTag>::ConvLayer(const DataDescriptor& inputDescriptor,
                                              index_t numFilters,
                                              const IndexVector_t& spatialFilterVector,
                                              Initializer initializer)
    {
        IndexVector_t stridesVector = spatialFilterVector;
        IndexVector_t paddingVector = IndexVector_t::Ones(stridesVector.size());
        *this = ConvLayer<data_t, _BackendTag>(inputDescriptor, numFilters, spatialFilterVector,
                                               stridesVector, paddingVector, initializer);
    }

    template class ConvLayer<float, MlBackend::Dnnl>;
} // namespace elsa
