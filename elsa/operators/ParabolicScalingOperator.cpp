#include "ParabolicScalingOperator.h"

namespace elsa
{
    template <typename data_t>
    ParabolicScalingOperator<data_t>::ParabolicScalingOperator(real_t scalingParameter)
        : LinearOperator<data_t>(VolumeDescriptor{2, 2}, VolumeDescriptor{2, 2}),
          _scalingParameter{scalingParameter}
    {
        if (scalingParameter <= 0) {
            throw LogicError("ParabolicScalingOperator: the scaling parameter a cannot be <= 0");
        }
    }

    template <typename data_t>
    void ParabolicScalingOperator<data_t>::applyInverse(const DataContainer<data_t>& x,
                                                        DataContainer<data_t>& Aax) const
    {
        // TODO add logic
        Timer timeguard("ParabolicScalingOperator", "applyInverse");

        index_t numberOfRows = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0];
        index_t numberOfColumns = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1];

        printf("rowz I iz %ld", numberOfRows);
        printf("colz I iz %ld", numberOfColumns);

        if (numberOfRows != 2) {
            throw LogicError(
                "ParabolicScalingOperator: the number of rows of the vector x should be 2");
        }

        if (Aax.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0] != 2
            || Aax.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1]
                   != numberOfColumns) {
            throw LogicError(
                "ParabolicScalingOperator: the number of rows of the vector x should be 2");
        }

        for (index_t j = 0; j < numberOfColumns; j++) {
            Aax[j] = (1 / _scalingParameter) * x[j];
            Aax[j + numberOfColumns] = (1 / std::sqrt(_scalingParameter)) * x[j + numberOfColumns];
        }
        // A is 2x2, y is 2xn, output is 2xn
    }

    template <typename data_t>
    void ParabolicScalingOperator<data_t>::applyInverseAdjoint(const DataContainer<data_t>& y,
                                                               DataContainer<data_t>& Aaty) const
    {
        /// similar logic since the operator is symmetric // TODO remove me before MR
        Timer timeguard("ParabolicScalingOperator", "applyInverseAdjoint");
        applyImpl(y, Aaty);
    }

    template <typename data_t>
    void ParabolicScalingOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                     DataContainer<data_t>& Aax) const
    {
        // TODO add logic
        Timer timeguard("ParabolicScalingOperator", "apply");

        index_t numberOfRows = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0];
        index_t numberOfColumns = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1];

        printf("rowz iz %ld", numberOfRows);
        printf("colz iz %ld", numberOfColumns);

        if (numberOfRows != 2) {
            throw LogicError(
                "ParabolicScalingOperator: the number of rows of the vector x should be 2");
        }

        if (Aax.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0] != 2
            || Aax.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1]
                   != numberOfColumns) {
            throw LogicError(
                "ParabolicScalingOperator: the number of rows of the vector x should be 2");
        }

        for (index_t j = 0; j < numberOfColumns; j++) {
            Aax[j] = _scalingParameter * x[j];
            Aax[j + numberOfColumns] = std::sqrt(_scalingParameter) * x[j + numberOfColumns];
        }
        // A is 2x2, y is 2xn, output is 2xn
    }
    // j goes from [0 to 4]
    // 0 1 2 3 4
    // 5 6 7 8 9

    template <typename data_t>
    void ParabolicScalingOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                            DataContainer<data_t>& Aaty) const
    {
        /// similar logic since the operator is symmetric // TODO remove me before MR
        Timer timeguard("ParabolicScalingOperator", "applyAdjoint");
        applyImpl(y, Aaty);
    }

    template <typename data_t>
    ParabolicScalingOperator<data_t>* ParabolicScalingOperator<data_t>::cloneImpl() const
    {
        return new ParabolicScalingOperator(_scalingParameter);
    }

    template <typename data_t>
    bool ParabolicScalingOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherParabolicScaling = dynamic_cast<const ParabolicScalingOperator*>(&other);
        return static_cast<bool>(otherParabolicScaling);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ParabolicScalingOperator<float>;
    //    template class ParabolicScalingOperator<std::complex<float>>;
    template class ParabolicScalingOperator<double>;
    //    template class ParabolicScalingOperator<std::complex<double>>;
} // namespace elsa
