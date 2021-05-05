#include "ShearingOperator.h"
#include "VolumeDescriptor.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    ShearingOperator<data_t>::ShearingOperator(const DataDescriptor& rangeDescriptor,
                                               real_t shearingParameter)
        : LinearOperator<data_t>(
            VolumeDescriptor{rangeDescriptor.getNumberOfCoefficientsPerDimension()[0],
                             rangeDescriptor.getNumberOfCoefficientsPerDimension()[0]},
            rangeDescriptor),
          _shearingParameter{shearingParameter}
    {
        if (rangeDescriptor.getNumberOfCoefficientsPerDimension()[0]
            != _shearingOperatorDimensions) {
            throw LogicError("ShearingOperator: the number of rows of the vector x should be 2");
        }
    }

    template <typename data_t>
    void ShearingOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                             DataContainer<data_t>& Ssx) const
    {
        Timer timeguard("ShearingOperator", "apply");

        index_t numberOfRows = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0];
        index_t numberOfColumns = x.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1];

        if (numberOfRows != _shearingOperatorDimensions) {
            throw LogicError("ShearingOperator: the number of rows of the vector x should be 2");
        }

        if (Ssx.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0]
                != _shearingOperatorDimensions
            || Ssx.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1]
                   != numberOfColumns) {
            throw LogicError("ShearingOperator: the number of rows of the vector x should be 2");
        }

        for (index_t j = 0; j < numberOfColumns; j++) {
            Ssx[j] = x[j] + _shearingParameter * x[j + numberOfColumns];
            Ssx[j + numberOfColumns] = x[j + numberOfColumns];
        }
    }

    template <typename data_t>
    void ShearingOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                    DataContainer<data_t>& Ssty) const
    {
        Timer timeguard("ShearingOperator", "applyAdjoint");

        index_t numberOfRows = y.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0];
        index_t numberOfColumns = y.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1];

        if (numberOfRows != _shearingOperatorDimensions) {
            throw LogicError("ShearingOperator: the number of rows of the vector y should be 2");
        }

        if (Ssty.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0]
                != _shearingOperatorDimensions
            || Ssty.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1]
                   != numberOfColumns) {
            throw LogicError("ShearingOperator: the number of rows of the vector y should be 2");
        }

        for (index_t j = 0; j < numberOfColumns; j++) {
            Ssty[j] = y[j];
            Ssty[j + numberOfColumns] = _shearingParameter * y[j] + y[j + numberOfColumns];
        }
    }

    template <typename data_t>
    ShearingOperator<data_t>* ShearingOperator<data_t>::cloneImpl() const
    {
        return new ShearingOperator(this->getRangeDescriptor(), _shearingParameter);
    }

    template <typename data_t>
    bool ShearingOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherShearingOp = dynamic_cast<const ShearingOperator*>(&other);
        return static_cast<bool>(otherShearingOp);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ShearingOperator<float>;
    //    template class ShearingOperator<std::complex<float>>;
    template class ShearingOperator<double>;
    //    template class ShearingOperator<std::complex<double>>;
} // namespace elsa
