#include "Matrix.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    Matrix<data_t>::Matrix(const DataDescriptor& descriptor)
        : LinearOperator<data_t>(initDomainDescriptor(descriptor), initRangeDescriptor(descriptor)),
          _matrix{DataContainer<data_t>(descriptor)}
    {
    }

    template <typename data_t>
    Matrix<data_t>::Matrix(const DataContainer<data_t>& data)
        : LinearOperator<data_t>(initDomainDescriptor(data.getDataDescriptor()),
                                 initRangeDescriptor(data.getDataDescriptor())),
          _matrix{data}
    {
    }

    template <typename data_t>
    VolumeDescriptor Matrix<data_t>::initDomainDescriptor(const DataDescriptor& descriptor)
    {
        auto volumeDesc = downcast_safe<VolumeDescriptor>(&descriptor);
        if (!volumeDesc || volumeDesc->getNumberOfDimensions() != 2) {
            throw InvalidArgumentError(
                "Matrix can only be initialized from data with 2 dimensional VolumeDescriptor");
        }

        return VolumeDescriptor({volumeDesc->getNumberOfCoefficientsPerDimension()[1]});
    }

    template <typename data_t>
    VolumeDescriptor Matrix<data_t>::initRangeDescriptor(const DataDescriptor& descriptor)
    {
        auto volumeDesc = downcast_safe<VolumeDescriptor>(&descriptor);
        if (!volumeDesc || volumeDesc->getNumberOfDimensions() != 2) {
            throw InvalidArgumentError(
                "Matrix can only be initialized from data with 2 dimensional VolumeDescriptor");
        }

        return VolumeDescriptor({volumeDesc->getNumberOfCoefficientsPerDimension()[0]});
    }

    template <typename data_t>
    void Matrix<data_t>::applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const
    {
        Timer timeguard("Matrix", "apply");

        if (x.getDataDescriptor() != *_domainDescriptor
            || Ax.getDataDescriptor() != *_rangeDescriptor)
            throw InvalidArgumentError("Matrix::apply: incorrect input/output sizes");

        IndexVector_t coeffsPerDim =
            _matrix.getDataDescriptor().getNumberOfCoefficientsPerDimension();

        Ax = 0;
        for (index_t i = 0; i < coeffsPerDim[0]; ++i) {
            for (index_t j = 0; j < coeffsPerDim[1]; ++j) {
                Ax[i] += _matrix(i, j) * x(j);
            }
        }
    }

    template <typename data_t>
    void Matrix<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                          DataContainer<data_t>& Aty) const
    {
        Timer timeguard("Matrix", "applyAdjoint");

        if (y.getDataDescriptor() != *_rangeDescriptor
            || Aty.getDataDescriptor() != *_domainDescriptor)
            throw InvalidArgumentError("Matrix::applyAdjoint: incorrect input/output sizes");

        IndexVector_t coeffsPerDim =
            _matrix.getDataDescriptor().getNumberOfCoefficientsPerDimension();

        Aty = 0;
        for (index_t i = 0; i < coeffsPerDim[1]; ++i) {
            for (index_t j = 0; j < coeffsPerDim[0]; ++j) {
                Aty[i] += _matrix(j, i) * y(j);
            }
        }
    }

    template <typename data_t>
    Matrix<data_t>* Matrix<data_t>::cloneImpl() const
    {
        return new Matrix(_matrix);
    }

    template <typename data_t>
    bool Matrix<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherMatrix = downcast_safe<Matrix>(&other);
        if (!otherMatrix)
            return false;

        if (_matrix != otherMatrix->_matrix)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Matrix<float>;
    template class Matrix<double>;

} // namespace elsa
