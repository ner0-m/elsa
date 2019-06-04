#include "DataDescriptor.h"

#include <stdexcept>

namespace elsa
{
    DataDescriptor::DataDescriptor(IndexVector_t numberOfCoefficientsPerDimension)
        : _numberOfDimensions{numberOfCoefficientsPerDimension.size()},
          _numberOfCoefficientsPerDimension{numberOfCoefficientsPerDimension},
          _spacingPerDimension{RealVector_t::Ones(_numberOfDimensions)},
          _productOfCoefficientsPerDimension{numberOfCoefficientsPerDimension}
    {
        // sanity checks
        if ( (numberOfCoefficientsPerDimension.array() <= 0).any() )
            throw std::invalid_argument("DataDescriptor: non-positive number of coefficients not allowed");

        // pre-compute the partial products for index computations
        for (index_t i = 0; i < _numberOfDimensions; ++i)
            _productOfCoefficientsPerDimension(i) = _numberOfCoefficientsPerDimension.head(i).prod();
    }


    DataDescriptor::DataDescriptor(IndexVector_t numberOfCoefficientsPerDimension, RealVector_t spacingPerDimension)
         : _numberOfDimensions{numberOfCoefficientsPerDimension.size()},
           _numberOfCoefficientsPerDimension{numberOfCoefficientsPerDimension},
           _spacingPerDimension{spacingPerDimension},
           _productOfCoefficientsPerDimension{numberOfCoefficientsPerDimension}
    {
        // sanity checks
        if ( (numberOfCoefficientsPerDimension.array() <= 0).any() )
            throw std::invalid_argument("DataDescriptor: non-positive number of coefficients not allowed");
        if (numberOfCoefficientsPerDimension.size() != spacingPerDimension.size())
            throw std::invalid_argument("DataDescriptor: mismatch between numberOfCoefficientsPerDimension and spacingPerDimension");

        // pre-compute the partial products for index computations
        for (index_t i = 0; i < _numberOfDimensions; ++i)
            _productOfCoefficientsPerDimension(i) = _numberOfCoefficientsPerDimension.head(i).prod();
    }

    index_t DataDescriptor::getNumberOfDimensions() const
    {
        return _numberOfDimensions;
    }

    index_t DataDescriptor::getNumberOfCoefficients() const
    {
        return _numberOfCoefficientsPerDimension.prod();
    }

    IndexVector_t DataDescriptor::getNumberOfCoefficientsPerDimension() const
    {
        return _numberOfCoefficientsPerDimension;
    }

    RealVector_t DataDescriptor::getSpacingPerDimension() const
    {
        return _spacingPerDimension;
    }


    index_t DataDescriptor::getIndexFromCoordinate(elsa::IndexVector_t coordinate) const
    {
        // sanity check
        if (coordinate.size() != _productOfCoefficientsPerDimension.size() )
            throw std::invalid_argument("DataDescriptor: mismatch of coordinate and descriptor size");

        return _productOfCoefficientsPerDimension.cwiseProduct(coordinate).sum();
    }


    IndexVector_t DataDescriptor::getCoordinateFromIndex(elsa::index_t index) const
    {
        // sanity check
        if (index < 0 || index >= getNumberOfCoefficients())
            throw std::invalid_argument("DataDescriptor: invalid index");

        IndexVector_t coordinate(_numberOfDimensions);

        index_t leftOver = index;
        for (index_t i = _numberOfDimensions - 1; i >= 1; --i) {
            coordinate(i) = leftOver / _productOfCoefficientsPerDimension(i);
            leftOver %= _productOfCoefficientsPerDimension(i);
        }
        coordinate(0) = leftOver;

        return coordinate;
    }


    DataDescriptor* DataDescriptor::cloneImpl() const
    {
        return new DataDescriptor(*this);
    }

    bool DataDescriptor::isEqual(const DataDescriptor& other) const
    {
        return (_numberOfDimensions == other._numberOfDimensions) &&
                (_numberOfCoefficientsPerDimension == other._numberOfCoefficientsPerDimension) &&
                (_spacingPerDimension == other._spacingPerDimension);
    }

} // namespace elsa
