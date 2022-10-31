#include "DataDescriptor.h"

#include <algorithm>

#include "Error.h"

namespace elsa
{
    DataDescriptor::DataDescriptor(IndexVector_t numberOfCoefficientsPerDimension)
        : _numberOfDimensions{numberOfCoefficientsPerDimension.size()},
          _numberOfCoefficientsPerDimension{numberOfCoefficientsPerDimension},
          _spacingPerDimension{RealVector_t::Ones(_numberOfDimensions)},
          _productOfCoefficientsPerDimension{numberOfCoefficientsPerDimension}
    {
        // sanity checks
        if ((numberOfCoefficientsPerDimension.array() <= 0).any())
            throw InvalidArgumentError(
                "DataDescriptor: non-positive number of coefficients not allowed");

        // set the origin at center
        _locationOfOrigin = static_cast<real_t>(0.5)
                            * (_numberOfCoefficientsPerDimension.cast<real_t>().array()
                               * _spacingPerDimension.array());

        // pre-compute the partial products for index computations
        for (index_t i = 0; i < _numberOfDimensions; ++i)
            _productOfCoefficientsPerDimension(i) =
                _numberOfCoefficientsPerDimension.head(i).prod();
    }

    DataDescriptor::DataDescriptor(IndexVector_t numberOfCoefficientsPerDimension,
                                   RealVector_t spacingPerDimension)
        : _numberOfDimensions{numberOfCoefficientsPerDimension.size()},
          _numberOfCoefficientsPerDimension{numberOfCoefficientsPerDimension},
          _spacingPerDimension{spacingPerDimension},
          _productOfCoefficientsPerDimension{numberOfCoefficientsPerDimension}
    {
        // sanity checks
        if ((numberOfCoefficientsPerDimension.array() <= 0).any())
            throw InvalidArgumentError(
                "DataDescriptor: non-positive number of coefficients not allowed");
        if (numberOfCoefficientsPerDimension.size() != spacingPerDimension.size())
            throw InvalidArgumentError("DataDescriptor: mismatch between "
                                       "numberOfCoefficientsPerDimension and spacingPerDimension");
        if ((spacingPerDimension.array() < 0).any())
            throw InvalidArgumentError("DataDescriptor: non-positive spacing not allowed");

        // set the origin at center
        _locationOfOrigin = static_cast<real_t>(0.5)
                            * (_numberOfCoefficientsPerDimension.cast<real_t>().array()
                               * _spacingPerDimension.array());

        // pre-compute the partial products for index computations
        for (index_t i = 0; i < _numberOfDimensions; ++i)
            _productOfCoefficientsPerDimension(i) =
                _numberOfCoefficientsPerDimension.head(i).prod();
    }

    DataDescriptor::~DataDescriptor() {}

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

    RealVector_t DataDescriptor::getLocationOfOrigin() const
    {
        return _locationOfOrigin;
    }

    index_t DataDescriptor::getIndexFromCoordinate(const elsa::IndexVector_t& coordinate) const
    {
        // sanity check
        if (coordinate.size() != _productOfCoefficientsPerDimension.size())
            throw InvalidArgumentError(
                "DataDescriptor: mismatch of coordinate and descriptor size");

        return _productOfCoefficientsPerDimension.cwiseProduct(coordinate).sum();
    }

    IndexVector_t DataDescriptor::getCoordinateFromIndex(elsa::index_t index) const
    {
        // sanity check
        if (index < 0 || index >= getNumberOfCoefficients())
            throw InvalidArgumentError("DataDescriptor: invalid index");

        IndexVector_t coordinate(_numberOfDimensions);

        index_t leftOver = index;
        for (index_t i = _numberOfDimensions - 1; i >= 1; --i) {
            coordinate(i) = leftOver / _productOfCoefficientsPerDimension(i);
            leftOver %= _productOfCoefficientsPerDimension(i);
        }
        coordinate(0) = leftOver;

        return coordinate;
    }

    bool DataDescriptor::isEqual(const DataDescriptor& other) const
    {
        if (typeid(other) != typeid(*this))
            return false;

        return (_numberOfDimensions == other._numberOfDimensions)
               && (_numberOfCoefficientsPerDimension == other._numberOfCoefficientsPerDimension)
               && (_spacingPerDimension == other._spacingPerDimension)
               && (_locationOfOrigin == other._locationOfOrigin);
    }

} // namespace elsa
