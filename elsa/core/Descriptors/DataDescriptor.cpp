#include "DataDescriptor.h"

#include <algorithm>

#include "Error.h"

namespace elsa
{
    DataDescriptor::DataDescriptor(IndexVector_t numberOfCoefficientsPerDimension)
        : _numberOfDimensions{numberOfCoefficientsPerDimension.size()},
          _numberOfCoefficientsPerDimension{numberOfCoefficientsPerDimension},
          _spacingPerDimension{RealVector_t::Ones(_numberOfDimensions)},
          _productOfCoefficientsPerDimension{computeStrides(numberOfCoefficientsPerDimension)}
    {
        // sanity checks
        if ((numberOfCoefficientsPerDimension.array() <= 0).any())
            throw InvalidArgumentError(
                "DataDescriptor: non-positive number of coefficients not allowed");

        // set the origin at center
        _locationOfOrigin = static_cast<real_t>(0.5)
                            * (_numberOfCoefficientsPerDimension.cast<real_t>().array()
                               * _spacingPerDimension.array());
    }

    DataDescriptor::DataDescriptor(IndexVector_t numberOfCoefficientsPerDimension,
                                   RealVector_t spacingPerDimension)
        : _numberOfDimensions{numberOfCoefficientsPerDimension.size()},
          _numberOfCoefficientsPerDimension{numberOfCoefficientsPerDimension},
          _spacingPerDimension{spacingPerDimension},
          _productOfCoefficientsPerDimension{computeStrides(numberOfCoefficientsPerDimension)}
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

    IndexVector_t DataDescriptor::getStrides() const
    {
        return _productOfCoefficientsPerDimension;
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
        if (coordinate.size() != getStrides().size())
            throw InvalidArgumentError(
                "DataDescriptor: mismatch of coordinate and descriptor size");

        return ravelIndex(coordinate, getStrides());
    }

    IndexVector_t DataDescriptor::getCoordinateFromIndex(elsa::index_t index) const
    {
        // sanity check
        if (index < 0 || index >= getNumberOfCoefficients())
            throw InvalidArgumentError("DataDescriptor: invalid index");

        return unravelIndex(index, getStrides());
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

    IndexVector_t computeStrides(const IndexVector_t& shape) noexcept
    {
        IndexVector_t strides(shape.size());

        /// TODO: this is just a scan
        for (index_t i = 0; i < shape.size(); ++i) {
            strides(i) = shape.head(i).prod();
        }

        return strides;
    }

    index_t ravelIndex(const IndexVector_t& coord, const IndexVector_t& strides) noexcept
    {
        return strides.cwiseProduct(coord).sum();
    }

    IndexVector_t unravelIndex(const index_t& index, const IndexVector_t& strides) noexcept
    {
        const auto dim = strides.size();
        IndexVector_t coordinate(dim);

        index_t leftOver = index;
        for (index_t i = dim - 1; i >= 1; --i) {
            coordinate(i) = leftOver / strides(i);
            leftOver %= strides(i);
        }
        coordinate(0) = leftOver;

        return coordinate;
    }

} // namespace elsa
