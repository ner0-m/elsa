#include "DataDescriptor.h"

#include <algorithm>

#include "Complex.h"
#include "Error.h"
#include "DataContainer.h"

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

        return detail::coord2Idx(coordinate, _productOfCoefficientsPerDimension);
    }

    IndexVector_t DataDescriptor::getCoordinateFromIndex(elsa::index_t index) const
    {
        // sanity check
        if (index < 0 || index >= getNumberOfCoefficients())
            throw InvalidArgumentError("DataDescriptor: invalid index");

        return detail::idx2Coord(index, _productOfCoefficientsPerDimension);
    }

    template <class data_t>
    DataContainer<data_t> DataDescriptor::element() const
    {
        return DataContainer<data_t>(*this);
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

    IndexVector_t DataDescriptor::getProductOfCoefficientsPerDimension() const
    {
        return _productOfCoefficientsPerDimension;
    }

    // ------------------------------------------
    // explicit template instantiation
    template DataContainer<index_t> DataDescriptor::element() const;
    template DataContainer<float> DataDescriptor::element() const;
    template DataContainer<double> DataDescriptor::element() const;
    template DataContainer<complex<float>> DataDescriptor::element() const;
    template DataContainer<complex<double>> DataDescriptor::element() const;
} // namespace elsa
