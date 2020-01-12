#include "DataDescriptor.h"

#include <stdexcept>
#include <algorithm>

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
            throw std::invalid_argument(
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
            throw std::invalid_argument(
                "DataDescriptor: non-positive number of coefficients not allowed");
        if (numberOfCoefficientsPerDimension.size() != spacingPerDimension.size())
            throw std::invalid_argument("DataDescriptor: mismatch between "
                                        "numberOfCoefficientsPerDimension and spacingPerDimension");

        // set the origin at center
        _locationOfOrigin = static_cast<real_t>(0.5)
                            * (_numberOfCoefficientsPerDimension.cast<real_t>().array()
                               * _spacingPerDimension.array());

        // pre-compute the partial products for index computations
        for (index_t i = 0; i < _numberOfDimensions; ++i)
            _productOfCoefficientsPerDimension(i) =
                _numberOfCoefficientsPerDimension.head(i).prod();
    }

    index_t DataDescriptor::getNumberOfDimensions() const { return _numberOfDimensions; }

    index_t DataDescriptor::getNumberOfCoefficients() const
    {
        return _numberOfCoefficientsPerDimension.prod();
    }

    IndexVector_t DataDescriptor::getNumberOfCoefficientsPerDimension() const
    {
        return _numberOfCoefficientsPerDimension;
    }

    RealVector_t DataDescriptor::getSpacingPerDimension() const { return _spacingPerDimension; }

    RealVector_t DataDescriptor::getLocationOfOrigin() const { return _locationOfOrigin; }

    index_t DataDescriptor::getIndexFromCoordinate(elsa::IndexVector_t coordinate) const
    {
        // sanity check
        if (coordinate.size() != _productOfCoefficientsPerDimension.size())
            throw std::invalid_argument(
                "DataDescriptor: mismatch of coordinate and descriptor size");

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

    std::unique_ptr<DataDescriptor>
        DataDescriptor::bestCommon(const std::vector<const DataDescriptor*>& descList)
    {
        if (descList.empty())
            throw std::invalid_argument("DataDescriptor::bestCommon: descriptor list empty");

        const auto& firstDesc = *descList[0];
        auto coeffs = firstDesc.getNumberOfCoefficientsPerDimension();
        auto size = firstDesc.getNumberOfCoefficients();
        auto spacing = firstDesc.getSpacingPerDimension();

        bool allSame =
            std::all_of(descList.begin(), descList.end(),
                        [&firstDesc](const DataDescriptor* d) { return *d == firstDesc; });
        if (allSame)
            return firstDesc.clone();

        bool allSameCoeffs =
            std::all_of(descList.begin(), descList.end(), [&coeffs](const DataDescriptor* d) {
                return d->getNumberOfCoefficientsPerDimension().size() == coeffs.size()
                       && d->getNumberOfCoefficientsPerDimension() == coeffs;
            });

        if (allSameCoeffs) {
            bool allSameSpacing =
                std::all_of(descList.begin(), descList.end(), [&spacing](const DataDescriptor* d) {
                    return d->getSpacingPerDimension() == spacing;
                });
            if (allSameSpacing) {
                return std::make_unique<DataDescriptor>(coeffs, spacing);
            } else {
                return std::make_unique<DataDescriptor>(coeffs);
            }
        }

        bool allSameSize =
            std::all_of(descList.begin(), descList.end(), [size](const DataDescriptor* d) {
                return d->getNumberOfCoefficients() == size;
            });

        if (!allSameSize)
            throw std::invalid_argument(
                "DataDescriptor::bestCommon: descriptor sizes do not match");

        return std::make_unique<DataDescriptor>(IndexVector_t::Constant(1, size));
    }

    DataDescriptor* DataDescriptor::cloneImpl() const { return new DataDescriptor(*this); }

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
