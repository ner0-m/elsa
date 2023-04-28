#pragma once

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"
#include "PartitionDescriptor.h"
#include "Scaling.h"
#include "TypeTraits.hpp"
#include "VolumeDescriptor.h"
#include "elsaDefines.h"
#include <cstdlib>
#include <functional>

namespace elsa
{

    template <typename data_t = float>
    using Filter = Scaling<add_complex_t<data_t>>;

    /**
     * @brief Generates filter form transfer function in k-Space
     *
     * @tparam data_t underlying floating point type
     * @param descriptor Data descriptor describing filter dimensions
     * @param transferFunction Transfer function lambda receiving a wavenumber vector
     * @return Filter<data_t> Scaling with DC bin at position 0
     */
    template <typename data_t = float>
    Filter<data_t> makeFilter(const DataDescriptor& descriptor,
                              std::function<add_complex_t<data_t>(IndexVector_t)> transferFunction)
    {

        auto dim = descriptor.getNumberOfDimensions();
        auto coefficients = descriptor.getNumberOfCoefficientsPerDimension();
        coefficients[dim - 1] = 1;

        auto sliceDesc = VolumeDescriptor{coefficients};
        auto midPoint = coefficients / 2;

        DataContainer<complex_t> filter{sliceDesc};

        if (!(dim == 2 || dim == 3)) {
            throw InvalidArgumentError("makeFilter:: Should only be used for 2D or 3D sinograms");
        }

        for (index_t i = 0; i < sliceDesc.getNumberOfCoefficients(); ++i) {
            IndexVector_t nu = sliceDesc.getCoordinateFromIndex(i) - midPoint;
            auto H = transferFunction(nu);
            filter[i] = H;
        }

        return Filter<data_t>{sliceDesc, ifftShift(filter)};
    }

    template <typename data_t = float>
    data_t kMax(const DataDescriptor& descriptor)
    {
        return 2 * pi_t
               * (descriptor.getNumberOfCoefficientsPerDimension().head(
                      descriptor.getNumberOfDimensions() - 1)
                  / 2)
                     .norm();
    }

    // TODO: Non-mononotonic filter function start to increase again for 2d filters (k_max  *=
    // sqrt(2)?)

    // TODO: skimage computes spectrum of space representation to "lessen artifacts" -> investigate
    template <typename data_t = float>
    Filter<data_t> makeRamLak(const DataDescriptor& descriptor)
    {
        auto n = descriptor.getNumberOfCoefficientsPerDimension()[0] / 2;
        auto kmax = kMax(descriptor);
        return makeFilter(descriptor, [&](IndexVector_t nu) {
            auto deltak = 2 * pi_t / n; // TODO: DC bin might have different factor in 3d
            auto k = 2 * pi_t * nu.norm();
            return (nu.isZero() ? 0.25 * deltak : k) / kmax;
        });
    }

    template <typename data_t = float>
    Filter<data_t> makeSheppLogan(const DataDescriptor& descriptor)
    {
        auto kmax = kMax(descriptor);
        return makeFilter(descriptor, [&](IndexVector_t nu) {
            auto k = 2 * pi_t * nu.norm();
            return std::sin(pi_t / 2 * k / kmax) * 2
                   / pi_t; // TODO: Look up definition of sinc (factor 1/pi?)
        });
    }

    template <typename data_t = float>
    Filter<data_t> makeCosine(const DataDescriptor& descriptor)
    {
        auto kmax = kMax(descriptor);
        return makeFilter(descriptor, [&](IndexVector_t nu) {
            auto k = 2 * pi_t * nu.norm();
            return std::sin(pi_t * k / kmax) / pi_t;
        });
    }

    template <typename data_t = float>
    Filter<data_t> makeHann(const DataDescriptor& descriptor)
    {
        auto kmax = kMax(descriptor);
        return makeFilter(descriptor, [&](IndexVector_t nu) {
            auto k = 2 * pi_t * nu.norm();
            return k / kmax * std::cos(pi_t / 2 * k / kmax) * std::cos(pi_t / 2 * k / kmax);
        });
    }

} // namespace elsa
