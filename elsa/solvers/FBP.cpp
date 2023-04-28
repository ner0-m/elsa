#include "FBP.h"
#include "DataContainer.h"
#include "Filter.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include "TypeTraits.hpp"
#include "elsaDefines.h"
#include "PartitionDescriptor.h"
#include <Eigen/src/Core/util/Constants.h>
#include <cstring>

namespace elsa
{
    // TODO: tests
    template <typename data_t>
    FBP<data_t>::FBP(const LinearOperator<data_t>& P, const Filter<data_t>& g)
        : projector_{P.clone()}, filter_{downcast<Filter<data_t>>(g.clone())}
    {
    }

    template <typename data_t, bool forward = true>
    DataContainer<add_complex_t<data_t>> sliceWiseFFT(const DataContainer<data_t>& signal)
    {
        auto& desc = signal.getDataDescriptor();
        auto dim = desc.getNumberOfDimensions();
        auto sizeOfLastDim = desc.getNumberOfCoefficientsPerDimension()[dim - 1];

        DataContainer<add_complex_t<data_t>> spectrum{desc};

        if constexpr (isComplex<data_t>) {
            spectrum = signal;
        } else {
            spectrum = signal.asComplex();
        }

        for (auto i = 0; i < sizeOfLastDim; i++) {
            auto slice = spectrum.slice(i);
            auto F = FourierTransform<data_t>{slice.getDataDescriptor()};
            if constexpr (forward) {
                spectrum.slice(i) = F.apply(slice);
            } else {
                spectrum.slice(i) = F.applyAdjoint(slice);
            }
        }

        return spectrum;
    }

    template <typename data_t>
    DataContainer<add_complex_t<data_t>> sliceWiseIFFT(const DataContainer<data_t>& signal)
    {
        return sliceWiseFFT<data_t, false>(signal);
    }

    template <typename data_t>
    DataContainer<add_complex_t<data_t>> applyRowWise(const DataContainer<data_t>& signal,
                                                      const std::unique_ptr<Filter<data_t>>& filter)
    {
        auto& desc = signal.getDataDescriptor();
        auto dim = desc.getNumberOfDimensions();
        auto sizeOfLastDim = desc.getNumberOfCoefficientsPerDimension()[dim - 1];

        DataContainer<add_complex_t<data_t>> filtered{desc};

        for (auto i = 0; i < sizeOfLastDim; i++) {
            auto& slice = signal.slice(i);
            filtered.slice(i) = filter->apply(slice);
        }

        return filtered;
    }

    // TODO: pad sinogram to power of two as in skimage?
    template <typename data_t>
    DataContainer<data_t> FBP<data_t>::apply(const DataContainer<data_t>& sinogram) const
    {
        auto& descriptor = sinogram.getDataDescriptor();

        if (!(descriptor.getNumberOfDimensions() == 2 || descriptor.getNumberOfDimensions())) {
            throw InvalidArgumentError("FBP.apply:: Can only handle [2,3]D data");
        }
        auto c = sinogram.asComplex();
        auto Fb = sliceWiseFFT(c);
        auto filtered = applyRowWise(Fb, filter_);
        auto b_prime = sliceWiseIFFT(filtered);
        auto backprojected = projector_->applyAdjoint(elsa::real(b_prime));

        auto numSlices =
            descriptor
                .getNumberOfCoefficientsPerDimension()[descriptor.getNumberOfDimensions() - 1];

        return backprojected * pi_t / 2 / numSlices; // This normalization is necessary because the
                                                     // projectors are not normalized
    }

    // ------------------------------------------
    // explicit template instantiation
    template class FBP<float>;
    template class FBP<double>;

} // namespace elsa
