#pragma once
#include "Cloneable.h"
#include "DataDescriptor.h"
#include "DetectorDescriptor.h"
#include "TypeCasts.hpp"
#include "elsaDefines.h"
#include "DataContainer.h"

#include <memory>
#include <vector>

namespace elsa::phantoms
{

    template <typename data_t>
    class Image : public Cloneable<Image<data_t>>
    {
    public:
        DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor);
        virtual data_t traceRay(const Ray_t<data_t>& ray) = 0;

        virtual Image<data_t>* cloneImpl() const = 0;

    protected:
        virtual bool isEqual(const Image<data_t>& other) const = 0;
    };

    template <typename data_t>
    DataContainer<data_t> Image<data_t>::makeSinogram(const DataDescriptor& sinogramDescriptor)
    {
        assert(is<DetectorDescriptor>(sinogramDescriptor));
        assert(sinogramDescriptor.getNumberOfDimensions() == 2
               || sinogramDescriptor.getNumberOfDimensions() == 3);

        DataContainer<data_t> sinogram{sinogramDescriptor};
        auto& detDesc = downcast<DetectorDescriptor>(sinogramDescriptor);

#pragma omp parallel for
        for (index_t index = 0; index < detDesc.getNumberOfCoefficients(); index++) {

            auto coord = detDesc.getCoordinateFromIndex(index);
            auto ray = detDesc.computeRayFromDetectorCoord(coord);
            sinogram[index] = traceRay(ray);
        }
        return sinogram;
    }

} // namespace elsa::phantoms
