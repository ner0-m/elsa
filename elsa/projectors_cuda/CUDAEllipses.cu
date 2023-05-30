#include "CUDAEllipses.h"
#include "DataContainer.h"
#include "elsaDefines.h"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/ParametrizedLine.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

namespace elsa::phantoms
{

    template <typename data_t, int n>
    data_t __device__ CUDAEllipses<data_t, n>::Functor::operator()(const data_t& I,
                                                                   const Ray& ray) const
    {
        auto o = c - ray.origin();
        auto d = ray.direction();

        auto Ro = R * o;
        auto Rd = R * d;

        auto alpha = Ro.dot(A * Ro) - 1;
        auto beta = Rd.dot(A * Ro);
        auto gamma = Rd.dot(A * Rd);
        auto discriminant = beta * beta / (alpha * alpha) - gamma / alpha;

        if (discriminant >= 0) {
            return I + 2 * sqrt(discriminant) * w;
        } else {
            return I;
        }
    }

    template <typename data_t, int n>
    DataContainer<data_t>
        CUDAEllipses<data_t, n>::makeSinogram(const DataDescriptor& sinogramDescriptor)
    {
        assert(is<DetectorDescriptor>(sinogramDescriptor));
        assert(sinogramDescriptor.getNumberOfDimensions() == 2
               || sinogramDescriptor.getNumberOfDimensions() == 3);

        auto& detDesc = downcast<DetectorDescriptor>(sinogramDescriptor);

        thrust::host_vector<Ray> hRays{detDesc.getNumberOfCoefficients()};

#pragma omp parallel for
        for (index_t index = 0; index < detDesc.getNumberOfCoefficients(); index++) {

            auto coord = detDesc.getCoordinateFromIndex(index);
            auto ray = detDesc.computeRayFromDetectorCoord(coord).cast<data_t>();

            Eigen::Vector<data_t, n> origin = ray.origin();
            Eigen::Vector<data_t, n> direction = ray.direction();

            hRays[index] = {origin, direction};
        }

        thrust::device_vector<Ray> dRays = hRays;
        thrust::device_vector<data_t> dSino{detDesc.getNumberOfCoefficients(), 0.0};

        for (auto& ellipse : components) {
            thrust::transform(dSino.begin(), dSino.end(), dRays.begin(), dSino.begin(), ellipse);
        }

        DataContainer<data_t> sinogram{detDesc};
        sinogram.storage() = dSino;
        return sinogram;
    }

} // namespace elsa::phantoms
