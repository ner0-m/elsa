#pragma once
#include "Image.h"
#include <Eigen/src/Core/util/Constants.h>

namespace elsa::phantoms
{

    template <int n, typename data_t = float>
    class Ellipsoid : public Image<data_t>
    {
    public:
        Ellipsoid(data_t density, Eigen::Vector<data_t, n> center,
                  Eigen::DiagonalMatrix<data_t, n> A, Eigen::Matrix<data_t, n, n> R)
            : w{density}, c{center}, A{A}, R{R}
        {
        }
        Ellipsoid(data_t density, Eigen::Vector<data_t, n> center, Eigen::Vector<data_t, n> axes,
                  Eigen::Matrix<data_t, n, n> R)
            : w{density}, c{center}, A{axes.array().square().cwiseInverse().matrix()}, R{R}
        {
        }

        DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor) override
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

                auto o = c - ray.origin();
                auto d = ray.direction();

                auto Ro = R * o;
                auto Rd = R * d;

                auto alpha = Ro.dot(A * Ro) - 1;
                auto beta = Rd.dot(A * Ro);
                auto gamma = Rd.dot(A * Rd);
                auto discriminant = beta * beta / (alpha * alpha) - gamma / alpha;

                if (discriminant < 0) {
                    sinogram(coord) = 0;
                } else {
                    sinogram(coord) = 2 * sqrt(discriminant) * w;
                }
            }

            return sinogram;
        }

    protected:
        virtual Ellipsoid<n, data_t>* cloneImpl() const override
        {
            return new Ellipsoid<n, data_t>{*this};
        }
        virtual bool isEqual(const Image<data_t>& other) const override
        {
            if (!is<Ellipsoid<n, data_t>>(other))
                return false;
            const auto& asEllipse = downcast<Ellipsoid<n, data_t>>(other);
            return *this == asEllipse;
        };

    private:
        data_t w;
        Eigen::Vector<data_t, n> c;
        Eigen::DiagonalMatrix<data_t, n> A;
        Eigen::Matrix<data_t, n, n> R;
    };

} // namespace elsa::phantoms