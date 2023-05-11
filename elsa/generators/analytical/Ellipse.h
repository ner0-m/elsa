#pragma once
#include "Image.h"

namespace elsa::phantoms
{

    template <typename data_t = float>
    class Ellipse : public Image<data_t>
    {
    public:
        Ellipse(data_t density, Position<data_t> center, data_t a, data_t b, data_t phi = 0)
            : w{density}, c{center}, A{1 / (a * a), 1 / (b * b)}, R{phi}
        {
        }

        DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor) override
        {
            assert(is<DetectorDescriptor>(sinogramDescriptor));
            assert(sinogramDescriptor.getNumberOfDimensions() == 2);

            DataContainer<data_t> sinogram{sinogramDescriptor};
            auto& detDesc = downcast<DetectorDescriptor>(sinogramDescriptor);

            for (index_t pose = 0; pose < detDesc.getNumberOfGeometryPoses(); pose++) {
                for (index_t pixel = 0; pixel < detDesc.getNumberOfCoefficientsPerDimension()[0];
                     pixel++) {

                    auto ray = detDesc.computeRayFromDetectorCoord(IndexVector_t{{pixel, pose}});

                    auto o = c - ray.origin();
                    auto d = ray.direction();

                    auto Ro = R * o;
                    auto Rd = R * d;

                    auto alpha = Ro.dot(A * Ro) - 1;
                    auto beta = Rd.dot(A * Ro);
                    auto gamma = Rd.dot(A * Rd);
                    auto discriminant = beta * beta / (alpha * alpha) - gamma / alpha;

                    if (discriminant < 0) {
                        sinogram(IndexVector_t{{pixel, pose}}) = 0;
                    } else {
                        sinogram(IndexVector_t{{pixel, pose}}) = 2 * sqrt(discriminant) * w;
                    }
                }
            }

            return sinogram;
        }

    protected:
        virtual Ellipse<data_t>* cloneImpl() const override { return new Ellipse{*this}; }
        virtual bool isEqual(const Image<data_t>& other) const override
        {
            if (!is<Ellipse<data_t>>(other))
                return false;
            const auto& asEllipse = downcast<Ellipse<data_t>>(other);
            return *this == asEllipse;
        };

    private:
        data_t w;
        Position<data_t> c;
        Eigen::DiagonalMatrix<data_t, 2> A;
        Eigen::Rotation2D<data_t> R;
    };

} // namespace elsa::phantoms