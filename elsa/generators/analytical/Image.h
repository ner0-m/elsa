#include "Cloneable.h"
#include "CurvedDetectorDescriptor.h"
#include "DataDescriptor.h"
#include "DetectorDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "TypeCasts.hpp"
#include "elsaDefines.h"
#include "DataContainer.h"
#include <Eigen/src/Core/DiagonalMatrix.h>
#include <Eigen/src/Core/Matrix.h>

#include <Eigen/src/Geometry/Rotation2D.h>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <concepts>

namespace elsa::phantoms
{

    template <typename data_t>
    using Position = Eigen::Vector2<data_t>;

    template <typename data_t>
    class Image : public Cloneable<Image<data_t>>
    {
    public:
        virtual DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor)
        {
            assert(is<CurvedDetectorDescriptor>(sinogramDescriptor));
            assert(sinogramDescriptor.getNumberOfDimensions() == 2);
            return DataContainer<data_t>{sinogramDescriptor};
        }

    protected:
        virtual Image<data_t>* cloneImpl() const = 0;
        virtual bool isEqual(const Image<data_t>& other) const = 0;
    };

    template <typename data_t>
    class Sum : public Image<data_t>
    {
    public:
        Sum(std::unique_ptr<Image<data_t>> a, std::unique_ptr<Image<data_t>> b)
            : a{std::move(a)}, b{std::move(b)}
        {
        }

        DataContainer<data_t> makeSinogram(const DataDescriptor& sinogramDescriptor) override
        {
            assert(is<PlanarDetectorDescriptor>(sinogramDescriptor));
            assert(sinogramDescriptor.getNumberOfDimensions() == 2);

            DataContainer<data_t> sinogram{*sinogramDescriptor.clone()};
            sinogram += a->makeSinogram(sinogramDescriptor);
            sinogram += b->makeSinogram(sinogramDescriptor);
            return sinogram;
        }

        std::unique_ptr<Image<data_t>> a, b;

        virtual Image<data_t>* cloneImpl() const override
        {
            return new Sum<data_t>{a->clone(), b->clone()};
        }
        virtual bool isEqual(const Image<data_t>&) const override { return false; };

    private:
    };

    template <typename data_t>
    Sum<data_t> operator+(const Image<data_t>& a, const Image<data_t>& b)
    {
        return Sum{a.clone(), b.clone()};
    }

    template <typename data_t = float>
    class Ellipse : public Image<data_t>
    {
    public:
        // TODO: Allow multidimensional ellipsoid with matrix
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
        virtual Ellipse<data_t>* cloneImpl() const override { return new Ellipse(*this); }
        virtual bool isEqual(const Image<data_t>& other) const override
        {
            (void) other;
            return false;
        };

    private:
        data_t w;
        Position<data_t> c;
        Eigen::DiagonalMatrix<data_t, 2> A;
        Eigen::Rotation2D<data_t> R;
    };

} // namespace elsa::phantoms
