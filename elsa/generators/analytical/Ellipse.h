#pragma once
#include "Image.h"

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

        data_t traceRay(const Ray_t<data_t>& ray) override
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
                return 2 * sqrt(discriminant) * w;
            } else {
                return 0;
            }
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

    public:
        data_t w;
        Eigen::Vector<data_t, n> c;
        Eigen::DiagonalMatrix<data_t, n> A;
        Eigen::Matrix<data_t, n, n> R;
    };

} // namespace elsa::phantoms