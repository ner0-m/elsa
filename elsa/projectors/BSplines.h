#pragma once

#include "elsaDefines.h"
#include "Assertions.h"
#include "Math.hpp"
#include "Luts.hpp"

#include <cmath>

namespace elsa
{
    namespace bspline
    {
        /// @brief Evaluate the 1-dimensional B-Spline of degree m, with m + 2 equally spaced knots
        ///
        /// This is based on the implementation given as in e.g.:
        /// Fast B-spline transforms for continuous image representation and interpolation, Unser
        /// et. al. (1991), equation 2.2
        ///
        /// @param x coordinate to evaluate 1-dimensional B-Spline at
        /// @param m order of B-Spline
        template <typename data_t>
        constexpr data_t bspline1d_evaluate(data_t x, index_t m) noexcept
        {
            const auto xs = x + static_cast<data_t>(m + 1) / data_t{2};
            data_t res = 0;

            for (int n = 0; n <= m + 1; ++n) {
                data_t nf = static_cast<data_t>(n);
                const auto tmp1 = math::heaviside<data_t>(xs - nf, 0);
                const auto tmp2 = std::pow<data_t>(xs - nf, m);
                const auto tmp3 = static_cast<data_t>(math::binom(m + 1, n));
                const auto tmp4 = std::pow<data_t>(-1, n) / static_cast<data_t>(math::factorial(m));
                res += static_cast<data_t>(tmp1 * tmp2 * tmp3 * tmp4);
            }

            return res;
        }

        /// @brief Evaluate the 1-dimensional B-Spline Derivative of degree m
        ///
        /// @param x coordinate to evaluate 1-dimensional B-Spline at
        /// @param m order of B-Spline
        template <typename data_t>
        constexpr data_t bsplineDerivative1d_evaluate(data_t x, index_t m) noexcept
        {
            return bspline1d_evaluate(x + 0.5, m - 1) - bspline1d_evaluate(x - 0.5, m - 1);
        }

        /// @brief Evaluate n-dimensional B-Spline of degree m. As B-Splines are separable, this is
        /// just the product of 1-dimensional B-Splines.
        ///
        /// @param x n-dimensional coordinate to evaluate B-Spline at
        /// @param m order of B-Spline
        template <typename data_t>
        constexpr data_t nd_bspline_evaluate(const Vector_t<data_t>& x, index_t m) noexcept
        {
            data_t res = bspline1d_evaluate(x[0], m);
            for (int i = 1; i < x.size(); ++i) {
                res *= bspline1d_evaluate(x[i], m);
            }
            return res;
        }

        /// @brief Evaluate n-dimensional B-Spline at a given coordinate for the first dimension,
        /// and at the center (i.e. `0`) at all the other dimensions. This is particular useful as
        /// an approximation during the calculation of the line integral
        ///
        /// @param x 1-dimensional coordinate to evaluate B-Spline at
        /// @param m order of B-Spline
        /// @param dim dimension of B-Spline
        template <typename data_t>
        constexpr data_t nd_bspline_centered(data_t x, index_t m, index_t dim) noexcept
        {
            data_t res = bspline1d_evaluate<data_t>(x, m);
            for (int i = 1; i < dim; ++i) {
                const auto inc = bspline1d_evaluate<data_t>(0.f, m);
                res *= inc;
            }
            return res;
        }

        /// @brief Evaluate the derivative of an n-dimensional B-Spline at a given coordinate for
        /// the first dimension, and at the center (i.e. `0`) at all the other dimensions. This is
        /// particular useful as an approximation during the calculation of the line integral
        ///
        /// @param x 1-dimensional coordinate to evaluate B-Spline at
        /// @param m order of B-Spline
        /// @param dim dimension of B-Spline
        template <typename data_t>
        constexpr data_t nd_bspline_derivative_centered(data_t x, int m, int dim) noexcept
        {
            data_t res = bsplineDerivative1d_evaluate(x, m);
            for (int i = 1; i < dim; ++i) {
                const auto inc = bspline1d_evaluate(0., m);
                res *= inc;
            }
            return res;
        }

        const index_t DEFAULT_ORDER = 2;
    } // namespace bspline

    /// @brief Represent a B-Spline basis function of a given dimension and order
    template <typename data_t>
    class BSpline
    {
    public:
        BSpline(index_t dim, index_t order);

        data_t operator()(Vector_t<data_t> x);

        data_t derivative(data_t s);

        index_t order() const;

        data_t radius() const;

        index_t dim() const;

    private:
        /// Dimension of B-Spline
        index_t dim_;

        /// Order of B-Spline
        index_t order_;
    };

    /// @brief Represent a projected B-Spline basis function of a given dimension and order.
    /// Projected B-Splines are again B-Spline of n-1 dimensions. Using the fact, that B-Splines
    /// are close to symmetrical, we can approximate the projection only based on distance.
    template <typename data_t, size_t N = DEFAULT_PROJECTOR_LUT_SIZE>
    class ProjectedBSpline
    {
    public:
        constexpr ProjectedBSpline(index_t dim, index_t order)
            : dim_(dim),
              order_(order),
              radius_((order + 1) * 0.5),
              lut_([this](data_t s) { return this->operator()<true>(s); }, radius_),
              derivative_lut_(
                  [this](data_t s) { return order_ > 0 ? this->derivative<true>(s) : 0; }, radius_),
              normalized_gradient_lut_(
                  [this](data_t s) { return order_ > 0 ? this->normalized_gradient(s) : 0; },
                  radius_)
        {
        }

        template <bool accurate = false>
        constexpr data_t operator()(data_t x) const
        {
            if constexpr (accurate)
                return bspline::nd_bspline_centered(x, order_, dim_ - 1);
            else
                return lut_(std::abs(x));
        }

        template <bool accurate = false>
        constexpr data_t derivative(data_t x) const
        {
            if constexpr (accurate)
                return bspline::nd_bspline_derivative_centered(x, order_, dim_ - 1);
            else
                return derivative_lut_(std::abs(x)) * math::sgn(x);
        }

        constexpr data_t normalized_gradient(data_t x) const
        {
            // compute f'(x)/x
            if (x == 0)
                x = 1e-10;
            return bspline::nd_bspline_derivative_centered(x, order_, dim_ - 1) / x;
        }

        constexpr index_t dim() const { return dim_; }

        constexpr index_t order() const { return order_; }

        constexpr data_t radius() const { return radius_; }

        constexpr const Lut<data_t, N>& get_lut() const { return lut_; }

        constexpr const Lut<data_t, N>& get_derivative_lut() const { return derivative_lut_; }

        constexpr const Lut<data_t, N>& get_normalized_gradient_lut() const
        {
            return normalized_gradient_lut_;
        }

    private:
        /// Dimension of B-Spline
        const index_t dim_;

        /// Order of B-Spline
        const index_t order_;

        /// Radius of B-Spline
        const data_t radius_;

        /// LUT for projected B-Spline
        const Lut<data_t, N> lut_;

        /// LUT for projected derivative
        const Lut<data_t, N> derivative_lut_;

        /// LUT for projected normalized gradient
        const Lut<data_t, N> normalized_gradient_lut_;
    };
} // namespace elsa
