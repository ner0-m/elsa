#pragma once

#include "elsaDefines.h"
#include "Assertions.h"
#include "Math.hpp"

#include <cmath>
#include "spdlog/fmt/bundled/core.h"
#include "spdlog/fmt/fmt.h"

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
            const auto xs = x + (m + 1) / data_t{2};
            data_t res = 0;

            for (int n = 0; n <= m + 1; ++n) {
                const auto tmp1 = math::heaviside<data_t>(xs - n, 0);
                const auto tmp2 = std::pow<data_t>(xs - n, m);
                const auto tmp3 = math::binom(m + 1, n);
                const auto tmp4 = std::pow<data_t>(-1, n) / math::factorial(m);
                res += tmp1 * tmp2 * tmp3 * tmp4;
            }

            return res;
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
        constexpr data_t nd_bspline_centered(data_t x, int m, int dim) noexcept
        {
            data_t res = bspline1d_evaluate<data_t>(x, m);
            for (int i = 1; i < dim; ++i) {
                const auto inc = bspline1d_evaluate<data_t>(0., m);
                res *= inc;
            }
            return res;
        }
    } // namespace bspline

    /// @brief Represent a B-Spline basis function of a given dimension and order
    template <typename data_t>
    class BSpline
    {
    public:
        BSpline(index_t dim, index_t order);

        data_t operator()(Vector_t<data_t> x);

        index_t order() const;

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
    template <typename data_t>
    class ProjectedBSpline
    {
    public:
        ProjectedBSpline(index_t dim, index_t order);

        data_t operator()(data_t s);

        index_t order() const;

        index_t dim() const;

    private:
        /// Dimension of B-Spline
        index_t dim_;

        /// Order of B-Spline
        index_t order_;
    };
} // namespace elsa
