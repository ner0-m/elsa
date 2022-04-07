#pragma once

#include "elsaDefines.h"

namespace elsa
{
    namespace math
    {
        /// Compute factorial \f$n!\f$ recursively
        constexpr inline index_t factorial(index_t n) noexcept
        {
            return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
        }

        /// Compute binomial coefficient
        constexpr inline index_t binom(index_t n, index_t k) noexcept
        {
            return (k > n)
                       ? 0
                       : (k == 0 || k == n) ? 1
                                            : (k == 1 || k == n - 1)
                                                  ? n
                                                  : (k + k < n) ? (binom(n - 1, k - 1) * n) / k
                                                                : (binom(n - 1, k) * n) / (n - k);
        }

        /// Compute Heaviside-function
        /// \f[
        /// x \mapsto
        /// \begin{cases}
        /// 0: & x < 0 \\
        /// c: & x = 0 \\
        /// 1: & x > 0
        /// \end{cases}
        /// \f]
        template <typename data_t>
        constexpr data_t heaviside(data_t x1, data_t c)
        {
            if (x1 == 0) {
                return c;
            } else if (x1 < 0) {
                return 0;
            } else {
                return 1;
            }
        }
    } // namespace math

    /// proposed in Y. Meyer, Oscillating Patterns in Image Processing and Nonlinear Evolution
    /// Equations. AMS, 2001
    template <typename data_t>
    data_t meyerFunction(data_t x)
    {
        if (x < 0) {
            return 0;
        } else if (0 <= x && x <= 1) {
            return 35 * std::pow(x, 4) - 84 * std::pow(x, 5) + 70 * std::pow(x, 6)
                   - 20 * std::pow(x, 7);
        } else {
            return 1;
        }
    }

    namespace shearlet
    {
        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        template <typename data_t>
        data_t b(data_t w)
        {
            if (1 <= std::abs(w) && std::abs(w) <= 2) {
                return std::sin(pi<data_t> / 2.0 * meyerFunction(std::abs(w) - 1));
            } else if (2 < std::abs(w) && std::abs(w) <= 4) {
                return std::cos(pi<data_t> / 2.0 * meyerFunction(1.0 / 2 * std::abs(w) - 1));
            } else {
                return 0;
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        template <typename data_t>
        data_t phi(data_t w)
        {
            if (std::abs(w) <= 1.0 / 2) {
                return 1;
            } else if (1.0 / 2 < std::abs(w) && std::abs(w) < 1) {
                return std::cos(pi<data_t> / 2.0 * meyerFunction(2 * std::abs(w) - 1));
            } else {
                return 0;
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        template <typename data_t>
        data_t phiHat(data_t w, data_t h)
        {
            if (std::abs(h) <= std::abs(w)) {
                return phi(w);
            } else {
                return phi(h);
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        template <typename data_t>
        data_t psiHat1(data_t w)
        {
            return std::sqrt(std::pow(b(2 * w), 2) + std::pow(b(w), 2));
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        template <typename data_t>
        data_t psiHat2(data_t w)
        {
            if (w <= 0) {
                return std::sqrt(meyerFunction(1 + w));
            } else {
                return std::sqrt(meyerFunction(1 - w));
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        template <typename data_t>
        data_t psiHat(data_t w, data_t h)
        {
            if (w == 0) {
                return 0;
            } else {
                return psiHat1(w) * psiHat2(h / w);
            }
        }
    } // namespace shearlet
} // namespace elsa
