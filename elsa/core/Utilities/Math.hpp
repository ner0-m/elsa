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
            return (k > n)                  ? 0
                   : (k == 0 || k == n)     ? 1
                   : (k == 1 || k == n - 1) ? n
                   : (k + k < n)            ? (binom(n - 1, k - 1) * n) / k
                                            : (binom(n - 1, k) * n) / (n - k);
        }

        /**
         * Compute Heaviside-function
         * \f[
         * x \mapsto
         * \begin{cases}
         * 0: & x < 0 \\
         * c: & x = 0 \\
         * 1: & x > 0
         * \end{cases}
         * \f]
         */
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

        template <typename T>
        constexpr inline int sgn(T val)
        {
            return (T(0) < val) - (val < T(0));
        }

        template <typename data_t>
        data_t lerp(data_t a, SelfType_t<data_t> b, SelfType_t<data_t> t)
        {
            if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0))
                return t * b + (1 - t) * a;

            if (t == 1)
                return b;

            const data_t x = a + t * (b - a);

            if ((t > 1) == (b > a))
                return b < x ? x : b;
            else
                return x < b ? x : b;
        }
    } // namespace math

    /// proposed in Y. Meyer, Oscillating Patterns in Image Processing and Nonlinear Evolution
    /// Equations. AMS, 2001
    template <typename data_t>
    data_t meyerFunction(data_t x)
    {
        if (x < 0.f) {
            return 0;
        } else if (0.f <= x && x <= 1.f) {
            return 35 * std::pow(x, 4.f) - 84 * std::pow(x, 5.f) + 70 * std::pow(x, 6.f)
                   - 20 * std::pow(x, 7.f);
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
                return std::sin(pi<data_t> / 2.f * meyerFunction(std::abs(w) - 1));
            } else if (2 < std::abs(w) && std::abs(w) <= 4) {
                return std::cos(pi<data_t> / 2.f * meyerFunction(1.f / 2.f * std::abs(w) - 1.f));
            } else {
                return 0;
            }
        }

        /// defined in Sören Häuser and Gabriele Steidl, Fast Finite Shearlet Transform: a
        /// tutorial, 2014
        template <typename data_t>
        data_t phi(data_t w)
        {
            if (std::abs(w) <= 1.f / 2) {
                return 1;
            } else if (1.f / 2 < std::abs(w) && std::abs(w) < 1) {
                return std::cos(pi<data_t> / 2.f * meyerFunction(2.f * std::abs(w) - 1));
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
            return std::sqrt(std::pow(b(2.f * w), 2.f) + std::pow(b(w), 2.f));
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

    namespace axdt
    {
        template <typename data_t>
        data_t ratio_of_factorials(index_t x, index_t y)
        {
            data_t ratio = 1.0;
            if (x == y) {
                return ratio;
            } else if (x > y) {
                for (index_t i = y + 1; i <= x; ++i) {
                    ratio *= static_cast<data_t>(i);
                }
                return ratio;
            } else {
                return static_cast<data_t>(1.0) / ratio_of_factorials<data_t>(y, x);
            }
        }

        template <typename data_t>
        data_t double_factorial(index_t x)
        {
            data_t df = 1.0;
            while (x > 1) {
                df *= static_cast<data_t>(x);
                x -= 2;
            }
            return df;
        }

        template <typename data_t>
        Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> assoc_legendre_pol(index_t l,
                                                                                 data_t x)
        {
            if (x < -1 || x > 1) {
                throw std::invalid_argument("math::axdt::assoc_legendre_pol: Can only evaluate "
                                            "polynomials at x in the interval [-1,1].");
            }

            Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> result =
                Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(l + 1, l + 1);

            result(0, 0) = static_cast<data_t>(1);

            for (int i = 1; i <= l; ++i) {
                for (int j = 0; j <= i - 2; ++j) {
                    result(i, j) = ((2 * static_cast<data_t>(i) - 1) * x * result(i - 1, j)
                                    - static_cast<data_t>(i + j - 1) * result(i - 2, j))
                                   / static_cast<data_t>(i - j);
                }
                result(i, i - 1) = x * static_cast<data_t>(2 * i - 1) * result(i - 1, i - 1);
                result(i, i) = static_cast<data_t>(pow(-1, i) * double_factorial<data_t>(2 * i - 1)
                                                   * pow(1 - x * x, i / 2.0));
            }

            return result;
        }

        template <typename data_t>
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> SH_basis_real(index_t l, data_t theta, data_t phi)
        {

            Eigen::Matrix<data_t, Eigen::Dynamic, 1> result((l + 1) * (l + 1));

            Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> legendre_pol =
                assoc_legendre_pol(l, static_cast<data_t>(std::cos(theta)));

            for (int i = 0; i <= l; ++i) {
                auto c = static_cast<data_t>(sqrt((2 * i + 1) / (4.0 * pi_t)));
                for (int j = -i; j <= i; ++j) {
                    if (j < 0) {
                        result(i * i + j + i) = static_cast<data_t>(
                            sqrt(2) * pow(-1, j) * c
                            * sqrt(ratio_of_factorials<data_t>(i + j, i - j)) * legendre_pol(i, -j)
                            * sin(static_cast<data_t>(-j) * phi));
                    } else if (j == 0) {
                        result(i * i + j + i) = c * legendre_pol(i, j);
                    } else {
                        result(i * i + j + i) = static_cast<data_t>(
                            sqrt(2) * pow(-1, j) * c
                            * sqrt(ratio_of_factorials<data_t>(i - j, i + j)) * legendre_pol(i, j)
                            * cos(static_cast<data_t>(j) * phi));
                    }
                }
            }

            return result;
        }
    } // namespace axdt

    /// @brief Compute the sign of the given value. Will return -1, for negative values, 1 for
    /// positive ones and 0 otherwise
    template <typename T, typename Ret = int>
    Ret sign(T val)
    {
        return static_cast<Ret>((T{0} < val) - (val < T{0}));
    }

} // namespace elsa
