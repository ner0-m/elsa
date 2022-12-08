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

    namespace axdt {
        template <typename data_t>
        data_t ratio_of_factorials(index_t x,index_t y) {
            data_t ratio = 1.0;
            if (x == y) {
                return ratio;
            }
            else if(x > y) {
                for (index_t i = y + 1; i <= x; ++i) {
                    ratio *= static_cast<data_t>(i);
                }
                return ratio;
            }
            else {
                return 1.0 / ratio_of_factorials<data_t>(y,x);
            }
        }

        template <typename data_t>
        data_t double_factorial(index_t x) {
            data_t df = 1.0;
            while (x > 1) {
                df *= x;
                x -= 2;
            }
            return df;
        }

        template <typename data_t>
        Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> assoc_legendre_pol(index_t l, data_t x)
        {
            if (x < -1 || x > 1){
                throw std::invalid_argument("math::axdt::assoc_legendre_pol: Can only evaluate polynomials at x in the interval [-1,1].");
            }

            Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> result = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(l + 1, l + 1);

            result(0, 0) = static_cast<data_t>(1);

            for (int i = 1; i <= l; ++i) {
                for (int j = 0; j <= i - 2; ++j) {
                    result(i, j) = ((2*i - 1)*x*result(i - 1, j) - (i + j - 1)*result(i - 2, j))/(i - j);
                }
                result(i, i - 1) = x*(2*i - 1)*result(i - 1, i - 1);
                result(i, i) = pow(-1, i)*double_factorial<data_t>(2*i - 1)*pow(1 - x*x, i/2.0);
            }

            return result;
        }

        template <typename data_t>
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> SH_basis_real(index_t l, data_t theta, data_t phi) {

            Eigen::Matrix<data_t, Eigen::Dynamic, 1> result((l + 1)*(l + 1));

            Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> legendre_pol = assoc_legendre_pol(l, static_cast<data_t>(std::cos(theta)));

            for (int i = 0;i <= l; ++i) {
                data_t c = sqrt((2*i + 1) / (4.0*pi_t));
                for (int j = -i; j <= i; ++j) {
                    if (j < 0) {
                        result(i*i + j + i) = sqrt(2)*pow(-1, j)*c*sqrt(ratio_of_factorials<data_t>(i + j,i - j))*legendre_pol(i, -j)*sin(-j*phi);
                    }
                    else if (j==0) {
                        result(i*i + j + i) = c*legendre_pol(i, j);
                    }
                    else {
                        result(i*i + j + i) = sqrt(2)*pow(-1, j)*c*sqrt(ratio_of_factorials<data_t>(i-j,i+j))*legendre_pol(i,j)*cos(j*phi);
                    }
                }
            }

            return result;
        }

        template <typename data_t>
        Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> rotation_matrix(data_t theta, data_t psi, data_t phi)
        {
            const auto c0 = static_cast<data_t>(std::cos(theta));
            const auto c1 = static_cast<data_t>(std::cos(psi));
            const auto c2 = static_cast<data_t>(std::cos(phi));
            const auto s0 = static_cast<data_t>(std::sin(theta));
            const auto s1 = static_cast<data_t>(std::sin(psi));
            const auto s2 = static_cast<data_t>(std::sin(phi));

            Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> R(3,3);
            R <<
                -s1 * s2 + c0 * c1 * c2, -s0 * c1, s1 * c2 + c0 * c1 * s2,
                s0 * c2,                 c0,       s0 * s2,
                -c1 * s2 - c0 * s1 * c2, s0 * s1,  c1 * c2 - c0 * s1 * s2;

            return R;
        }
    } // namespace axdt
} // namespace elsa
