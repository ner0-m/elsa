#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

#include <numeric>
#include <cmath>
#include <utility>

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

    namespace statistics
    {
        /**
         * @brief Compute the Mean Squared Error of two given signals.
         *
         * Calculate it based on the formula of @f$ \frac{1}{n} \sum_{i=1}^n (x_{i} - y_{i})^{2}
         * @f$.
         *
         * @param dc1 DataContainer signal
         * @param dc2 DataContainer signal
         *
         * @author Andi Braimllari - initial code
         *
         * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
         */
        template <typename data_t = real_t>
        data_t meanSquaredError(DataContainer<data_t> dc1, DataContainer<data_t> dc2)
        {
            if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
                throw InvalidArgumentError(
                    "Statistics::meanSquaredError: shapes of both signals should match");
            }

            DataContainer<data_t> diff = dc1 - dc2;
            return diff.squaredL2Norm() / dc1.getSize();
        }

        /// @brief Calculate mean and standard deviation of a container
        /// @param v Any container, such as `std::vector`
        /// @tparam Container Container type of argument (e.g. `std::vector`)
        /// @return a pair of mean and standard deviation (of type `Container::value_type`)
        template <typename Container>
        constexpr auto calculateMeanStddev(Container v)
            -> std::pair<typename Container::value_type, typename Container::value_type>
        {
            // value type of the vector
            using T = typename Container::value_type;

            // Sum all elements and divide by the size to get mean
            auto sum = std::accumulate(std::begin(v), std::end(v), T());
            auto mean = sum / static_cast<T>(std::size(v));

            // New vector with x_i - mean entries
            std::vector<T> diff(v.size());
            std::transform(std::begin(v), std::end(v), std::begin(diff),
                           [mean](auto x) { return x - mean; });

            // Sum of product of each element
            auto sqSum =
                std::inner_product(std::begin(diff), std::end(diff), std::begin(diff), T());
            auto stdev = std::sqrt(sqSum / static_cast<T>(std::size(v)));

            return std::make_pair(mean, stdev);
        }

        /**
         * @brief Compute the 95% confidence interval for a given number of samples `n`
         * and the mean and standard deviation of the measurements.
         *
         * Compute it as \f$mean - c(n) * stddev, mean + c(n) * stddev\f$, where
         * \f$c(n)\f$ is the n-th entry in the two tails T distribution table. For \f$n > 30\f$,
         * it is assumed that \f$n = 1.96\f$.
         *
         * @param n Number of of samples
         * @param mean mean of samples
         * @param stddev standard deviation of samples
         * @return pair of lower and upper bound of 95% confidence interval
         */
        template <typename data_t = real_t>
        std::pair<real_t, real_t> confidenceInterval95(std::size_t n, data_t mean, data_t stddev)
        {
            // Do we run often enough to assume a large sample size?
            if (n > 30) {
                // 1.96 is a magic number for 95% confidence intervals, equivalent to c in the other
                // branch
                const auto lower = mean - 1.96 * stddev;
                const auto upper = mean + 1.96 * stddev;
                return std::make_pair(lower, upper);
            } else {
                // t Table for 95% with offset to handle 1 iteration
                // In that case the mean is the lower and upper bound
                constexpr std::array c = {
                    0.0,      12.70620, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912, 2.364624,
                    2.306004, 2.262157, 2.228139, 2.200985, 2.178813, 2.160369, 2.144787, 2.131450,
                    2.119905, 2.109816, 2.100922, 2.093024, 2.085963, 2.079614, 2.073873, 2.068658,
                    2.063899, 2.059539, 2.055529, 2.051831, 2.048407, 2.045230};

                const auto degreeOfFreedome = n - 1;
                const auto lower = mean - c[degreeOfFreedome] * stddev;
                const auto upper = mean + c[degreeOfFreedome] * stddev;
                return std::make_pair(lower, upper);
            }
        }

        /**
         * @brief Compute the Relative Error between two given signals.
         *
         * Calculate it based on the formula of @f$ \| x - y \|_{2} / \| y \|_{2} @f$.
         *
         * @param dc1 DataContainer signal
         * @param dc2 DataContainer signal
         *
         * @author Andi Braimllari - initial code
         *
         * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
         */
        template <typename data_t = real_t>
        data_t relativeError(DataContainer<data_t> dc1, DataContainer<data_t> dc2)
        {
            if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
                throw InvalidArgumentError(
                    "statistics::relativeError: shapes of both signals should match");
            }

            DataContainer<data_t> diff = dc1 - dc2;
            return diff.l2Norm() / dc2.l2Norm();
        }

        /**
         * @brief Compute the Peak Signal-to-Noise Ratio of a given signal S.
         *
         * Calculate it based on the formula of @f$ 20 * log_{10}(MAX_{I}) - 10 * log_{10}(MSE) @f$
         * in which @f$ MAX_{I} @f$ is the maximum possible pixel value of the image and @f$ MSE @f$
         * is the mean squared error.
         *
         * @param dc1 DataContainer signal
         * @param dc2 DataContainer signal
         *
         * @author Andi Braimllari - initial code
         *
         * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
         */
        template <typename data_t = real_t>
        data_t peakSignalToNoiseRatio(DataContainer<data_t> dc1, DataContainer<data_t> dc2)
        {
            if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
                throw InvalidArgumentError(
                    "statistics::peakSignalToNoiseRatio: shapes of both signals should match");
            }

            return 20 * std::log10(dc1.maxElement()) - 10 * std::log10(meanSquaredError(dc1, dc2));
        }
    } // namespace statistics

    namespace shearlet
    {
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
