#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <utility>

namespace elsa
{
    /**
     * @brief Compute the Mean Squared Error of two given signals.
     *
     * @param dc1 DataContainer signal
     * @param dc2 DataContainer signal
     */
    template <typename data_t = real_t>
    constexpr auto meanSquaredError(DataContainer<data_t> dc1, DataContainer<data_t> dc2)
        -> long double
    {
        if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
            throw LogicError(
                std::string("Statistics::meanSquaredError: shapes of both signals should match"));
        }

        DataContainer<data_t> diff = dc1 - dc2;
        return DataContainer{square(diff)}.sum() / dc1.getSize();
    }

    /// Calculate mean and standard deviation of a container
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
        auto sqSum = std::inner_product(std::begin(diff), std::end(diff), std::begin(diff), T());
        auto stdev = std::sqrt(sqSum / mean);

        return std::make_pair(mean, stdev);
    }

    /**
     * @brief Compute the 95% confidence interval for a given number of samples `n`
     * and the mean and standard deviation of the measurements.
     *
     * Compute it as \f$[mean - c(n) * stddev, mean + c(n) * stddev]\f$, where
     * \f$c(n)\$ is the n-th entry in the two tails T distribution table. For \f$n > 30\f$,
     * it is assumed that \f$n = 1.96\f$.
     *
     * @param n Number of of samples
     * @param mean mean of samples
     * @param stddev standard deviation of samples
     * @return pair of lower and upper bound of 95% confidence interval
     */
    template <typename data_t = real_t>
    constexpr auto confidenceInterval95(std::size_t n, data_t mean, data_t stddev)
        -> std::pair<real_t, real_t>
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
            constexpr std::array c = {0.0,      12.70620, 4.302653, 3.182446, 2.776445, 2.570582,
                                      2.446912, 2.364624, 2.306004, 2.262157, 2.228139, 2.200985,
                                      2.178813, 2.160369, 2.144787, 2.131450, 2.119905, 2.109816,
                                      2.100922, 2.093024, 2.085963, 2.079614, 2.073873, 2.068658,
                                      2.063899, 2.059539, 2.055529, 2.051831, 2.048407, 2.045230};

            const auto degreeOfFreedome = n - 1;
            const auto lower = mean - c[degreeOfFreedome] * stddev;
            const auto upper = mean + c[degreeOfFreedome] * stddev;
            return std::make_pair(lower, upper);
        }
    }
} // namespace elsa
