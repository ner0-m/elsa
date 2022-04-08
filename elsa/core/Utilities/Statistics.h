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
     * @brief Class representing ...
     *
     * This class represents an ...
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * References:
     */
    template <typename data_t = real_t>
    class Statistics
    {
        /**
         * @brief Compute the Mean Squared Error of two given signals.
         *
         * @param dc1 DataContainer signal
         * @param dc2 DataContainer signal
         */
        data_t meanSquaredError(DataContainer<data_t> dc1, DataContainer<data_t> dc2);

        /// @brief Calculate mean and standard deviation of a container
        /// @param v Any container, such as `std::vector`
        /// @tparam Container Container type of argument (e.g. `std::vector`)
        /// @return a pair of mean and standard deviation (of type `Container::value_type`)
        template <typename Container>
        constexpr auto calculateMeanStddev(Container v)
            -> std::pair<typename Container::value_type, typename Container::value_type>;

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
        std::pair<real_t, real_t> confidenceInterval95(std::size_t n, data_t mean, data_t stddev);

        /**
         * @brief Compute the Relative Error between two given signals.
         *
         * @param dc1 DataContainer signal
         * @param dc2 DataContainer signal
         */
        data_t relativeError(DataContainer<data_t> dc1, DataContainer<data_t> dc2);

        /**
         * @brief Compute the Peak Signal-to-Noise Ratio.
         *
         * @param dc1 DataContainer signal
         * @param dc2 DataContainer signal
         * @param dataRange The data range of the signals (distance between minimum and maximum
         * possible values).
         */
        data_t peakSignalToNoiseRatio(DataContainer<data_t> dc1, DataContainer<data_t> dc2,
                                              data_t dataRange);

        /**
         * @brief Compute the Peak Signal-to-Noise Ratio.
         *
         * @param dc1 DataContainer signal
         * @param dc2 DataContainer signal
         */
        data_t peakSignalToNoiseRatio(DataContainer<data_t> dc1, DataContainer<data_t> dc2);
    };
} // namespace elsa
