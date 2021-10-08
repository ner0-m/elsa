#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "Statistics.hpp"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <utility>

namespace elsa
{
    /**
     * @brief Compute the Relative Error between two given signals.
     *
     * @param dc1 DataContainer signal
     * @param dc2 DataContainer signal
     */
    template <typename data_t = real_t>
    constexpr auto relativeError(DataContainer<data_t> dc1, DataContainer<data_t> dc2)
        -> long double
    {
        if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
            throw LogicError(
                std::string("Metrics::relativeError: shapes of both signals should match"));
        }

        DataContainer<data_t> diff = dc1 - dc2;
        return diff.l2Norm() / dc2.l2Norm();
    }

    /**
     * @brief Compute the Peak Signal-to-Noise Ratio.
     *
     * @param dc1 DataContainer signal
     * @param dc2 DataContainer signal
     * @param dataRange The data range of the signals (distance between minimum and maximum possible
     * values).
     */
    template <typename data_t = real_t>
    constexpr auto peakSignalToNoiseRatio(DataContainer<data_t> dc1, DataContainer<data_t> dc2,
                                          data_t dataRange) -> long double
    {
        if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
            throw LogicError(std::string(
                "Metrics::peakSignalToNoiseRatio: shapes of both signals should match"));
        }

        long double err = meanSquaredError<data_t>(dc1, dc2);
        return 10 * std::log10((std::pow(dataRange, 2) / err));
    }

    /**
     * @brief Compute the Peak Signal-to-Noise Ratio.
     *
     * @param dc1 DataContainer signal
     * @param dc2 DataContainer signal
     */
    template <typename data_t = real_t>
    constexpr auto peakSignalToNoiseRatio(DataContainer<data_t> dc1, DataContainer<data_t> dc2)
        -> long double
    {
        if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
            throw LogicError(std::string(
                "Metrics::peakSignalToNoiseRatio: shapes of both signals should match"));
        }

        data_t dataMin = std::numeric_limits<data_t>::min();
        data_t dataMax = std::numeric_limits<data_t>::max();

        data_t trueMin = std::min(minOfDataContainer(dc1), minOfDataContainer(dc2));
        data_t trueMax = std::max(maxOfDataContainer(dc1), maxOfDataContainer(dc2));

        if (trueMin < dataMin || trueMax > dataMax) {
            throw LogicError(
                std::string("Metrics::peakSignalToNoiseRatio: extreme values of the signals "
                            "exceed the range expected for its data type"));
        }

        data_t dataRange;
        if (trueMin >= 0) {
            dataRange = dataMax;
        } else {
            dataRange = dataMax - dataMin;
        }

        return peakSignalToNoiseRatio(dc1, dc2, dataRange);
    }

    // TODO remove me as soon as possible and implement me as a reduction in DataContainer
    template <typename data_t>
    data_t maxOfDataContainer(DataContainer<data_t> signal)
    {
        data_t currMax = signal[0];

        for (data_t element : signal) {
            if (element > currMax) {
                currMax = element;
            }
        }

        return currMax;
    }

    // TODO remove me as soon as possible and implement me as a reduction in DataContainer
    template <typename data_t>
    data_t minOfDataContainer(DataContainer<data_t> signal)
    {
        data_t currMin = signal[0];

        for (data_t element : signal) {
            if (element < currMin) {
                currMin = element;
            }
        }

        return currMin;
    }
} // namespace elsa
