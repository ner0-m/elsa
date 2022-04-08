#include "Statistics.h"

namespace elsa
{
    template <typename data_t>
    data_t Statistics<data_t>::meanSquaredError(DataContainer<data_t> dc1,
                                                DataContainer<data_t> dc2)
    {
        if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
            throw LogicError(
                std::string("Statistics::meanSquaredError: shapes of both signals should match"));
        }

        DataContainer<data_t> diff = dc1 - dc2;
        return diff.squaredL2Norm() / dc1.getSize();
    }

    template <typename data_t>
    template <typename Container>
    constexpr auto Statistics<data_t>::calculateMeanStddev(Container v)
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

    template <typename data_t>
    std::pair<real_t, real_t> Statistics<data_t>::confidenceInterval95(std::size_t n, data_t mean,
                                                                       data_t stddev)
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

    template <typename data_t>
    data_t Statistics<data_t>::relativeError(DataContainer<data_t> dc1, DataContainer<data_t> dc2)
    {
        if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
            throw InvalidArgumentError(
                std::string("Metrics::relativeError: shapes of both signals should match"));
        }

        DataContainer<data_t> diff = dc1 - dc2;
        return diff.l2Norm() / dc2.l2Norm();
    }

    template <typename data_t>
    data_t Statistics<data_t>::peakSignalToNoiseRatio(DataContainer<data_t> dc1,
                                                      DataContainer<data_t> dc2, data_t dataRange)
    {
        if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
            throw InvalidArgumentError(std::string(
                "Metrics::peakSignalToNoiseRatio: shapes of both signals should match"));
        }

        return data_t(10) * std::log10((std::pow(dataRange, 2) / meanSquaredError(dc1, dc2)));
    }

    template <typename data_t>
    data_t Statistics<data_t>::peakSignalToNoiseRatio(DataContainer<data_t> dc1,
                                                      DataContainer<data_t> dc2)
    {
        if (dc1.getDataDescriptor() != dc2.getDataDescriptor()) {
            throw InvalidArgumentError(std::string(
                "Metrics::peakSignalToNoiseRatio: shapes of both signals should match"));
        }

        data_t dataMin = std::numeric_limits<data_t>::min();
        data_t dataMax = std::numeric_limits<data_t>::max();

        data_t trueMin = std::min(dc1.minElement(), dc2.minElement());
        data_t trueMax = std::max(dc1.maxElement(), dc2.maxElement());

        if (trueMin < dataMin || trueMax > dataMax) {
            throw InvalidArgumentError(
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

    // ------------------------------------------
    // explicit template instantiation
    template class Statistics<float>;
    template class Statistics<double>;
} // namespace elsa
