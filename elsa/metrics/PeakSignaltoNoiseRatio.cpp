#include "PeakSignaltoNoiseRatio.h"
#include "RelativeError.h"

namespace elsa
{
    template <typename data_t>
    long double PeakSignaltoNoiseRatio<data_t>::calculate(DataContainer<data_t> leftSignal,
                                                          DataContainer<data_t> rightSignal,
                                                          data_t dataRange)
    {
        if (leftSignal.getDataDescriptor() != rightSignal.getDataDescriptor()) {
            throw LogicError(
                std::string("PeakSignaltoNoiseRatio: shapes of both signals should match"));
        }

        long double err = RelativeError<data_t>::calculate(leftSignal, rightSignal);
        return 10 * std::log10((std::pow(dataRange, 2) / err));
    }

    template <typename data_t>
    long double PeakSignaltoNoiseRatio<data_t>::calculate(DataContainer<data_t> leftSignal,
                                                          DataContainer<data_t> rightSignal)
    {
        if (leftSignal.getDataDescriptor() != rightSignal.getDataDescriptor()) {
            throw LogicError(
                std::string("PeakSignaltoNoiseRatio: shapes of both signals should match"));
        }

        data_t dataMin = std::numeric_limits<data_t>::min();
        data_t dataMax = std::numeric_limits<data_t>::max();

        data_t trueMin = std::min(minOfDataContainer(leftSignal), minOfDataContainer(rightSignal));
        data_t trueMax = std::max(maxOfDataContainer(leftSignal), maxOfDataContainer(rightSignal));

        if (trueMin < dataMin || trueMax > dataMax) {
            throw LogicError(std::string("PeakSignaltoNoiseRatio: extreme values of the signals "
                                         "exceed the range expected for its data type"));
        }

        data_t dataRange;
        if (trueMin >= 0) {
            dataRange = dataMax;
        } else {
            dataRange = dataMax - dataMin;
        }

        return calculate(leftSignal, rightSignal, dataRange);
    }

    template <typename data_t>
    data_t PeakSignaltoNoiseRatio<data_t>::maxOfDataContainer(DataContainer<data_t> signal)
    {
        data_t currMax = signal[0];

        for (data_t element : signal) {
            if (element > currMax) {
                currMax = element;
            }
        }

        return currMax;
    }

    template <typename data_t>
    data_t PeakSignaltoNoiseRatio<data_t>::minOfDataContainer(DataContainer<data_t> signal)
    {
        data_t currMin = signal[0];

        for (data_t element : signal) {
            if (element < currMin) {
                currMin = element;
            }
        }

        return currMin;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class PeakSignaltoNoiseRatio<short>;
    template class PeakSignaltoNoiseRatio<unsigned short>;
    template class PeakSignaltoNoiseRatio<int>;
    template class PeakSignaltoNoiseRatio<unsigned int>;
    template class PeakSignaltoNoiseRatio<long>;
    template class PeakSignaltoNoiseRatio<unsigned long>;
    template class PeakSignaltoNoiseRatio<float>;
    template class PeakSignaltoNoiseRatio<double>;
    template class PeakSignaltoNoiseRatio<long double>;
    // TODO consider complex types, do they make sense here? probably not
} // namespace elsa