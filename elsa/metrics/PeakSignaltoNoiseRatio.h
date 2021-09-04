#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * @brief Class representing the Peak Signal-to-Noise Ratio (PSNR)
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the values of the operator, defaulting to real_t
     *
     * References:
     */
    template <typename data_t = real_t>
    class PeakSignaltoNoiseRatio
    {
    public:
        static long double calculate(DataContainer<data_t> leftSignal,
                                     DataContainer<data_t> rightSignal, data_t dataRange);

        static long double calculate(DataContainer<data_t> leftSignal,
                                     DataContainer<data_t> rightSignal);

    private:
        // TODO remove me (this is probably already implemented somewhere)
        static data_t maxOfDataContainer(DataContainer<data_t> signal);

        // TODO remove me (this is probably already implemented somewhere)
        static data_t minOfDataContainer(DataContainer<data_t> signal);
    };
} // namespace elsa