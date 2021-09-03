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
    class PSNR
    {
    public:
        static data_t calculate(DataContainer<data_t> x, DataContainer<data_t> y,
                                unsigned int dataRange);

        static data_t calculate(DataContainer<data_t> x, DataContainer<data_t> y);
    };
} // namespace elsa