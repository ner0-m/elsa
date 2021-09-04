#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * @brief Class representing the Relative Error (RE)
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the values of the operator, defaulting to real_t
     *
     * References:
     */
    template <typename data_t = real_t>
    class RelativeError
    {
    public:
        static long double calculate(DataContainer<data_t> leftSignal,
                                     DataContainer<data_t> rightSignal);
    };
} // namespace elsa