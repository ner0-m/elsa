#pragma once

#include "elsaDefines.h"

namespace elsa
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
} // namespace elsa
