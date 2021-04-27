#pragma once

#include "Wavelet.h"
#include "elsaDefines.h"
//#include "Cloneable.h"

namespace elsa
{
    template <typename data_t = real_t>
    class HaarWavelet: public Wavelet<data_t>
    {
    public:
        int waveletFunction(real_t t)
        {
            if (t >= 0 && t < 1 / 2) {
                return 1;
            } else if (t >= 1 / 2 && t < 1) {
                return -1;
            } else {
                return 0;
            }
        }

        int scalingFunction(real_t t)
        {
            if (t >= 0 && t < 1) {
                return 1;
            } else {
                return 0;
            }
        }
    };
} // namespace elsa