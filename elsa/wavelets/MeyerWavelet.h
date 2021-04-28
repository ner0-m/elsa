#pragma once

#include "Wavelet.h"

// TODO might be used as a generating function in shearlet systems
namespace elsa
{
    // TODO what methods should be defined here?
    template <typename data_t = real_t>
    class MeyerWavelet : public Wavelet<data_t>
    {
    };
} // namespace elsa
