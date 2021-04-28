#pragma once

#include "DataContainer.h"
#include "Cloneable.h"

// TODO dead class?
namespace elsa
{
    // TODO does it make sense for Wavelet to be superclass of Curvelet? and the latter of Shearlet?
    //  if yes then what methods should be defined here?
    template <typename data_t = real_t>
    class Wavelet //: public Cloneable<data_t> //???
    {
    };
} // namespace elsa
