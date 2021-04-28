#pragma once

#include "Curvelet.h"

// TODO dead class?
namespace elsa
{
    // TODO does inheritance make sense here? decide towards the end of the implementation?
    //  most likely it doesn't make sense
    // TODO decide if it is regular/irregular shearlet
    template <typename data_t = real_t>
    class Shearlet //: public Curvelet<data_t>
    {
    public:
        // image to wavefront?
        // SH: R ^ n^2 -> R ^ J x n x n
        DataContainer<data_t> transform(DataContainer<data_t> image);

        // wavefront to image
        // SH^-1: R ^ J x n x n -> R ^ n^2
        DataContainer<data_t> inverseTransform(DataContainer<data_t> wavefront); // wavefront?

        // NN: R ^ J x n x n -> R ^ J x n x n
    };
} // namespace elsa
