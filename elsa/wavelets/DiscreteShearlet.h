#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

namespace elsa
{
    template <typename data_t = real_t>
    class DiscreteShearlet //: Shearlet<data_t> // TODO inheritance makes more sense here?
    {
        // image to wavefront?
        // SH: R ^ n^2 -> R ^ J x n x n
        DataContainer<data_t> transform(DataContainer<data_t> image, int j, int k,
                                        std::vector<int> m)
        {
            // dot product of the image and psi(j, k, m)
        }

        // wavefront to image
        // SH^-1: R ^ J x n x n -> R ^ 2
        DataContainer<data_t> inverseTransform(DataContainer<data_t> wavefront); // wavefront?

        // scale index j, the orientation index k, and the position index m.
        DataContainer<data_t> psi(int j, int k, std::vector<int> m)
        { // m should be of size 2
          // here we could have e.g. 2^(3/4)j * psi(SkA2^j · −m)
        }

        // NN: R ^ J x n x n -> R ^ J x n x n
    };
} // namespace elsa
