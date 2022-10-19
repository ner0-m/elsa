#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

namespace elsa::phantoms
{
    /**
     * @brief Create a modified Shepp-Logan phantom in 2d or 3d (with enhanced contrast).
     *
     * The phantom specifications are adapted from Matlab (which in turn references A.K. Jain,
     * "Fundamentals of Digital Image Processing", p. 439, and P.A. Toft, "The Radon Transform,
     * Theory and Implementation", p. 199).
     *
     * Warning: the 3D version is currently very inefficient to compute (cubic algorithm).
     *
     * @param[in] sizes a 2d/3d vector indicating the requested size (has to be square!)
     *
     * @returns DataContainer of specified size containing the phantom.
     */
    template <typename data_t = real_t>
    DataContainer<data_t> modifiedSheppLogan(IndexVector_t sizes);

    /**
     * @brief Create a phantom with a simple n-dimensional rectangle  going from lower to upper.
     * It is assumed that lower < upper.
     *
     * @param[in] volumesize size of the volume
     * @param[in] lower the lower corner of the rectangle
     * @param[in] upper the upper corner of the rectangle
     */
    template <typename data_t = real_t>
    DataContainer<data_t> rectangle(IndexVector_t volumesize, IndexVector_t lower,
                                    IndexVector_t upper);

    /**
     * @brief Create a phantom with a simple n-dimensional sphere centered in the middle with
     * given raidus
     *
     * @param[in] volumesize size of the volume
     * @param[in] radius the radius of the circle
     */
    template <typename data_t = real_t>
    DataContainer<data_t> circular(IndexVector_t volumesize, data_t radius);

    /**
     * @brief  https://de.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom
     *         This head phantom is the same as the Shepp-Logan except
     *         the intensities are changed to yield higher contrast in
     *         the image.  Taken from Toft, 199-200.
     *
     */
    template <typename data_t, typename = std::enable_if_t<std::is_floating_point<data_t>::value>>
    inline constexpr std::array<std::array<data_t, 10>, 10> modifiedSheppLoganParameters{{
        // clang-format off
             //
             // A: amplitude
             // a: width according to x axis
             // b: width according to y axis
             // c: width according to z axis
             // x0,y0,z0: center coordinates
             // phi:    rotation around x axis
             // theta:  rotation around y axis
             // psi:    rotation arount z axis
             //
             //   A       a     b     c   x0      y0      z0    phi  theta  psi
             //  -----------------------------------------------------------------
             {{   1,   .6900, .920, .810,  0,     0,      0,      0,   0,   0    }},
             {{   -.8, .6624, .874, .780,  0,    -.0184,  0,      0,   0,   0    }},
             {{   -.2, .1100, .310, .220,  .22,   0,      0,    -18,   0,   10   }},
             {{   -.2, .1600, .410, .280, -.22,   0,      0,     18,   0,   10   }},
             {{   .1,  .2100, .250, .410,  0,     .35,   -.15,    0,   0,   0    }},
             {{   .1,  .0460, .046, .050,  0,     .1,     .25,    0,   0,   0    }},
             {{   .1,  .0460, .046, .050,  0,    -.1,     .25,    0,   0,   0    }},
             {{   .1,  .0460, .023, .050,  -.08, -.605,   0,      0,   0,   0    }},
             {{   .1,  .0230, .023, .020,  0,    -.606,   0,      0,   0,   0    }},
             {{   .1,  .0230, .046, .020,  .06,  -.605,   0,      0,   0,   0    }}
        // clang-format on
    }};

    // explicit template instantiation
    template const std::array<std::array<double, 10>, 10> modifiedSheppLoganParameters<double>;
    template const std::array<std::array<float, 10>, 10> modifiedSheppLoganParameters<float>;

} // namespace elsa::phantoms
