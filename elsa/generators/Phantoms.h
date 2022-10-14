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
} // namespace elsa::phantoms
