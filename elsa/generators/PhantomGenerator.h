#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * @brief This class generates 2d/3d phantoms, like the Shepp-Logan phantom.
     *
     * @author Maximilian Hornung - initial code
     * @author David Frank - fixes and modifications
     * @author Tobias Lasser - rewrite
     *
     * @tparam data_t data type for the DataContainer, defaulting to real_t
     */
    template <typename data_t = real_t>
    class PhantomGenerator
    {
    public:
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
        static DataContainer<data_t> createModifiedSheppLogan(IndexVector_t sizes);

        /**
         * @brief Create a phantom with a simple n-dimensional rectangle  going from lower to upper.
         * It is assumed that lower < upper.
         *
         * @param[in] volumesize size of the volume
         * @param[in] lower the lower corner of the rectangle
         * @param[in] upper the upper corner of the rectangle
         */
        static DataContainer<data_t> createRectanglePhantom(IndexVector_t volumesize,
                                                            IndexVector_t lower,
                                                            IndexVector_t upper);

        /**
         * @brief Create a phantom with a simple n-dimensional sphere centered in the middle with
         * given raidus
         *
         * @param[in] volumesize size of the volume
         * @param[in] radius the radius of the circle
         */
        static DataContainer<data_t> createCirclePhantom(IndexVector_t volumesize, data_t radius);

    private:
        /// scale sizes from [0,1] to the (square) phantom size, producing indices (integers)
        static index_t scale(const DataDescriptor& dd, data_t value);

        /// scale and shift center coordinates to the (square) phantom size, producing indices
        /// (integers)
        static index_t scaleShift(const DataDescriptor& dd, data_t value);
    };
} // namespace elsa
