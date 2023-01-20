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

    template <typename data_t>
    DataContainer<data_t> forbildHead(IndexVector_t sizes);

    template <typename data_t>
    DataContainer<data_t> forbildAbdomen(IndexVector_t sizes);

    template <typename data_t>
    DataContainer<data_t> forbildThorax(IndexVector_t sizes);

    template <typename data_t = real_t>
    DataContainer<data_t> smoothBlob(IndexVector_t sizes, double radius_manipulation = 1.0);

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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
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
             //   A     a       b      c      x0     y0       z0      phi   theta  psi
             //  --------------------------------------------------------------------------
             {{  1.0f,  .6900f, .920f, .810f, 0.00f, 0.f,     0.f,     0.f,  0.f,  0.f   }},
             {{  -.8f,  .6624f, .874f, .780f, 0.00f, -.0184f, 0.f,     0.f,  0.f,  0.f   }},
             {{  -.2f,  .1100f, .310f, .220f,  .22f, 0.f,     0.f,   -18.f,  0.f, 10.f   }},
             {{  -.2f,  .1600f, .410f, .280f, -.22f, 0.f,     0.f,    18.f,  0.f, 10.f   }},
             {{   .1f,  .2100f, .250f, .410f, 0.00f,  .35f,   -.15f,   0.f,  0.f,  0.f   }},
             {{   .1f,  .0460f, .046f, .050f, 0.00f,  .1f,     .25f,   0.f,  0.f,  0.f   }},
             {{   .1f,  .0460f, .046f, .050f, 0.00f, -.1f,     .25f,   0.f,  0.f,  0.f   }},
             {{   .1f,  .0460f, .023f, .050f, -.08f, -.605f,  0.f,     0.f,  0.f,  0.f   }},
             {{   .1f,  .0230f, .023f, .020f, 0.00f, -.606f,  0.f,     0.f,  0.f,  0.f   }},
             {{   .1f,  .0230f, .046f, .020f,  .06f, -.605f,  0.f,     0.f,  0.f,  0.f   }},
        // clang-format on
    }};
#pragma GCC diagnostic pop

    // explicit template instantiation
    template const std::array<std::array<double, 10>, 10> modifiedSheppLoganParameters<double>;
    template const std::array<std::array<float, 10>, 10> modifiedSheppLoganParameters<float>;

    namespace old
    {

        /// short-hand for 3d vector
        using Vec3 = Eigen::Matrix<index_t, 3, 1>;
        /**
         *
         * @deprecated Is a reference for testing - @see elsa::phantoms::Ellipsoid
         *
         * @brief Draw a rotated, filled 3d ellipsoid
         *
         * @param[in,out] dc the DataContainer where the ellipsoid should be drawn in
         * @param[in] amplitude the "color" of the ellipsoid and its filling
         * @param[in] center the 3d index of where to place the center of the ellipsoid in dc
         * @param[in] sizes the radii (in x/y/z) of the ellipsoid
         * @param[in] phi euler angle of rotation of the ellipsoid
         * @param[in] theta euler angle of rotation of the ellipsoid
         * @param[in] psi euler angle of rotation of the ellipsoid
         *
         *
         * Warning: this method is currently using an inefficient, cubic algorithm. Thus it's very
         * slow for big volumes!
         *
         */
        template <typename data_t = real_t>
        [[deprecated]] static void drawFilledEllipsoid3d(DataContainer<data_t>& dc,
                                                         data_t amplitude, Vec3 center, Vec3 sizes,
                                                         data_t phi, data_t theta, data_t psi);

        /**
         * @deprecated @see elsa::phantoms::modifiedSheppLogan
         *
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
        [[deprecated]] DataContainer<data_t> modifiedSheppLogan(IndexVector_t sizes);

    } // namespace old
} // namespace elsa::phantoms
