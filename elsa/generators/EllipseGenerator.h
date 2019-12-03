#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * \brief This class draws 2d and 3d rotated ellipses into DataContainers.
     *
     * \author Tobias Lasser - initial code
     *
     * \tparam data_t data type for the DataContainer, defaulting to real_t
     *
     * Drawing is done by adding a given constant to the appropriate places in the 2d/3d image.
     */
    template <typename data_t = real_t>
    class EllipseGenerator
    {
    public:
        /// short-hand for 2d vector
        using Vec2 = Eigen::Matrix<index_t, 2, 1>;
        /// short-hand for 3d vector
        using Vec3 = Eigen::Matrix<index_t, 3, 1>;

        /**
         * \brief Draw a rotated, filled 2d ellipse
         *
         * \param[in,out] dc the DataContainer where the ellipse should be drawn in
         * \param[in] amplitude the "color" of the ellipse and its filling
         * \param[in] center the 2d index of where to place the center of the ellipse in dc
         * \param[in] sizes the radii (horizontal/vertical) of the ellipse
         * \param[in] angle rotation angle of the ellipse in degrees (with respect to y axis)
         */
        static void drawFilledEllipse2d(DataContainer<data_t>& dc, data_t amplitude, Vec2 center,
                                        Vec2 sizes, data_t angle);

        /**
         * \brief Draw a rotated, filled 3d ellipsoid
         *
         * \param[in,out] dc the DataContainer where the ellipsoid should be drawn in
         * \param[in] amplitude the "color" of the ellipsoid and its filling
         * \param[in] center the 3d index of where to place the center of the ellipsoid in dc
         * \param[in] sizes the radii (in x/y/z) of the ellipsoid
         * \param[in] phi euler angle of rotation of the ellipsoid
         * \param[in] theta euler angle of rotation of the ellipsoid
         * \param[in] psi euler angle of rotation of the ellipsoid
         *
         * Warning: this method is currently using an inefficient, cubic algorithm. Thus it's very
         * slow for big volumes!
         */
        static void drawFilledEllipsoid3d(DataContainer<data_t>& dc, data_t amplitude, Vec3 center,
                                          Vec3 sizes, data_t phi, data_t theta, data_t psi);

    private:
        /**
         * \brief Draw a sheared filled 2d ellipse using a Bresenham-type algorithm
         *
         * \param[in,out] dc the DataContainer where the ellipse should be drawn in
         * \param[in] amplitude the "color" of the ellipse and its filling
         * \param[in] center the 2d index of where to place the center of the ellipse in dc
         * \param[in] sizes the radii (horizontal/vertical) of the ellipse
         * \param[in] shear the amount of shearing to apply in x/y direction
         *
         * This method uses an adapted algorithm from John Kennedy, "A Fast Bresenham Type Algorithm
         * For Drawing Ellipses".
         */
        static void drawShearedFilledEllipse2d(DataContainer<data_t>& dc, data_t amplitude,
                                               Vec2 center, Vec2 sizes, Vec2 shear);

        /**
         * \brief draw sheared 2d line pairs for ellipses
         *
         * \param[in,out] dc the DataContainer where the lines should be drawn in
         * \param[in] amplitude the "color" of the line
         * \param[in] center the 2d index of where to center the line in dc
         * \param[in] xOffset the x offset from the center
         * \param[in] yOffset the y offset from the center
         * \param[in] shear the amonut of shearing to apply in x/y direction
         *
         * Using ellipse symmetry, this draws four points with the coordinates center[0] +- xOffset
         * / center[1] +- yOffset, as well as the connecting lines between them (proceeding along
         * the x axis).
         */
        static void drawShearedLinePairs2d(DataContainer<data_t>& dc, data_t amplitude, Vec2 center,
                                           index_t xOffset, index_t yOffset, Vec2 shear);
    };

} // namespace elsa
