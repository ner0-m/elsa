#pragma once
#include "PhantomDefines.h"

namespace elsa::phantoms
{

    template <typename data_t = double>
    class Ellipsoid
    {

    private:
        data_t _amplit;
        Vec3i _center;
        Vec3X<data_t> _halfAxis;
        Vec3X<data_t> _eulers;

        data_t bSqrcSqr;
        data_t aSqrcSqr;
        data_t aSqrbSqr;
        data_t aSqrbSqrcSqr;

        // setup euler rotation matrix
        Eigen::Matrix<data_t, 3, 3> rot;
        bool rotated = false;

    public:
        Ellipsoid(data_t amplit, Vec3i center, Vec3X<data_t> halfAxis, Vec3X<data_t> eulers);
        /**
         * @brief returns true if there is any not zero euler angle
         */
        bool isRotated() { return rotated; };
        /**
         * @brief returns the center of the ellipsoid
         */
        const Vec3i& getCenter() const { return _center; };
        /**
         * @brief returns the amplitude to color the voxel
         */
        const data_t getAmplitude() const { return _amplit; };
        /**
         * @brief get inverse rotation matrix
         */
        const Eigen::Matrix<data_t, 3, 3> getInvRotationMatrix() const { return rot; };

        bool isInEllipsoid(const Vec3i& idx) const;
        bool isInEllipsoid(const Vec3X<data_t>& idx) const;

        /**
         * @brief returns the ceil max floating point width as a double of the longest half axis
         */
        index_t getRoundMaxWidth() const;
    };

    /**
     * @brief Rasterizes the given ellipsoid in the given data container.
     */
    template <typename data_t>
    void rasterize(Ellipsoid<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc);

} // namespace elsa::phantoms
