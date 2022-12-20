#pragma once
#include "PhantomDefines.h"
#include <functional>

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
        Eigen::Matrix<data_t, 3, 3> rot = Eigen::Matrix<data_t, 3, 3>::Identity();
        bool rotated = false;

    public:
        Ellipsoid(data_t amplit, Vec3i center, Vec3X<data_t> halfAxis, Vec3X<data_t> eulers);
        /**
         * @brief returns true if there is any not zero euler angle
         */
        bool isRotated() { return rotated; };
        /**
         * @brief returns the amplitude to color the voxel
         */
        const data_t getAmplitude() const { return _amplit; };
        /**
         * @brief returns the center of the ellipsoid
         */
        const Vec3i& getCenter() const { return _center; };
        /**
         * @brief returns the half axis of the ellipsoid
         */
        const Vec3X<data_t>& getHalfAxis() const { return _halfAxis; };
        /**
         * @brief returns the euler angels of the ellipsoid
         */
        const Vec3X<data_t>& getEulerAngels() const { return _eulers; };
        /**
         * @brief get inverse rotation matrix
         */
        const Eigen::Matrix<data_t, 3, 3> getInvRotationMatrix() const { return rot; };
        /**
         * @brief returns the ceil max floating point width as a double of the longest half axis
         */
        data_t getRoundMaxWidth() const;

        bool isInEllipsoid(const Vec3i& idx) const;
        bool isInEllipsoid(const Vec3X<data_t>& idx) const;
    };

    template <typename data_t>
    using MinMaxFunction = std::function<std::array<data_t, 6>(std::array<data_t, 6>)>;

    /**
     * @brief Rasterizes the given ellipsoid in the given data container.
     */
    template <typename data_t>
    void rasterize(Ellipsoid<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc);

    template <typename data_t>
    void rasterizeWithClipping(Ellipsoid<data_t>& el, VolumeDescriptor& dd,
                               DataContainer<data_t>& dc, MinMaxFunction<data_t> clipping);

} // namespace elsa::phantoms

/**
 * @brief Ellipsoid formatter to use the Logger.h functions
 */
template <typename data_t>
struct fmt::formatter<elsa::phantoms::Ellipsoid<data_t>> : fmt::formatter<std::string> {
    auto format(elsa::phantoms::Ellipsoid<data_t> ell, format_context& ctx) -> decltype(ctx.out())
    {
        auto _center = ell.getCenter();
        auto _amplit = ell.getAmplitude();
        auto _halfAxis = ell.getHalfAxis();
        auto _eulers = ell.getEulerAngels();

        return format_to(ctx.out(),
                         "Ellipsoid with amplitude {}, center ({},{},{}) , half axis ({},{},{}) "
                         "euler angels ({},{},{})",
                         _amplit, _center[elsa::phantoms::INDEX_X],
                         _center[elsa::phantoms::INDEX_Y], _center[elsa::phantoms::INDEX_Z],
                         _halfAxis[elsa::phantoms::INDEX_X], _halfAxis[elsa::phantoms::INDEX_Y],
                         _halfAxis[elsa::phantoms::INDEX_Z], _eulers[elsa::phantoms::INDEX_X],
                         _eulers[elsa::phantoms::INDEX_Y], _eulers[elsa::phantoms::INDEX_Z]);
    }
};
