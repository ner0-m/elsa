#pragma once
#include "PhantomDefines.h"

namespace elsa::phantoms
{

    // Fix 3d vector
    template <typename data_t = double,
              typename = std::enable_if_t<std::is_floating_point<data_t>::value>>
    using Vec2X = Eigen::Matrix<data_t, 2, 1>;

    template <typename data_t = double>
    class EllipCylinderFree
    {

    private:
        data_t aSqr;
        data_t bSqr;
        data_t aSqrbSqr;

        data_t _amplit;
        Vec3i _center;

        /* halfAxis for the ellipse always dx,dy */
        Vec2X<data_t> _halfAxis;

        data_t _length;

        Vec3X<data_t> _eulers;

        // rotation matrix
        Eigen::Matrix<data_t, 3, 3> rot;

        /* Same as _center but with another typ for rotation calculation*/
        Vec3X<data_t> _centerX;

    public:
        /**
         * @param amlpit amplitude wich is added to the voxel on rasterization
         * @param center center of the object
         * @param halfAxis halfAxis for the ellipse always in the order of {dx,dy,dz} - e.g. for
         * ORIENTATION X_AXIS {dy,dz}
         * @param length the length from one side throught the center to the other side
         *
         */
        EllipCylinderFree(data_t amplit, Vec3i center, Vec2X<data_t> halfAxis, data_t length,
                          Vec3X<data_t> eulers);
        /**
         * @brief returns the amplitude to color the voxel
         */
        const data_t getAmplitude() const { return _amplit; };
        /**
         * @brief returns the center of the EllipCylinderFree
         */
        const Vec3i& getCenter() const { return _center; };
        /**
         * @brief returns the center of the EllipCylinderFree
         */
        const Vec2X<data_t>& getHalfAxis() const { return _halfAxis; };
        /**
         * @brief returns the length of the EllipCylinderFree
         */
        const data_t getLength() const { return _length; };
        /**
         * @brief returns the euler angels of the EllipCylinderFree
         */
        const Vec3X<data_t>& getEulerAngels() const { return _eulers; };
        /**
         * @brief get inverse rotation matrix
         */
        const Eigen::Matrix<data_t, 3, 3> getInvRotationMatrix() const { return rot; };

        bool isInEllipCylinderFree(const Vec3X<data_t>& idx, index_t halfLength) const;
    };

    /**
     * @brief Rasterizes the given EllipCylinderFree in the given data container.
     */
    template <typename data_t>
    void rasterize(EllipCylinderFree<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending<data_t> b);

} // namespace elsa::phantoms

/**
 * @brief EllipCylinderFree formatter to use the Logger.h functions
 */
template <typename data_t>
struct fmt::formatter<elsa::phantoms::EllipCylinderFree<data_t>> : fmt::formatter<std::string> {
    auto format(elsa::phantoms::EllipCylinderFree<data_t> ell, format_context& ctx)
        -> decltype(ctx.out())
    {
        auto _center = ell.getCenter();
        auto _amplit = ell.getAmplitude();
        auto _halfAxis = ell.getHalfAxis();
        auto _length = ell.getLength();
        auto _eulers = ell.getEulerAngels();

        return format_to(
            ctx.out(),
            "EllipCylinderFree with amplitude {}, Center ({},{},{}) , half axis ({},{}), "
            "length "
            "{} with euler angels ({},{},{}) ",
            _amplit, _center[elsa::phantoms::INDEX_X], _center[elsa::phantoms::INDEX_Y],
            _center[elsa::phantoms::INDEX_Z], _halfAxis[elsa::phantoms::INDEX_X],
            _halfAxis[elsa::phantoms::INDEX_Y], _length, _eulers[elsa::phantoms::INDEX_A],
            _eulers[elsa::phantoms::INDEX_B], _eulers[elsa::phantoms::INDEX_C]);
    }
};
