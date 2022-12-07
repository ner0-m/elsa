#pragma once
#include "PhantomDefines.h"

namespace elsa::phantoms
{

    // Fix 3d vector
    template <typename data_t = double,
              typename = std::enable_if_t<std::is_floating_point<data_t>::value>>
    using Vec2X = Eigen::Matrix<data_t, 2, 1>;

    template <typename data_t = double>
    class EllipCylinder
    {

    private:
        data_t aSqr;
        data_t bSqr;
        data_t aSqrbSqr;

        Orientation _orientation;
        data_t _amplit;
        Vec3i _center;
        /* halfAxis for the ellipse always in the order of {dx,dy,dz} - e.g. for
         * ORIENTATION X_AXIS  => {dy,dz}, for ORIENTATION Y_AXIS  => {dx,dz}*/
        Vec2X<data_t> _halfAxis;
        data_t _length;

    public:
        /**
         * @param amlpit amplitude wich is added to the voxel on rasterization
         * @param center center of the object
         * @param halfAxis halfAxis for the ellipse always in the order of {dx,dy,dz} - e.g. for
         * ORIENTATION X_AXIS {dy,dz}
         * @param length the length from one side throught the center to the other side
         *
         */
        EllipCylinder(Orientation o, data_t amplit, Vec3i center, Vec2X<data_t> halfAxis,
                      data_t length);

        /**
         * @brief returns the orientation
         */
        Orientation getOrientation() const { return _orientation; };
        /**
         * @brief returns the amplitude to color the voxel
         */
        const data_t getAmplitude() const { return _amplit; };
        /**
         * @brief returns the center of the EllipCylinder
         */
        const Vec3i& getCenter() const { return _center; };
        /**
         * @brief returns the center of the EllipCylinder
         */
        const Vec2X<data_t>& getHalfAxis() const { return _halfAxis; };
        /**
         * @brief returns the length of the EllipCylinder
         */
        const data_t getLength() const { return _length; };

        bool isInEllipCylinder(const Vec3i& idx) const;
    };

    /**
     * @brief Rasterizes the given EllipCylinder in the given data container.
     */
    template <typename data_t>
    void rasterize(EllipCylinder<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc);

} // namespace elsa::phantoms

/**
 * @brief EllipCylinder formatter to use the Logger.h functions
 */
template <typename data_t>
struct fmt::formatter<elsa::phantoms::EllipCylinder<data_t>> : fmt::formatter<std::string> {
    auto format(elsa::phantoms::EllipCylinder<data_t> ell, format_context& ctx)
        -> decltype(ctx.out())
    {
        auto _center = ell.getCenter();
        auto _amplit = ell.getAmplitude();
        auto _halfAxis = ell.getHalfAxis();
        auto _length = ell.getLength();
        auto _orientation = ell.getOrientation();

        return format_to(ctx.out(),
                         "EllipCylinder with amplitude {}, Center ({},{},{}) , half axis ({},{}), "
                         "length "
                         "{} along orientation {} ",
                         _amplit, _center[elsa::phantoms::INDEX_X],
                         _center[elsa::phantoms::INDEX_Y], _center[elsa::phantoms::INDEX_Z],
                         _halfAxis[elsa::phantoms::INDEX_X], _halfAxis[elsa::phantoms::INDEX_Y],
                         _length, _orientation);
    }
};
