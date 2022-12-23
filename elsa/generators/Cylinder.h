#pragma once
#include "PhantomDefines.h"

namespace elsa::phantoms
{

    template <typename data_t = double>
    class Cylinder
    {

    private:
        Orientation _orientation;
        data_t _amplit;
        Vec3i _center;
        data_t _radius;
        data_t _length;

    public:
        /**
         * @param o Orientation along X_AXIS, Y_AXIS or Z_AXIS
         * @param amlpit amplitude wich is added to the voxel on rasterization
         * @param center center of the object
         * @param radius radius of the circle in the cross section
         * @param length the length from one side throught the center to the other side
         *
         */
        Cylinder(Orientation o, data_t amplit, Vec3i center, data_t radius, data_t length);

        /**
         * @brief returns the orientation to color the voxel
         */
        Orientation getOrientation() const { return _orientation; };
        /**
         * @brief returns the amplitude to color the voxel
         */
        const data_t getAmplitude() const { return _amplit; };
        /**
         * @brief returns the center of the Cylinder
         */
        const Vec3i& getCenter() const { return _center; };
        /**
         * @brief returns the center of the Cylinder
         */
        const data_t getRadius() const { return _radius; };
        /**
         * @brief returns the length of the Cylinder
         */
        const data_t getLength() const { return _length; };
    };

    /**
     * @brief Rasterizes the given Cylinder in the given data container. Wrapper around
     * EllipCylinder
     */
    template <typename data_t, class Blending>
    void rasterize(Cylinder<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending b);

} // namespace elsa::phantoms

template <typename data_t>
struct fmt::formatter<elsa::phantoms::Cylinder<data_t>> : fmt::formatter<std::string> {
    auto format(elsa::phantoms::Cylinder<data_t> ell, format_context& ctx) -> decltype(ctx.out())
    {
        auto _center = ell.getCenter();
        auto _amplit = ell.getAmplitude();
        auto _radius = ell.getRadius();
        auto _length = ell.getLength();
        auto _orientation = ell.getOrientation();

        return format_to(ctx.out(),
                         "Cylinder with amplitude {}, Center ({},{},{}) ,radius ({}), "
                         "length "
                         "{} along orientation {} ",
                         _amplit, _center[elsa::phantoms::INDEX_X],
                         _center[elsa::phantoms::INDEX_Y], _center[elsa::phantoms::INDEX_Z],
                         _radius, _length, getString(_orientation));
    }
};
