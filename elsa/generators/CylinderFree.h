#pragma once
#include "PhantomDefines.h"

namespace elsa::phantoms
{

    template <typename data_t = double>
    class CylinderFree
    {

    private:
        data_t _amplit;
        Vec3i _center;
        data_t _radius;
        data_t _length;
        Vec3X<data_t> _eulers;

    public:
        /**
         * @param amlpit amplitude wich is added to the voxel on rasterization
         * @param center center of the object
         * @param radius radius of the circle in the cross section
         * @param length the length from one side throught the center to the other side
         * @param eulers euler angels
         *
         */
        CylinderFree(data_t amplit, Vec3i center, data_t radius, data_t length,
                     Vec3X<data_t> eulers);

        /**
         * @brief returns the amplitude to color the voxel
         */
        const data_t getAmplitude() const { return _amplit; };
        /**
         * @brief returns the center of the CylinderFree
         */
        const Vec3i& getCenter() const { return _center; };
        /**
         * @brief returns the center of the CylinderFree
         */
        const data_t getRadius() const { return _radius; };
        /**
         * @brief returns the length of the CylinderFree
         */
        const data_t getLength() const { return _length; };
        /**
         * @brief returns the euler angels of the CylinderFree
         */
        const Vec3X<data_t>& getEulerAngels() const { return _eulers; };
    };

    /**
     * @brief Rasterizes the given CylinderFree in the given data container.
     */
    template <typename data_t>
    void rasterize(CylinderFree<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending<data_t> b);

} // namespace elsa::phantoms

template <typename data_t>
struct fmt::formatter<elsa::phantoms::CylinderFree<data_t>> : fmt::formatter<std::string> {
    auto format(elsa::phantoms::CylinderFree<data_t> cylFree, format_context& ctx)
        -> decltype(ctx.out())
    {
        auto _center = cylFree.getCenter();
        auto _amplit = cylFree.getAmplitude();
        auto _radius = cylFree.getRadius();
        auto _length = cylFree.getLength();
        auto _eulers = cylFree.getEulerAngels();

        return format_to(ctx.out(),
                         "CylinderFree with amplitude {}, Center ({},{},{}) ,radius ({}), "
                         "length "
                         "{} , euler angels ({},{},{}) ",
                         _amplit, _center[elsa::phantoms::INDEX_X],
                         _center[elsa::phantoms::INDEX_Y], _center[elsa::phantoms::INDEX_Z],
                         _radius, _length, _eulers[elsa::phantoms::INDEX_X],
                         _eulers[elsa::phantoms::INDEX_Y], _eulers[elsa::phantoms::INDEX_Z]);
    }
};
