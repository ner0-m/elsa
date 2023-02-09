#pragma once
#include "PhantomDefines.h"

namespace elsa::phantoms
{

    template <typename data_t = double>
    class Box
    {

    private:
        data_t _amplit;
        Vec3i _center;
        Vec3X<data_t> _edgeLength;

    public:
        /**
         * @param amlpit amplitude wich is added to the voxel on rasterization
         * @param center center of the object
         * @param edgeLength length of the box edges X,Y,Z
         *
         */
        Box(data_t amplit, Vec3i center, Vec3X<data_t> edgeLength);

        /**
         * @brief returns the amplitude to color the voxel
         */
        const data_t getAmplitude() const { return _amplit; };
        /**
         * @brief returns the center of the box
         */
        const Vec3i& getCenter() const { return _center; };
        /**
         * @brief returns the length of the edges x,y,z
         */
        const Vec3X<data_t>& getEdgeLength() const { return _edgeLength; };
    };

    /**
     * @brief Rasterizes the given box in the given data container.
     */
    template <Blending b, typename data_t>
    void rasterize(Box<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc);

} // namespace elsa::phantoms

/**
 * @brief Box Formatter to use the Logger.h functions
 */
template <typename data_t>
struct fmt::formatter<elsa::phantoms::Box<data_t>> : fmt::formatter<std::string> {
    auto format(elsa::phantoms::Box<data_t> box, format_context& ctx) -> decltype(ctx.out())
    {
        auto _center = box.getCenter();
        data_t _amplit = box.getAmplitude();
        auto _edgeLength = box.getEdgeLength();
        return format_to(
            ctx.out(), "Box with amplitude {}, Center ({},{},{}) , edge length ({},{},{})", _amplit,
            _center[elsa::phantoms::INDEX_X], _center[elsa::phantoms::INDEX_Y],
            _center[elsa::phantoms::INDEX_Z], _edgeLength[elsa::phantoms::INDEX_X],
            _edgeLength[elsa::phantoms::INDEX_Y], _edgeLength[elsa::phantoms::INDEX_Z]);
    }
};
