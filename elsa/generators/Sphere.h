#pragma once
#include "PhantomDefines.h"

namespace elsa::phantoms
{
    template <typename data_t = double>
    class Sphere
    {

    private:
        data_t _amplit;
        Vec3i _center;
        data_t _radius;

    public:
        Sphere(data_t amplit, Vec3i center, data_t radius);
        /**
         * @brief returns the center of the sphere
         */
        const Vec3i& getCenter() const { return _center; };
        /**
         * @brief returns the center of the sphere
         */
        const data_t getRadius() const { return _radius; };
        /**
         * @brief returns the amplitude to color the voxel
         */
        const data_t getAmplitude() const { return _amplit; };
    };

    /**
     * @brief Rasterizes the given sphere in the given data container.
     */
    template <typename data_t>
    void rasterize(Sphere<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc);

} // namespace elsa::phantoms
