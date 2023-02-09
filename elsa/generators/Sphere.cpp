#include "Sphere.h"
#include "Ellipsoid.h"

namespace elsa::phantoms
{

    template <typename data_t>
    Sphere<data_t>::Sphere(data_t amplit, Vec3i center, data_t radius)
        : _amplit{amplit}, _center{center}, _radius{radius}
    {

        Logger::get("Sphere")->info("Sphere with amplitude {}, Center ({},{},{}) radius {}",
                                    _amplit, _center[INDEX_X], _center[INDEX_Y], _center[INDEX_Z],
                                    _radius);
    };

    template <Blending b, typename data_t>
    void rasterize(Sphere<data_t>& sphere, VolumeDescriptor& dd, DataContainer<data_t>& dc)
    {
        // Rasterize sphere as ellipsoid with no rotation an equal half axis
        Vec3X<data_t> halfAxis(3);
        halfAxis << sphere.getRadius(), sphere.getRadius(), sphere.getRadius();
        Vec3X<data_t> noRotation(3);
        noRotation << 0, 0, 0;

        Ellipsoid el{sphere.getAmplitude(), sphere.getCenter(), halfAxis, noRotation};
        rasterize<b, data_t>(el, dd, dc);
    };

    // ------------------------------------------
    // explicit template instantiation
    template class Sphere<float>;
    template class Sphere<double>;

    template void rasterize<Blending::ADDITION, float>(Sphere<float>& el, VolumeDescriptor& dd,
                                                       DataContainer<float>& dc);
    template void rasterize<Blending::ADDITION, double>(Sphere<double>& el, VolumeDescriptor& dd,
                                                        DataContainer<double>& dc);

    template void rasterize<Blending::OVERWRITE, float>(Sphere<float>& el, VolumeDescriptor& dd,
                                                        DataContainer<float>& dc);
    template void rasterize<Blending::OVERWRITE, double>(Sphere<double>& el, VolumeDescriptor& dd,
                                                         DataContainer<double>& dc);

} // namespace elsa::phantoms
