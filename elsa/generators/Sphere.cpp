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

    template <typename data_t>
    void rasterize(Sphere<data_t>& sphere, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending<data_t> b)
    {
        // Rasterize sphere as ellipsoid with no rotation an equal half axis
        Vec3X<data_t> halfAxis(3);
        halfAxis << sphere.getRadius(), sphere.getRadius(), sphere.getRadius();
        Vec3X<data_t> noRotation(3);
        noRotation << 0, 0, 0;

        Ellipsoid el{sphere.getAmplitude(), sphere.getCenter(), halfAxis, noRotation};
        rasterize(el, dd, dc, b);
    };

    // ------------------------------------------
    // explicit template instantiation
    template class Sphere<float>;
    template class Sphere<double>;

    template void rasterize<float>(Sphere<float>& el, VolumeDescriptor& dd,
                                   DataContainer<float>& dc, Blending<float> b);
    template void rasterize<double>(Sphere<double>& el, VolumeDescriptor& dd,
                                    DataContainer<double>& dc, Blending<double> b);

} // namespace elsa::phantoms
