#include "Cylinder.h"
#include "EllipCylinder.h"

namespace elsa::phantoms
{

    template <typename data_t>
    Cylinder<data_t>::Cylinder(Orientation o, data_t amplit, elsa::phantoms::Vec3i center,
                               data_t radius, data_t length)
        : _orientation{o}, _amplit{amplit}, _center{center}, _radius{radius}, _length{length}
    {
    }

    template <typename data_t>
    void rasterize(Cylinder<data_t>& cl, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending<data_t> b)
    {
        Vec2X<data_t> halfAxis;
        halfAxis << cl.getRadius(), cl.getRadius();
        EllipCylinder elCyl{cl.getOrientation(), cl.getAmplitude(), cl.getCenter(), halfAxis,
                            cl.getLength()};
        rasterize(elCyl, dd, dc, b);
    };

    // ------------------------------------------
    // explicit template instantiation
    template class Cylinder<float>;
    template class Cylinder<double>;

    template void rasterize<float>(Cylinder<float>& el, VolumeDescriptor& dd,
                                   DataContainer<float>& dc, Blending<float> b);
    template void rasterize<double>(Cylinder<double>& el, VolumeDescriptor& dd,
                                    DataContainer<double>& dc, Blending<double> b);

} // namespace elsa::phantoms
