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

    template <Blending b, typename data_t>
    void rasterize(Cylinder<data_t>& cl, VolumeDescriptor& dd, DataContainer<data_t>& dc)
    {
        Vec2X<data_t> halfAxis;
        halfAxis << cl.getRadius(), cl.getRadius();
        EllipCylinder elCyl{cl.getOrientation(), cl.getAmplitude(), cl.getCenter(), halfAxis,
                            cl.getLength()};
        rasterize<b, data_t>(elCyl, dd, dc);
    };

    // ------------------------------------------
    // explicit template instantiation
    template class Cylinder<float>;
    template class Cylinder<double>;

    template void rasterize<Blending::ADDITION, float>(Cylinder<float>& el, VolumeDescriptor& dd,
                                                       DataContainer<float>& dc);
    template void rasterize<Blending::ADDITION, double>(Cylinder<double>& el, VolumeDescriptor& dd,
                                                        DataContainer<double>& dc);

    template void rasterize<Blending::OVERWRITE, float>(Cylinder<float>& el, VolumeDescriptor& dd,
                                                        DataContainer<float>& dc);
    template void rasterize<Blending::OVERWRITE, double>(Cylinder<double>& el, VolumeDescriptor& dd,
                                                         DataContainer<double>& dc);

} // namespace elsa::phantoms
