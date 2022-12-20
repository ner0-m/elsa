#include "CylinderFree.h"
#include "EllipCylinderFree.h"

namespace elsa::phantoms
{

    template <typename data_t>
    CylinderFree<data_t>::CylinderFree(data_t amplit, elsa::phantoms::Vec3i center, data_t radius,
                                       data_t length, Vec3X<data_t> eulers)
        : _amplit{amplit}, _center{center}, _radius{radius}, _length{length}, _eulers{eulers}
    {
    }

    template <typename data_t>
    void rasterize(CylinderFree<data_t>& cl, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending<data_t> b)
    {
        Vec2X<data_t> halfAxis;
        halfAxis << cl.getRadius(), cl.getRadius();
        EllipCylinderFree elCylFree{cl.getAmplitude(), cl.getCenter(), halfAxis, cl.getLength(),
                                    cl.getEulerAngels()};
        rasterize(elCylFree, dd, dc, b);
    };

    // ------------------------------------------
    // explicit template instantiation
    template class CylinderFree<float>;
    template class CylinderFree<double>;

    template void rasterize<float>(CylinderFree<float>& el, VolumeDescriptor& dd,
                                   DataContainer<float>& dc, Blending<float> b);
    template void rasterize<double>(CylinderFree<double>& el, VolumeDescriptor& dd,
                                    DataContainer<double>& dc, Blending<double> b);

} // namespace elsa::phantoms
