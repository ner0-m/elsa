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

    template <Blending b, typename data_t>
    void rasterize(CylinderFree<data_t>& cl, VolumeDescriptor& dd, DataContainer<data_t>& dc)
    {
        Vec2X<data_t> halfAxis;
        halfAxis << cl.getRadius(), cl.getRadius();
        EllipCylinderFree elCylFree{cl.getAmplitude(), cl.getCenter(), halfAxis, cl.getLength(),
                                    cl.getEulerAngels()};
        rasterize<b, data_t>(elCylFree, dd, dc);
    };

    // ------------------------------------------
    // explicit template instantiation
    template class CylinderFree<float>;
    template class CylinderFree<double>;

    template void rasterize<Blending::ADDITION, float>(CylinderFree<float>& el,
                                                       VolumeDescriptor& dd,
                                                       DataContainer<float>& dc);
    template void rasterize<Blending::ADDITION, double>(CylinderFree<double>& el,
                                                        VolumeDescriptor& dd,
                                                        DataContainer<double>& dc);

    template void rasterize<Blending::OVERWRITE, float>(CylinderFree<float>& el,
                                                        VolumeDescriptor& dd,
                                                        DataContainer<float>& dc);
    template void rasterize<Blending::OVERWRITE, double>(CylinderFree<double>& el,
                                                         VolumeDescriptor& dd,
                                                         DataContainer<double>& dc);

} // namespace elsa::phantoms
