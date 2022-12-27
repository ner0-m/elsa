#include "Box.h"

namespace elsa::phantoms
{

    template <typename data_t>
    Box<data_t>::Box(data_t amplit, elsa::phantoms::Vec3i center, Vec3X<data_t> edgeLength)
        : _amplit{amplit}, _center{center}, _edgeLength{edgeLength} {};

    template <Blending b, typename data_t>
    void rasterize(Box<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc)
    {

        auto strides = dd.getProductOfCoefficientsPerDimension();

        // global offests for a fast memory reuse in the for loops
        index_t xOffset = 0;
        index_t xOffsetNeg = 0;
        index_t yOffset = 0;
        index_t yOffsetNeg = 0;
        index_t zOffset = 0;
        index_t zOffsetNeg = 0;

        Vec3i _center = el.getCenter();
        data_t _amplit = el.getAmplitude();

        index_t maxX = std::lround(el.getEdgeLength()[INDEX_X] / 2);
        index_t maxY = std::lround(el.getEdgeLength()[INDEX_Y] / 2);
        index_t maxZ = std::lround(el.getEdgeLength()[INDEX_Z] / 2);

        Vec3i idx(3);
        idx << 0, 0, 0;

        for (; idx[INDEX_Z] <= maxZ; idx[INDEX_Z]++) {
            zOffset = (idx[INDEX_Z] + _center[INDEX_Z]) * strides[INDEX_Z];
            zOffsetNeg = (-idx[INDEX_Z] + _center[INDEX_Z]) * strides[INDEX_Z];

            for (; idx[INDEX_Y] <= maxY; idx[INDEX_Y]++) {
                yOffset = (idx[INDEX_Y] + _center[INDEX_Y]) * strides[INDEX_Y];
                yOffsetNeg = (-idx[INDEX_Y] + _center[INDEX_Y]) * strides[INDEX_Y];

                for (; idx[INDEX_X] <= maxX; idx[INDEX_X]++) {
                    xOffset = (idx[INDEX_X] + _center[INDEX_X]) * strides[INDEX_X];
                    xOffsetNeg = (-idx[INDEX_X] + _center[INDEX_X]) * strides[INDEX_X];

                    blend<b>(dc, xOffset + yOffset + zOffset, _amplit);

                    // mirror the voxel at most 8 times

                    if (idx[INDEX_X] != 0) {
                        blend<b>(dc, xOffsetNeg + yOffset + zOffset, _amplit);
                    }
                    if (idx[INDEX_Y] != 0) {
                        blend<b>(dc, xOffset + yOffsetNeg + zOffset, _amplit);
                    }
                    if (idx[INDEX_Z] != 0) {
                        blend<b>(dc, xOffset + yOffset + zOffsetNeg, _amplit);
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Y] != 0) {
                        blend<b>(dc, xOffsetNeg + yOffsetNeg + zOffset, _amplit);
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Z] != 0) {
                        blend<b>(dc, xOffsetNeg + yOffset + zOffsetNeg, _amplit);
                    }

                    if (idx[INDEX_Y] != 0 && idx[INDEX_Z] != 0) {
                        blend<b>(dc, xOffset + yOffsetNeg + zOffsetNeg, _amplit);
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Y] != 0 && idx[INDEX_Z] != 0) {
                        blend<b>(dc, xOffsetNeg + yOffsetNeg + zOffsetNeg, _amplit);
                    }
                };
                idx[INDEX_X] = 0;
            }
            idx[INDEX_Y] = 0;
        }
    };

    // ------------------------------------------
    // explicit template instantiation
    template class Box<float>;
    template class Box<double>;

    template void rasterize<Blending::ADDITION, float>(Box<float>& el, VolumeDescriptor& dd,
                                                       DataContainer<float>& dc);
    template void rasterize<Blending::ADDITION, double>(Box<double>& el, VolumeDescriptor& dd,
                                                        DataContainer<double>& dc);

    template void rasterize<Blending::OVERWRITE, float>(Box<float>& el, VolumeDescriptor& dd,
                                                        DataContainer<float>& dc);
    template void rasterize<Blending::OVERWRITE, double>(Box<double>& el, VolumeDescriptor& dd,
                                                         DataContainer<double>& dc);

} // namespace elsa::phantoms
