#include "EllipCylinder.h"

namespace elsa::phantoms
{

    template <typename data_t>
    EllipCylinder<data_t>::EllipCylinder(Orientation o, data_t amplit, elsa::phantoms::Vec3i center,
                                         Vec2X<data_t> halfAxis, data_t length)
        : _orientation{o}, _amplit{amplit}, _center{center}, _halfAxis{halfAxis}, _length{length}
    {

        aSqr = _halfAxis[INDEX_A] * _halfAxis[INDEX_A];
        bSqr = _halfAxis[INDEX_B] * _halfAxis[INDEX_B];
        aSqrbSqr = aSqr * bSqr;
    };

    template <typename data_t>
    bool EllipCylinder<data_t>::isInEllipCylinder(const Vec3i& idx) const
    {
        // Depending on the orientation only one part is compiled
        if (_orientation == Orientation::X_AXIS) {
            return (double(idx[INDEX_Y] * idx[INDEX_Y]) * bSqr
                    + double(idx[INDEX_Z] * idx[INDEX_Z]) * aSqr)
                   <= aSqrbSqr;
        } else if (_orientation == Orientation::Y_AXIS) {
            return (double(idx[INDEX_X] * idx[INDEX_X]) * bSqr
                    + double(idx[INDEX_Z] * idx[INDEX_Z]) * aSqr)
                   <= aSqrbSqr;
        } else {
            return (double(idx[INDEX_X] * idx[INDEX_X]) * bSqr
                    + double(idx[INDEX_Y] * idx[INDEX_Y]) * aSqr)
                   <= aSqrbSqr;
        }
    };

    /**
     * @brief returns the min and max value of the axis to allow length bigger than the dimension of
     * the datacontainer. If the length is inside the container, there is no clipping. Return value
     * are final voxel indices
     *
     */
    std::pair<index_t, index_t> checkBounds(index_t halfLength, index_t center,
                                            index_t maxDataContainer)
    {
        index_t min, max;
        if (halfLength + center < maxDataContainer) {
            max = halfLength + center;
        } else {
            max = maxDataContainer;
        }

        if (center - halfLength > 0) {
            min = center - halfLength;
        } else {
            min = 0;
        }

        return {min, max};
    }

    template <typename data_t>
    void rasterize(EllipCylinder<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending<data_t> b)
    {
        // check ellipse in origin, mirror 4 points, draw line in the orthogonal directions with
        // length

        Vec3i idx(3);
        idx << 0, 0, 0;

        auto strides = dd.getProductOfCoefficientsPerDimension();

        // global offests for a fast memory reuse in the for loops
        index_t aOffset = 0;
        index_t aOffsetNeg = 0;

        index_t bOffset = 0;
        index_t bOffsetNeg = 0;

        index_t cOffset = 0;

        Vec3i _center = el.getCenter();
        data_t _amplit = el.getAmplitude();

        int TEMP_A = INDEX_X;
        int TEMP_B = INDEX_Y;
        int TEMP_C = INDEX_Z;

        if (el.getOrientation() == Orientation::X_AXIS) {
            TEMP_A = INDEX_Y;
            TEMP_B = INDEX_Z;
            TEMP_C = INDEX_X;

        } else if (el.getOrientation() == Orientation::Y_AXIS) {
            TEMP_A = INDEX_X;
            TEMP_B = INDEX_Z;
            TEMP_C = INDEX_Y;
        }

        index_t halfLength = std::lround(el.getLength() / 2);

        auto [minC, maxC] = checkBounds(halfLength, _center[INDEX_C],
                                        dd.getNumberOfCoefficientsPerDimension()[INDEX_C] - 1);

        // check ellipse on AxB Plane, draw ellipse along the C axis from minC to maxC
        for (; el.isInEllipCylinder(idx); idx[TEMP_A]++) {
            aOffset = (idx[TEMP_A] + _center[TEMP_A]) * strides[TEMP_A];
            aOffsetNeg = (-idx[TEMP_A] + _center[TEMP_A]) * strides[TEMP_A];

            for (; el.isInEllipCylinder(idx); idx[TEMP_B]++) {
                bOffset = (idx[TEMP_B] + _center[TEMP_B]) * strides[TEMP_B];
                bOffsetNeg = (-idx[TEMP_B] + _center[TEMP_B]) * strides[TEMP_B];

                for (index_t line = minC; line <= maxC; line++) {
                    cOffset = line * strides[TEMP_C];

                    dc[aOffset + bOffset + cOffset] = b(dc[aOffset + bOffset + cOffset], _amplit);

                    if (idx[TEMP_A] != 0) {
                        dc[aOffsetNeg + bOffset + cOffset] =
                            b(dc[aOffsetNeg + bOffset + cOffset], _amplit);
                    }

                    if (idx[TEMP_B] != 0) {
                        dc[aOffset + bOffsetNeg + cOffset] =
                            b(dc[aOffset + bOffsetNeg + cOffset], _amplit);
                    }

                    if (idx[TEMP_A] != 0 && idx[TEMP_B] != 0) {
                        dc[aOffsetNeg + bOffsetNeg + cOffset] =
                            b(dc[aOffsetNeg + bOffsetNeg + cOffset], _amplit);
                    }
                };
                idx[TEMP_C] = 0;
            }
            idx[TEMP_B] = 0;
        }
    };

    // ------------------------------------------
    // explicit template instantiation
    template class EllipCylinder<float>;
    template class EllipCylinder<double>;

    template void rasterize<float>(EllipCylinder<float>& el, VolumeDescriptor& dd,
                                   DataContainer<float>& dc, Blending<float> b);
    template void rasterize<double>(EllipCylinder<double>& el, VolumeDescriptor& dd,
                                    DataContainer<double>& dc, Blending<double> b);

} // namespace elsa::phantoms
