#include "EllipCylinderFree.h"

namespace elsa::phantoms
{

    template <typename data_t>
    EllipCylinderFree<data_t>::EllipCylinderFree(data_t amplit, elsa::phantoms::Vec3i center,
                                                 Vec2X<data_t> halfAxis, data_t length,
                                                 Vec3X<data_t> eulers)
        : _amplit{amplit}, _center{center}, _halfAxis{halfAxis}, _length{length}, _eulers{eulers}
    {

        aSqr = _halfAxis[INDEX_A] * _halfAxis[INDEX_A];
        bSqr = _halfAxis[INDEX_B] * _halfAxis[INDEX_B];
        aSqrbSqr = aSqr * bSqr;

        _centerX = _center.cast<data_t>();

        fillRotationMatrix(_eulers, rot);
        rot.transposeInPlace();
    };

    template <typename data_t>
    bool EllipCylinderFree<data_t>::isInEllipCylinderFree(const Vec3X<data_t>& idx,
                                                          index_t halfLength) const
    {
        // check length on z axis
        Vec3X<data_t> shifted{idx - _centerX};
        Vec3X<data_t> rotated{getInvRotationMatrix() * shifted};
        if (std::abs(rotated[INDEX_Z]) > data_t(halfLength)) {
            return false;
        } else {
            return (double(rotated[INDEX_X] * rotated[INDEX_X]) * bSqr
                    + double(rotated[INDEX_Y] * rotated[INDEX_Y]) * aSqr)
                   <= aSqrbSqr;
        }
    };

    template <typename data_t>
    index_t getHalfDiagonal(EllipCylinderFree<data_t>& el, const index_t centerHalfLength)
    {
        auto maxAxis = std::max(el.getHalfAxis()[INDEX_A], el.getHalfAxis()[INDEX_A]);
        // Diagonal from center to edge of ellipse on the end of the cylinder with theorem of
        // pythagoras
        return std::lround(std::ceil(
            std::sqrt(double(centerHalfLength) * double(centerHalfLength) + maxAxis * maxAxis)));
    }

    template <typename data_t>
    std::array<data_t, 6>
        boundingBox(const index_t halfLength, const Vec3i& _center,
                    const IndexVector_t& ncpd /*numberOfCoefficientsPerDimension zero based*/)
    {

        index_t minX, minY, minZ, maxX, maxY, maxZ;

        minX = _center[INDEX_X] - halfLength > 0 ? _center[INDEX_X] - halfLength : 0;
        minY = _center[INDEX_Y] - halfLength > 0 ? _center[INDEX_Y] - halfLength : 0;
        minZ = _center[INDEX_Z] - halfLength > 0 ? _center[INDEX_Z] - halfLength : 0;

        maxX = _center[INDEX_X] + halfLength > ncpd[INDEX_X] ? ncpd[INDEX_X]
                                                             : _center[INDEX_X] + halfLength;
        maxY = _center[INDEX_Y] + halfLength > ncpd[INDEX_Y] ? ncpd[INDEX_Y]
                                                             : _center[INDEX_Y] + halfLength;
        maxZ = _center[INDEX_Z] + halfLength > ncpd[INDEX_Z] ? ncpd[INDEX_Z]
                                                             : _center[INDEX_Z] + halfLength;

        return {data_t(minX), data_t(minY), data_t(minZ), data_t(maxX), data_t(maxY), data_t(maxZ)};
    };

    template <typename data_t>
    void rasterize(EllipCylinderFree<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending<data_t> b)
    {
        auto strides = dd.getProductOfCoefficientsPerDimension();
        // global offests for a fast memory reuse in the for loops
        index_t xOffset = 0;
        index_t yOffset = 0;
        index_t zOffset = 0;

        Vec3i _center = el.getCenter();
        data_t _amplit = el.getAmplitude();

        index_t halfLength = std::lround(el.getLength() / 2);

        auto [minX, minY, minZ, maxX, maxY, maxZ] = boundingBox<data_t>(
            getHalfDiagonal(el, halfLength), _center, dd.getNumberOfCoefficientsPerDimension());

        Vec3X<data_t> idx(3);
        idx << minX, minY, minZ;

        for (index_t z = index_t(minZ); z <= index_t(maxZ); z++, idx[INDEX_Z]++) {
            zOffset = z * strides[INDEX_Z];

            for (index_t y = index_t(minY); y <= index_t(maxY); y++, idx[INDEX_Y]++) {
                yOffset = y * strides[INDEX_Y];

                for (index_t x = index_t(minX); x <= index_t(maxX); x++, idx[INDEX_X]++) {
                    xOffset = x * strides[INDEX_X];
                    if (el.isInEllipCylinderFree(idx, halfLength)) {
                        dc[zOffset + yOffset + xOffset] =
                            b(dc[zOffset + yOffset + xOffset], _amplit);
                    }
                }

                idx[INDEX_X] = minX;
            }
            idx[INDEX_Y] = minY;
        }
    };

    // ------------------------------------------
    // explicit template instantiation
    template class EllipCylinderFree<float>;
    template class EllipCylinderFree<double>;

    template void rasterize<float>(EllipCylinderFree<float>& el, VolumeDescriptor& dd,
                                   DataContainer<float>& dc, Blending<float> b);
    template void rasterize<double>(EllipCylinderFree<double>& el, VolumeDescriptor& dd,
                                    DataContainer<double>& dc, Blending<double> b);

} // namespace elsa::phantoms
