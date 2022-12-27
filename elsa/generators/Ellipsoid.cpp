#include "Ellipsoid.h"

namespace elsa::phantoms
{

    template <typename data_t>
    Ellipsoid<data_t>::Ellipsoid(data_t amplit, elsa::phantoms::Vec3i center,
                                 Vec3X<data_t> halfAxis, Vec3X<data_t> eulers)
        : _amplit{amplit}, _center{center}, _halfAxis{halfAxis}, _eulers{eulers}
    {
        bSqrcSqr =
            _halfAxis[INDEX_B] * _halfAxis[INDEX_B] * _halfAxis[INDEX_C] * _halfAxis[INDEX_C];
        aSqrcSqr =
            _halfAxis[INDEX_A] * _halfAxis[INDEX_A] * _halfAxis[INDEX_C] * _halfAxis[INDEX_C];
        aSqrbSqr =
            _halfAxis[INDEX_A] * _halfAxis[INDEX_A] * _halfAxis[INDEX_B] * _halfAxis[INDEX_B];
        aSqrbSqrcSqr = aSqrbSqr * _halfAxis[INDEX_C] * _halfAxis[INDEX_C];

        rotated = (std::abs(eulers[INDEX_PHI]) + std::abs(eulers[INDEX_THETA])
                   + std::abs(eulers[INDEX_PSI]))
                  > 0;

        if (rotated) {
            fillRotationMatrix(eulers, rot);
            rot.transposeInPlace();
        }
    };

    template <typename data_t>
    bool Ellipsoid<data_t>::isInEllipsoid(const Vec3i& idx) const
    {
        return (double(idx[INDEX_X] * idx[INDEX_X]) * bSqrcSqr
                + double(idx[INDEX_Y] * idx[INDEX_Y]) * aSqrcSqr
                + double(idx[INDEX_Z] * idx[INDEX_Z]) * aSqrbSqr)
               <= aSqrbSqrcSqr;
    };

    template <typename data_t>
    bool Ellipsoid<data_t>::isInEllipsoid(const Vec3X<data_t>& idx) const
    {

        return (idx[INDEX_X] * idx[INDEX_X] * bSqrcSqr + idx[INDEX_Y] * idx[INDEX_Y] * aSqrcSqr
                + idx[INDEX_Z] * idx[INDEX_Z] * aSqrbSqr)
               <= aSqrbSqrcSqr;
    };

    template <typename data_t, class Blending>
    void rasterizeNoRotation(Ellipsoid<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                             Blending b)
    {

        Vec3i idx(3);
        idx << 0, 0, 0;

        auto strides = dd.getProductOfCoefficientsPerDimension();

        // global offests for a fast memory reuse in the for loops
        index_t zOffset = 0;
        index_t yOffset = 0;

        index_t zOffsetNeg = 0;
        index_t yOffsetNeg = 0;

        index_t xOffset = 0;
        index_t xOffsetNeg = 0;

        Vec3i _center = el.getCenter();
        data_t _amplit = el.getAmplitude();

        // Slicing on z axis to lower cache misses
        for (; el.isInEllipsoid(idx); idx[INDEX_Z]++) {
            zOffset = (idx[INDEX_Z] + _center[INDEX_Z]) * strides[INDEX_Z];
            zOffsetNeg = (-idx[INDEX_Z] + _center[INDEX_Z]) * strides[INDEX_Z];

            for (; el.isInEllipsoid(idx); idx[INDEX_Y]++) {
                yOffset = (idx[INDEX_Y] + _center[INDEX_Y]) * strides[INDEX_Y];
                yOffsetNeg = (-idx[INDEX_Y] + _center[INDEX_Y]) * strides[INDEX_Y];

                for (; el.isInEllipsoid(idx); idx[INDEX_X]++) {
                    xOffset = (idx[INDEX_X] + _center[INDEX_X]) * strides[INDEX_X];
                    xOffsetNeg = (-idx[INDEX_X] + _center[INDEX_X]) * strides[INDEX_X];

                    b(dc[xOffset + yOffset + zOffset], _amplit);

                    // Voxel in ellipsoids can be mirrored 8 times if they are not on a mirror
                    // plane. Depending on the voxels location they are mirrored or not.
                    // This exclude prevents double increment of the same voxel on the mirror plane.

                    if (idx[INDEX_X] != 0) {
                        b(dc[xOffsetNeg + yOffset + zOffset], _amplit);
                    }
                    if (idx[INDEX_Y] != 0) {
                        b(dc[xOffset + yOffsetNeg + zOffset], _amplit);
                    }
                    if (idx[INDEX_Z] != 0) {
                        b(dc[xOffset + yOffset + zOffsetNeg], _amplit);
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Y] != 0) {
                        b(dc[xOffsetNeg + yOffsetNeg + zOffset], _amplit);
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Z] != 0) {
                        b(dc[xOffsetNeg + yOffset + zOffsetNeg], _amplit);
                    }

                    if (idx[INDEX_Y] != 0 && idx[INDEX_Z] != 0) {
                        b(dc[xOffset + yOffsetNeg + zOffsetNeg], _amplit);
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Y] != 0 && idx[INDEX_Z] != 0) {
                        b(dc[xOffsetNeg + yOffsetNeg + zOffsetNeg], _amplit);
                    }
                }
                idx[INDEX_X] = 0;
            }
            idx[INDEX_Y] = 0;
        }
    };

    template <typename data_t>
    data_t Ellipsoid<data_t>::getRoundMaxHalfWidth() const
    {
        data_t max = _halfAxis.colwise().maxCoeff()[0];
        return std::ceil(max);
    };

    /**
     * Bounding Box in object space
     */
    template <typename data_t>
    std::array<data_t, 6>
        boundingBox(const data_t maxHalfAxis, const Vec3i& _center,
                    const IndexVector_t& ncpd /*numberOfCoefficientsPerDimension zero based*/)
    {

        data_t minX, minY, minZ, maxX, maxY, maxZ;

        minX = std::max(-maxHalfAxis, data_t(-_center[INDEX_X]));
        minY = std::max(-maxHalfAxis, data_t(-_center[INDEX_Y]));
        minZ = std::max(-maxHalfAxis, data_t(-_center[INDEX_Z]));

        maxX = std::min(data_t(ncpd[INDEX_X] - 1 - _center[INDEX_X]), maxHalfAxis);
        maxY = std::min(data_t(ncpd[INDEX_Y] - 1 - _center[INDEX_Y]), maxHalfAxis);
        maxZ = std::min(data_t(ncpd[INDEX_Z] - 1 - _center[INDEX_Z]), maxHalfAxis);

        return {minX, minY, minZ, maxX, maxY, maxZ};
    };

    template <typename data_t, class Blending>
    void rasterizeWithClipping(Ellipsoid<data_t>& el, VolumeDescriptor& dd,
                               DataContainer<data_t>& dc, MinMaxFunction<data_t> clipping,
                               Blending b)
    {
        const Vec3i _center = el.getCenter();
        data_t _amplit = el.getAmplitude();

        auto strides = dd.getProductOfCoefficientsPerDimension();

        auto [minX, minY, minZ, maxX, maxY, maxZ] = clipping(boundingBox<data_t>(
            el.getRoundMaxHalfWidth(), _center, dd.getNumberOfCoefficientsPerDimension()));

        index_t minXSchifted = index_t(minX) + _center[INDEX_X];
        index_t minYSchifted = index_t(minY) + _center[INDEX_Y];
        index_t minZSchifted = index_t(minZ) + _center[INDEX_Z];

        Vec3X<data_t> idx(3);
        Vec3i idx_shifted(3);

        idx << minX, minY, minZ;
        idx_shifted << minXSchifted, minYSchifted, minZSchifted;

        Vec3X<data_t> rotated(3);

        index_t xOffset = 0;
        index_t yOffset = 0;
        index_t zOffset = 0;

        for (; idx[INDEX_Z] <= maxZ; idx[INDEX_Z]++, idx_shifted[INDEX_Z]++) {
            zOffset = idx_shifted[INDEX_Z] * strides[INDEX_Z];
            for (; idx[INDEX_Y] <= maxY; idx[INDEX_Y]++, idx_shifted[INDEX_Y]++) {
                yOffset = idx_shifted[INDEX_Y] * strides[INDEX_Y];
                for (; idx[INDEX_X] <= maxX; idx[INDEX_X]++, idx_shifted[INDEX_X]++) {
                    xOffset = idx_shifted[INDEX_X] * strides[INDEX_X];
                    rotated = el.getInvRotationMatrix() * idx;
                    if (el.isInEllipsoid(rotated)) {
                        b(dc[xOffset + yOffset + zOffset], _amplit);
                    }
                }
                // reset X
                idx[INDEX_X] = minX;
                idx_shifted[INDEX_X] = minXSchifted;
            }
            // reset Y
            idx[INDEX_Y] = minY;
            idx_shifted[INDEX_Y] = minYSchifted;
        }
    };

    template <typename data_t>
    auto noClipping = [](std::array<data_t, 6> minMax) { return minMax; };

    template <typename data_t, class Blending>
    void rasterize(Ellipsoid<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc,
                   Blending b)
    {
        if (el.isRotated()) {
            rasterizeWithClipping<data_t, Blending>(el, dd, dc, noClipping<data_t>, b);
        } else {
            rasterizeNoRotation<data_t, Blending>(el, dd, dc, b);
        }
    };

    // ------------------------------------------
    // explicit template instantiation
    template class Ellipsoid<float>;
    template class Ellipsoid<double>;

    template void rasterize<float, decltype(additiveBlending<float>)>(
        Ellipsoid<float>& el, VolumeDescriptor& dd, DataContainer<float>& dc,
        decltype(additiveBlending<float>) b);
    template void rasterize<double, decltype(additiveBlending<double>)>(
        Ellipsoid<double>& el, VolumeDescriptor& dd, DataContainer<double>& dc,
        decltype(additiveBlending<double>) b);
} // namespace elsa::phantoms
