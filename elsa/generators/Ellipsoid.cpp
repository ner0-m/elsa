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
    index_t Ellipsoid<data_t>::getRoundMaxWidth() const
    {
        data_t max = _halfAxis.colwise().maxCoeff()[0] * 2;
        return long(std::ceil(max));
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

    template <typename data_t>
    void rasterize(Ellipsoid<data_t>& el, VolumeDescriptor& dd, DataContainer<data_t>& dc)
    {
        if (el.isRotated()) {
            rasterizeRotation(el, dd, dc);
        } else {
            rasterizeNoRotation(el, dd, dc);
        }
    };

    template <typename data_t>
    void rasterizeNoRotation(const Ellipsoid<data_t>& el, VolumeDescriptor& dd,
                             DataContainer<data_t>& dc)
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

                    dc[xOffset + yOffset + zOffset] += _amplit;

                    // Voxel in ellipsoids can be mirrored 8 times if they are not on a mirror
                    // plane. Depending on the voxels location they are mirrored or not.
                    // This exclude prevents double increment of the same voxel on the mirror plane.

                    if (idx[INDEX_X] != 0) {
                        dc[xOffsetNeg + yOffset + zOffset] += _amplit;
                    }
                    if (idx[INDEX_Y] != 0) {
                        dc[xOffset + yOffsetNeg + zOffset] += _amplit;
                    }
                    if (idx[INDEX_Z] != 0) {
                        dc[xOffset + yOffset + zOffsetNeg] += _amplit;
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Y] != 0) {
                        dc[xOffsetNeg + yOffsetNeg + zOffset] += _amplit;
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Z] != 0) {
                        dc[xOffsetNeg + yOffset + zOffsetNeg] += _amplit;
                    }

                    if (idx[INDEX_Y] != 0 && idx[INDEX_Z] != 0) {
                        dc[xOffset + yOffsetNeg + zOffsetNeg] += _amplit;
                    }

                    if (idx[INDEX_X] != 0 && idx[INDEX_Y] != 0 && idx[INDEX_Z] != 0) {
                        dc[xOffsetNeg + yOffsetNeg + zOffsetNeg] += _amplit;
                    }
                };
                idx[INDEX_X] = 0;
            }
            idx[INDEX_Y] = 0;
        }
    };

    template <typename data_t>
    std::pair<index_t, index_t> findBoundingBox(Ellipsoid<data_t> const& el)
    {
        auto maxCoeff = el.getRoundMaxWidth();
        auto halfCoeff = maxCoeff / 2;

        index_t min = -halfCoeff;
        index_t max = halfCoeff;

        return std::pair{min, max};
    }

    /**
     * Returns [min,max]
     * To find the smallest value to start and find the greatest value to stop.
     * 1. a part of the ellipsoid is out of the phantom
     * 2. the ellipsoid has a gap between his surface and the border of the phantom
     */
    template <typename data_t>
    std::pair<data_t, data_t> insidePhantom(VolumeDescriptor const& dd, Vec3i const& center,
                                            index_t minBoundingBox, index_t maxBoundingBox)
    {

        /**
         * Relative to dimension of datacontainer
         */
        Vec3i idx_shifted_min(3);
        Vec3i idx_shifted_max(3);

        idx_shifted_min << minBoundingBox + center[INDEX_X], minBoundingBox + center[INDEX_Y],
            minBoundingBox + center[INDEX_Z];
        idx_shifted_max << maxBoundingBox + center[INDEX_X], maxBoundingBox + center[INDEX_Y],
            maxBoundingBox + center[INDEX_Z];

        index_t minX = std::max(idx_shifted_min[INDEX_X], static_cast<index_t>(0));
        index_t minY = std::max(idx_shifted_min[INDEX_Y], static_cast<index_t>(0));
        index_t minZ = std::max(idx_shifted_min[INDEX_Z], static_cast<index_t>(0));

        index_t maxX =
            std::min(idx_shifted_max[INDEX_X], dd.getNumberOfCoefficientsPerDimension()[INDEX_X]);
        index_t maxY =
            std::min(idx_shifted_max[INDEX_Y], dd.getNumberOfCoefficientsPerDimension()[INDEX_Y]);
        index_t maxZ =
            std::min(idx_shifted_max[INDEX_Z], dd.getNumberOfCoefficientsPerDimension()[INDEX_Z]);

        /**
         * translate relative to bounding box
         */
        auto min =
            std::min({minX - center[INDEX_X], minY - center[INDEX_Y], minZ - center[INDEX_Z]});
        auto max =
            std::max({maxX - center[INDEX_X], maxY - center[INDEX_Y], maxZ - center[INDEX_Z]});
        return std::pair<data_t, data_t>{data_t(min), data_t(max)};
    }

    template <typename data_t>
    void rasterizeRotation(const Ellipsoid<data_t>& el, VolumeDescriptor& dd,
                           DataContainer<data_t>& dc)
    {

        const Vec3i _center = el.getCenter();
        data_t _amplit = el.getAmplitude();

        auto [minNoCheck, maxNoCheck] = findBoundingBox<data_t>(el);

        auto strides = dd.getProductOfCoefficientsPerDimension();

        auto [min, max] = insidePhantom<data_t>(dd, _center, minNoCheck, maxNoCheck);

        index_t minL = index_t(min);

        Vec3X<data_t> idx(3);
        Vec3i idx_shifted(3);

        idx << min, min, min;
        idx_shifted << minL + _center[INDEX_X], minL + _center[INDEX_Y], minL + _center[INDEX_Z];

        Vec3X<data_t> rotated(3);

        index_t xOffset = 0;
        index_t yOffset = 0;
        index_t zOffset = 0;

        for (; idx[INDEX_Z] <= max; idx[INDEX_Z]++, idx_shifted[INDEX_Z]++) {
            zOffset = idx_shifted[INDEX_Z] * strides[INDEX_Z];
            for (; idx[INDEX_Y] <= max; idx[INDEX_Y]++, idx_shifted[INDEX_Y]++) {
                yOffset = idx_shifted[INDEX_Y] * strides[INDEX_Y];
                for (; idx[INDEX_X] <= max; idx[INDEX_X]++, idx_shifted[INDEX_X]++) {
                    xOffset = idx_shifted[INDEX_X] * strides[INDEX_X];
                    rotated = el.getInvRotationMatrix() * idx;
                    if (el.isInEllipsoid(rotated)) {
                        dc[xOffset + yOffset + zOffset] += _amplit;
                    }
                }
                // reset X
                idx[INDEX_X] = min;
                idx_shifted[INDEX_X] = minL + _center[INDEX_X];
            }
            // reset Y
            idx[INDEX_Y] = min;
            idx_shifted[INDEX_Y] = minL + _center[INDEX_Y];
        }
    };

    // ------------------------------------------
    // explicit template instantiation
    template class Ellipsoid<float>;
    template class Ellipsoid<double>;

    template void rasterize<float>(Ellipsoid<float>& el, VolumeDescriptor& dd,
                                   DataContainer<float>& dc);
    template void rasterize<double>(Ellipsoid<double>& el, VolumeDescriptor& dd,
                                    DataContainer<double>& dc);

} // namespace elsa::phantoms
