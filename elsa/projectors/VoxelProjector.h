#pragma once

#include "elsaDefines.h"
#include "Timer.h"
#include "Luts.hpp"
#include "SliceTraversal.h"
#include "LinearOperator.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "DataContainer.h"
#include "BoundingBox.h"
#include "Logger.h"
#include "Blobs.h"
#include "CartesianIndices.h"

#include "XrayProjector.h"

#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ostr.h"

namespace elsa
{
    namespace math
    {
        template <typename T>
        constexpr inline int sgn(T val)
        {
            return (T(0) < val) - (val < T(0));
        }

        inline index_t imax(index_t a, index_t b)
        {
            // signed for arithmetic shift
            index_t mask = a - b;
            // mask < 0 means MSB is 1.
            return a + ((b - a) & (mask >> 63));
        }

        inline index_t imin(index_t a, index_t b)
        {
            // signed for arithmetic shift
            index_t mask = a - b;
            // mask < 0 means MSB is 1.
            return a + ((b - a) & (~mask >> 63));
        }
    } // namespace math

    template <typename data_t, typename Derived>
    class VoxelProjector;

    template <typename data_t = real_t>
    class BlobVoxelProjector;

    template <typename data_t = real_t>
    class PhaseContrastBlobVoxelProjector;

    template <typename data_t>
    struct XrayProjectorInnerTypes<BlobVoxelProjector<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    template <typename data_t>
    struct XrayProjectorInnerTypes<PhaseContrastBlobVoxelProjector<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    template <typename data_t, typename Derived>
    class VoxelProjector : public XrayProjector<Derived>
    {
    public:
        using self_type = VoxelProjector<data_t, Derived>;
        using base_type = XrayProjector<Derived>;
        using value_type = typename base_type::value_type;
        using forward_tag = typename base_type::forward_tag;
        using backward_tag = typename base_type::backward_tag;

        VoxelProjector(const VolumeDescriptor& domainDescriptor,
                       const DetectorDescriptor& rangeDescriptor)
            : base_type(domainDescriptor, rangeDescriptor)
        {
            // sanity checks
            auto dim = domainDescriptor.getNumberOfDimensions();
            if (dim < 2 || dim > 3) {
                throw InvalidArgumentError("VoxelLutProjector: only supporting 2d/3d operations");
            }

            if (dim != rangeDescriptor.getNumberOfDimensions()) {
                throw InvalidArgumentError(
                    "VoxelLutProjector: domain and range dimension need to match");
            }

            if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
                throw InvalidArgumentError(
                    "VoxelLutProjector: rangeDescriptor without any geometry");
            }
        }

        /// default destructor
        ~VoxelProjector() override = default;

    private:
        /// apply the binary method (i.e. forward projection)
        void forward(const BoundingBox aabb, const DataContainer<data_t>& x,
                     DataContainer<data_t>& Ax) const
        {
            (void) aabb;
            if (this->_rangeDescriptor->getNumberOfDimensions() == 2)
                forward2D(x, Ax);
            else if (this->_rangeDescriptor->getNumberOfDimensions() == 3)
                forward3D(x, Ax);
            else
                throw InvalidArgumentError(
                    "VoxelProjector: can only handle 1 or 2 dimensional Detectors");
        }

        void forward2D(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const
        {
            // Ax is the detector
            // x is the volume
            // given x, compute Ax
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(Ax.getDataDescriptor());

            index_t upperDetectorI = detectorDesc.getNumberOfCoefficientsPerDimension()[0] - 1;

            auto& volume = x.getDataDescriptor();
            const auto dimDetector = detectorDesc.getNumberOfDimensions();
            auto volumeSize = x.getSize();

            auto voxelRadius = this->self().radius();

            const RealVector_t volumeOriginShift = volume.getSpacingPerDimension() * 0.5;
            RealVector_t detectorOriginShift = volume.getSpacingPerDimension() * 0.5;
            detectorOriginShift[dimDetector - 1] = 0;

            IndexVector_t detectorIndexGeomStart(2);
            detectorIndexGeomStart << 0, 0;
#pragma omp parallel for
            for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                 geomIndex++) {

                detectorIndexGeomStart[1] = geomIndex;
                auto detectorZeroIndex =
                    detectorDesc.getIndexFromCoordinate(detectorIndexGeomStart);

                // loop over voxels
                for (index_t domainIndex = 0; domainIndex < volumeSize; ++domainIndex) {
                    const IndexVector_t& coord = volume.getCoordinateFromIndex(domainIndex);
                    auto voxelWeight = x[domainIndex];

                    // Cast to real_t and shift to center of voxel according to origin
                    RealVector_t volumeCoord = coord.template cast<real_t>() + volumeOriginShift;

                    // Project onto detector and compute the magnification
                    auto [detectorCoordShifted, scaling] =
                        detectorDesc.projectAndScaleVoxelOnDetector(volumeCoord, geomIndex);

                    // correct origin shift
                    auto detectorCoordVec = detectorCoordShifted - detectorOriginShift;
                    data_t detectorCoord = detectorCoordVec[0];

                    // find all detector pixels that are hit
                    auto radiusOnDetector = static_cast<index_t>(std::round(voxelRadius * scaling));
                    auto detectorIndex = static_cast<index_t>(std::round(detectorCoord));

                    auto lower = std::max((index_t) 0, detectorIndex - radiusOnDetector);
                    auto upper = std::min(upperDetectorI, detectorIndex + radiusOnDetector);

                    for (index_t neighbour = lower; neighbour <= upper; neighbour++) {
                        const data_t distance = (detectorCoord - neighbour);

                        const auto signum = math::sgn(distance);
                        Ax[detectorZeroIndex + neighbour] +=
                            this->self().weight(distance / scaling) * voxelWeight;
                    }
                }
            }
        }

        void forward3D(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const
        {

            // Ax is the detector
            // x is the volume
            // given x, compute Ax
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(Ax.getDataDescriptor());

            const IndexVector_t lowerDetector = IndexVector_t::Constant(3, 0);
            const IndexVector_t upperDetector =
                detectorDesc.getNumberOfCoefficientsPerDimension().array() - 1;
            const index_t detectorRowLength = upperDetector[0] + 1;

            const auto& volume = x.getDataDescriptor();
            const auto dimDetector = detectorDesc.getNumberOfDimensions();
            const auto volumeSize = x.getSize();

            const auto voxelRadius = this->self().radius();

            const RealVector_t volumeOriginShift = volume.getSpacingPerDimension() * 0.5;
            RealVector_t detectorOriginShift = volume.getSpacingPerDimension() * 0.5;
            detectorOriginShift[dimDetector - 1] = 0;
            //#pragma omp parallel for
            for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                 geomIndex++) {

                // loop over voxels
                for (index_t domainIndex = 0; domainIndex < volumeSize; ++domainIndex) {
                    const IndexVector_t& coord = volume.getCoordinateFromIndex(domainIndex);
                    auto voxelWeight = x[domainIndex];

                    // Cast to real_t and shift to center of voxel according to origin
                    RealVector_t volumeCoord = coord.template cast<real_t>() + volumeOriginShift;

                    // Project onto detector and compute the magnification
                    auto [detectorCoordShifted, scaling] =
                        detectorDesc.projectAndScaleVoxelOnDetector(volumeCoord, geomIndex);

                    // correct origin shift
                    auto detectorCoordWithGeom = detectorCoordShifted - detectorOriginShift;

                    // find all detector pixels that are hit
                    auto radiusOnDetector = IndexVector_t::Constant(
                        3, static_cast<index_t>(std::round(voxelRadius * scaling)));
                    auto detectorIndexVector = detectorCoordWithGeom.template cast<index_t>();

                    IndexVector_t lowerIndex = detectorIndexVector - radiusOnDetector;
                    IndexVector_t upperIndex = detectorIndexVector + radiusOnDetector;

                    lowerIndex = lowerIndex.cwiseMax(lowerDetector);
                    upperIndex = upperIndex.cwiseMin(upperDetector);

                    auto detectorCoord = detectorCoordWithGeom.head(2);
                    RealVector_t currentCoord = lowerIndex.head(2).template cast<real_t>();
                    lowerIndex[2] = geomIndex;
                    index_t currentIndex = detectorDesc.getIndexFromCoordinate(lowerIndex);
                    index_t iStride = upperIndex[0] - lowerIndex[0] + 1;
                    index_t jStride = detectorRowLength - iStride;

                    for (index_t j = lowerIndex[1]; j <= upperIndex[1]; j++) {
                        for (index_t i = lowerIndex[0]; i <= upperIndex[0]; i++) {
                            const auto distanceVec = (detectorCoord - currentCoord);
                            const auto distance = distanceVec.norm();
                            // let first axis always be the differential axis TODO
                            Ax[currentIndex] +=
                                this->self().weight(distance / scaling, distanceVec[0] / scaling)
                                * voxelWeight;
                            currentIndex += 1;
                            currentCoord[0] += 1;
                        }
                        currentIndex += jStride;
                        currentCoord[0] -= iStride;
                        currentCoord[1] += 1;
                    }
                }
            }
        }

        void backward(const BoundingBox aabb, const DataContainer<data_t>& y,
                      DataContainer<data_t>& Aty) const
        {
            (void) aabb;
            if (this->_rangeDescriptor->getNumberOfDimensions() == 2)
                backward2D(y, Aty);
            else if (this->_rangeDescriptor->getNumberOfDimensions() == 3)
                backward3D(y, Aty);
            else
                throw InvalidArgumentError(
                    "VoxelProjector: can only handle 1 or 2 dimensional Detectors");
        }

        void backward2D(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const
        {
            // y is the detector
            // Aty is the volume
            // given y, compute Aty
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(y.getDataDescriptor());

            index_t upperDetectorI = detectorDesc.getNumberOfCoefficientsPerDimension()[0] - 1;

            auto& volume = Aty.getDataDescriptor();
            const auto dimDetector = detectorDesc.getNumberOfDimensions();
            auto volumeSize = Aty.getSize();

            auto voxelRadius = this->self().radius();

            const RealVector_t volumeOriginShift = volume.getSpacingPerDimension() * 0.5;
            RealVector_t detectorOriginShift = volume.getSpacingPerDimension() * 0.5;
            detectorOriginShift[dimDetector - 1] = 0;

            IndexVector_t detectorIndexGeomStart(2);
            detectorIndexGeomStart << 0, 0;
#pragma omp parallel for
            // loop over voxels
            for (index_t domainIndex = 0; domainIndex < volumeSize; ++domainIndex) {
                auto coord = volume.getCoordinateFromIndex(domainIndex);
                // loop over geometries
                for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                     geomIndex++) {

                    detectorIndexGeomStart[1] = geomIndex;
                    auto detectorZeroIndex =
                        detectorDesc.getIndexFromCoordinate(detectorIndexGeomStart);

                    // Cast to real_t and shift to center of voxel according to origin
                    RealVector_t volumeCoord = coord.template cast<real_t>() + volumeOriginShift;

                    // Project onto detector and compute the magnification
                    auto [detectorCoordShifted, scaling] =
                        detectorDesc.projectAndScaleVoxelOnDetector(volumeCoord, geomIndex);

                    // correct origin shift
                    auto detectorCoordVec = detectorCoordShifted - detectorOriginShift;
                    data_t detectorCoord = detectorCoordVec[0];

                    // find all detector pixels that are hit
                    auto radiusOnDetector = static_cast<index_t>(std::round(voxelRadius * scaling));
                    auto detectorIndex = static_cast<index_t>(std::round(detectorCoord));

                    auto lower = std::max((index_t) 0, detectorIndex - radiusOnDetector);
                    auto upper = std::min(upperDetectorI, detectorIndex + radiusOnDetector);

                    for (index_t neighbour = lower; neighbour <= upper; neighbour++) {
                        const auto distance = (detectorCoord - neighbour);
#pragma omp atomic
                        Aty[domainIndex] += this->self().weight(distance / scaling)
                                            * y[detectorZeroIndex + neighbour];
                    }
                }
            }
        }

        void backward3D(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const
        {
            // y is the detector
            // Aty is the volume
            // given y, compute Aty
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(y.getDataDescriptor());

            const IndexVector_t lowerDetector = IndexVector_t::Constant(3, 0);
            const IndexVector_t upperDetector =
                detectorDesc.getNumberOfCoefficientsPerDimension().array() - 1;
            const index_t detectorRowLength = upperDetector[0] + 1;

            auto& volume = Aty.getDataDescriptor();
            const auto dimDetector = detectorDesc.getNumberOfDimensions();
            auto volumeSize = Aty.getSize();

            auto voxelRadius = this->self().radius();

            const RealVector_t volumeOriginShift = volume.getSpacingPerDimension() * 0.5;
            RealVector_t detectorOriginShift = volume.getSpacingPerDimension() * 0.5;
            detectorOriginShift[dimDetector - 1] = 0; // dont shift in pose dimension

#pragma omp parallel for
                                                      // loop over voxels
            for (index_t domainIndex = 0; domainIndex < volumeSize; ++domainIndex) {
                // loop over geometries
                auto coord = volume.getCoordinateFromIndex(domainIndex);
                for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                     geomIndex++) {

                    // Cast to real_t and shift to center of pixel
                    RealVector_t volumeCoord = coord.template cast<real_t>().array() + 0.5;

                    // Project onto detector and compute the magnification
                    auto [detectorCoordShifted, scaling] =
                        detectorDesc.projectAndScaleVoxelOnDetector(volumeCoord, geomIndex);

                    // correct origin shift
                    auto detectorCoordWithGeom = detectorCoordShifted - detectorOriginShift;

                    // find all detector pixels that are hit
                    auto radiusOnDetector = IndexVector_t::Constant(
                        3, static_cast<index_t>(std::round(voxelRadius * scaling)));
                    auto detectorIndexVector = detectorCoordWithGeom.template cast<index_t>();

                    IndexVector_t lowerIndex = detectorIndexVector - radiusOnDetector;
                    IndexVector_t upperIndex = detectorIndexVector + radiusOnDetector;

                    lowerIndex = lowerIndex.cwiseMax(lowerDetector);
                    upperIndex = upperIndex.cwiseMin(upperDetector);

                    auto detectorCoord = detectorCoordWithGeom.head(2);
                    RealVector_t currentCoord = lowerIndex.head(2).template cast<real_t>();
                    lowerIndex[2] = geomIndex;
                    index_t currentIndex = detectorDesc.getIndexFromCoordinate(lowerIndex);
                    index_t iStride = upperIndex[0] - lowerIndex[0] + 1;
                    index_t jStride = detectorRowLength - iStride;

                    for (index_t j = lowerIndex[1]; j <= upperIndex[1]; j++) {
                        for (index_t i = lowerIndex[0]; i <= upperIndex[0]; i++) {
                            const auto distanceVec = (detectorCoord - currentCoord);
                            const auto distance = distanceVec.norm();
                            // let first axis always be the differential axis TODO
#pragma omp atomic
                            Aty[domainIndex] +=
                                this->self().weight(distance / scaling, distanceVec[0] / scaling)
                                * y[currentIndex];
                            currentIndex += 1;
                            currentCoord[0] += 1;
                        }
                        currentIndex += jStride;
                        currentCoord[0] -= iStride;
                        currentCoord[1] += 1;
                    }
                }
            }
        }

        /// implement the polymorphic clone operation
        VoxelProjector<data_t, Derived>* _cloneImpl() const
        {
            return new VoxelProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                      downcast<DetectorDescriptor>(*this->_rangeDescriptor));
        }

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const
        {
            if (!LinearOperator<data_t>::isEqual(other))
                return false;

            auto otherOp = downcast_safe<VoxelProjector>(&other);
            return static_cast<bool>(otherOp);
        }

        friend class XrayProjector<Derived>;
    };

    template <typename data_t>
    class BlobVoxelProjector : public VoxelProjector<data_t, BlobVoxelProjector<data_t>>
    {
    public:
        using self_type = BlobVoxelProjector<data_t>;

        BlobVoxelProjector(data_t radius, data_t alpha, data_t order,
                           const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor);

        BlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return lut_.radius(); }

        data_t weight(data_t distance) const { return lut_(std::abs(distance)); }
        data_t weight(data_t distance, data_t primDistance) const
        {
            (void) primDistance;
            return lut_(distance);
        }

        /// implement the polymorphic clone operation
        BlobVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBlobLut<data_t, 100> lut_;

        using Base = VoxelProjector<data_t, BlobVoxelProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class PhaseContrastBlobVoxelProjector
        : public VoxelProjector<data_t, PhaseContrastBlobVoxelProjector<data_t>>
    {
    public:
        using self_type = PhaseContrastBlobVoxelProjector<data_t>;

        PhaseContrastBlobVoxelProjector(data_t radius, data_t alpha, data_t order,
                                        const VolumeDescriptor& domainDescriptor,
                                        const DetectorDescriptor& rangeDescriptor);

        PhaseContrastBlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                        const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return lut_.radius(); }

        data_t weight(data_t distance) const
        {
            return lut_(std::abs(distance)) * math::sgn(distance);
        }
        data_t weight(data_t distance, data_t primDistance) const
        {
            return lut_(distance) / (distance + 1e-10) * primDistance;
        } // 1e-10 is added to prevent division by zero TODO

        /// implement the polymorphic clone operation
        PhaseContrastBlobVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBlobDerivativeLut<data_t, 100> lut_;

        using Base = VoxelProjector<data_t, PhaseContrastBlobVoxelProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };

}; // namespace elsa