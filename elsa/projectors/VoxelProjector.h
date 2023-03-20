#pragma once

#include "elsaDefines.h"
#include "Luts.hpp"
#include "LinearOperator.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "DataContainer.h"
#include "Logger.h"
#include "Blobs.h"

#include "XrayProjector.h"

namespace elsa
{
    template <typename data_t, typename Derived>
    class VoxelProjector;

    template <typename data_t = real_t>
    class BlobVoxelProjector;

    template <typename data_t = real_t>
    class PhaseContrastBlobVoxelProjector;

    template <typename data_t = real_t>
    class BSplineVoxelProjector;

    template <typename data_t = real_t>
    class PhaseContrastBSplineVoxelProjector;

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

    template <typename data_t>
    struct XrayProjectorInnerTypes<BSplineVoxelProjector<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    template <typename data_t>
    struct XrayProjectorInnerTypes<PhaseContrastBSplineVoxelProjector<data_t>> {
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

        using RealVector2D_t = Eigen::Matrix<real_t, 2, 1>;
        using RealVector3D_t = Eigen::Matrix<real_t, 3, 1>;
        using RealVector4D_t = Eigen::Matrix<real_t, 4, 1>;
        using IndexVector2D_t = Eigen::Matrix<index_t, 2, 1>;
        using IndexVector3D_t = Eigen::Matrix<index_t, 3, 1>;

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
            auto volumeSize = x.getSize();
            const IndexVector2D_t productOfCoefficientsPerDimension =
                volume.getProductOfCoefficientsPerDimension();
            IndexVector2D_t coordinate;

            auto voxelRadius = this->self().radius();
            const auto& geometries{detectorDesc.getGeometry()};

            // loop over geometries/poses in parallel
#pragma omp parallel for private(coordinate)
            for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                 geomIndex++) {

                // helper to find index into sinogram line
                auto detectorZeroIndex = (upperDetectorI + 1) * geomIndex;

                const auto& geometry = geometries[asUnsigned(geomIndex)];
                const Eigen::Matrix<real_t, 2, 3>& projMatrix = geometry.getProjectionMatrix();
                const Eigen::Matrix<real_t, 2, 3>& extMatrix = geometry.getExtrinsicMatrix();

                // loop over voxels
                for (index_t domainIndex = 0; domainIndex < volumeSize; ++domainIndex) {
                    // compute coordinate from index
                    index_t leftOver = domainIndex;
                    coordinate[1] = leftOver / productOfCoefficientsPerDimension[1];
                    leftOver %= productOfCoefficientsPerDimension[1];
                    coordinate[0] = leftOver;

                    auto voxelWeight = x[domainIndex];

                    // Cast to real_t and shift to center of voxel according to origin
                    RealVector2D_t volumeCoord = coordinate.template cast<real_t>().array() + 0.5;

                    RealVector3D_t homogeneousVoxelCoord;
                    homogeneousVoxelCoord << volumeCoord, 1;

                    // Project voxel center onto detector using camera matrix
                    RealVector2D_t voxelCenterOnDetectorHomogenous =
                        (projMatrix * homogeneousVoxelCoord);
                    data_t detectorCoord =
                        voxelCenterOnDetectorHomogenous[0] / voxelCenterOnDetectorHomogenous[1]
                        - 0.5;

                    // compute source voxel distance
                    RealVector2D_t voxelInCameraSpace = (extMatrix * homogeneousVoxelCoord);
                    auto distance = voxelInCameraSpace.norm();

                    // compute scaling using intercept theorem
                    auto scaling = geometry.getSourceDetectorDistance() / distance;

                    auto radiusOnDetector = voxelRadius * scaling;
                    // compute bounding box
                    auto lower = std::max(
                        (index_t) 0, static_cast<index_t>(ceil(detectorCoord - radiusOnDetector)));
                    auto upper =
                        std::min((index_t) upperDetectorI,
                                 static_cast<index_t>(floor(detectorCoord + radiusOnDetector)));

                    // traverse detector pixel in voxel footprint
                    for (index_t neighbour = lower; neighbour <= upper; neighbour++) {
                        const data_t distance = (detectorCoord - neighbour);
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

            const IndexVector3D_t& upperDetectorI =
                detectorDesc.getNumberOfCoefficientsPerDimension();

            const auto upperDetectorX = upperDetectorI[0] - 1;
            const auto upperDetectorY = upperDetectorI[1] - 1;
            const auto detectorXStride = upperDetectorI[0];
            const auto detectorYStride = upperDetectorI[0] * upperDetectorI[1];

            const auto& volume = x.getDataDescriptor();
            const auto volumeSize = x.getSize();
            const IndexVector3D_t productOfCoefficientsPerDimension =
                volume.getProductOfCoefficientsPerDimension();
            IndexVector3D_t coordinate;
            const auto& geometries{detectorDesc.getGeometry()};

            const auto voxelRadius = this->self().radius();
#pragma omp parallel for
            for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                 geomIndex++) {

                const auto& geometry = geometries[asUnsigned(geomIndex)];
                const Eigen::Matrix<real_t, 3, 4>& projMatrix = geometry.getProjectionMatrix();
                const Eigen::Matrix<real_t, 3, 4>& extMatrix = geometry.getExtrinsicMatrix();

                // loop over voxels
                for (index_t domainIndex = 0; domainIndex < volumeSize; ++domainIndex) {
                    // compute coordinate from index
                    index_t leftOver = domainIndex;
                    for (index_t i = 2; i >= 1; --i) {
                        coordinate[i] = leftOver / productOfCoefficientsPerDimension[i];
                        leftOver %= productOfCoefficientsPerDimension[i];
                    }
                    coordinate[0] = leftOver;

                    auto voxelWeight = x[domainIndex];

                    // Cast to real_t and shift to center of voxel according to origin
                    RealVector3D_t volumeCoord = coordinate.template cast<real_t>().array() + 0.5;

                    RealVector4D_t homogeneousVoxelCoord;
                    homogeneousVoxelCoord << volumeCoord, 1;

                    // Project voxel center onto detector using camera matrix
                    RealVector3D_t voxelCenterOnDetectorHomogenous =
                        (projMatrix * homogeneousVoxelCoord);
                    voxelCenterOnDetectorHomogenous.block(0, 0, 2, 1) /=
                        voxelCenterOnDetectorHomogenous[2];

                    // correct origin shift
                    RealVector2D_t detectorCoord =
                        voxelCenterOnDetectorHomogenous.head(2).array() - 0.5;

                    // compute source voxel distance
                    RealVector3D_t voxelInCameraSpace = (extMatrix * homogeneousVoxelCoord);
                    auto distance = voxelInCameraSpace.norm();

                    // compute scaling using intercept theorem
                    auto scaling = geometry.getSourceDetectorDistance() / distance;

                    auto radiusOnDetector = voxelRadius * scaling;

                    // create bounding box
                    index_t lowerX =
                        std::max((index_t) 0,
                                 static_cast<index_t>(ceil(detectorCoord[0] - radiusOnDetector)));
                    index_t upperX =
                        std::min(upperDetectorX,
                                 static_cast<index_t>(floor(detectorCoord[0] + radiusOnDetector)));
                    index_t lowerY =
                        std::max((index_t) 0,
                                 static_cast<index_t>(ceil(detectorCoord[1] - radiusOnDetector)));
                    index_t upperY =
                        std::min(upperDetectorY,
                                 static_cast<index_t>(floor(detectorCoord[1] + radiusOnDetector)));

                    // initialize variables for performance
                    RealVector2D_t currentCoord{
                        {static_cast<real_t>(lowerX), static_cast<real_t>(lowerY)}};
                    index_t currentIndex =
                        lowerX + lowerY * detectorXStride + geomIndex * detectorYStride;
                    index_t iStride = upperX - lowerX + 1;
                    index_t jStride = detectorXStride - iStride;

                    for (index_t j = lowerY; j <= upperY; j++) {
                        for (index_t i = lowerX; i <= upperX; i++) {
                            const RealVector2D_t distanceVec = (detectorCoord - currentCoord);
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
            auto volumeSize = Aty.getSize();
            IndexVector2D_t productOfCoefficientsPerDimension =
                volume.getProductOfCoefficientsPerDimension();
            IndexVector2D_t coordinate;

            auto voxelRadius = this->self().radius();

            const auto& geometries{detectorDesc.getGeometry()};

            // loop over voxels in parallel
#pragma omp parallel for private(coordinate)
            for (index_t domainIndex = 0; domainIndex < volumeSize; ++domainIndex) {

                // compute coordinate from index
                index_t leftOver = domainIndex;
                coordinate[1] = leftOver / productOfCoefficientsPerDimension[1];
                leftOver %= productOfCoefficientsPerDimension[1];
                coordinate[0] = leftOver;

                // loop over geometries/poses
                for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                     geomIndex++) {
                    // helper to find index into sinogram line
                    auto detectorZeroIndex = (upperDetectorI + 1) * geomIndex;

                    const auto& geometry = geometries[asUnsigned(geomIndex)];
                    const Eigen::Matrix<real_t, 2, 3>& projMatrix = geometry.getProjectionMatrix();
                    const Eigen::Matrix<real_t, 2, 3>& extMatrix = geometry.getExtrinsicMatrix();

                    // Cast to real_t and shift to center of voxel according to origin
                    RealVector2D_t volumeCoord = coordinate.template cast<real_t>().array() + 0.5;

                    RealVector3D_t homogeneousVoxelCoord;
                    homogeneousVoxelCoord << volumeCoord, 1;

                    // Project onto detector using camera matrix
                    RealVector2D_t voxelCenterOnDetectorHomogenous =
                        (projMatrix * homogeneousVoxelCoord);
                    // correct origin shift
                    data_t detectorCoord =
                        voxelCenterOnDetectorHomogenous[0] / voxelCenterOnDetectorHomogenous[1]
                        - 0.5;

                    // compute source voxel distance
                    RealVector2D_t voxelInCameraSpace = (extMatrix * homogeneousVoxelCoord);
                    auto distance = voxelInCameraSpace.norm();

                    // compute scaling using intercept theorem
                    auto scaling = geometry.getSourceDetectorDistance() / distance;

                    auto radiusOnDetector = voxelRadius * scaling;

                    // bounding box on detector
                    auto lower = std::max(
                        (index_t) 0, static_cast<index_t>(ceil(detectorCoord - radiusOnDetector)));
                    auto upper =
                        std::min((index_t) upperDetectorI,
                                 static_cast<index_t>(floor(detectorCoord + radiusOnDetector)));

                    // traverse detector pixel in voxel footprint
                    for (index_t neighbour = lower; neighbour <= upper; neighbour++) {
                        const auto distance = (detectorCoord - neighbour);
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

            const IndexVector3D_t& upperDetectorI =
                detectorDesc.getNumberOfCoefficientsPerDimension();

            const auto upperDetectorX = upperDetectorI[0] - 1;
            const auto upperDetectorY = upperDetectorI[1] - 1;
            const auto detectorXStride = upperDetectorI[0];
            const auto detectorYStride = upperDetectorI[0] * upperDetectorI[1];

            auto& volume = Aty.getDataDescriptor();
            auto volumeSize = Aty.getSize();

            auto voxelRadius = this->self().radius();
            const IndexVector3D_t productOfCoefficientsPerDimension =
                volume.getProductOfCoefficientsPerDimension();
            IndexVector3D_t coordinate;
            const auto& geometries{detectorDesc.getGeometry()};

            // loop over voxels in parallel
#pragma omp parallel for
            for (index_t domainIndex = 0; domainIndex < volumeSize; ++domainIndex) {
                // compute coordinate from index
                index_t leftOver = domainIndex;
                for (index_t i = 2; i >= 1; --i) {
                    coordinate[i] = leftOver / productOfCoefficientsPerDimension[i];
                    leftOver %= productOfCoefficientsPerDimension[i];
                }
                coordinate[0] = leftOver;

                // loop over geometries
                auto coord = volume.getCoordinateFromIndex(domainIndex);
                for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                     geomIndex++) {

                    const auto& geometry = geometries[asUnsigned(geomIndex)];
                    const Eigen::Matrix<real_t, 3, 4>& projMatrix = geometry.getProjectionMatrix();
                    const Eigen::Matrix<real_t, 3, 4>& extMatrix = geometry.getExtrinsicMatrix();

                    // Cast to real_t and shift to center of voxel according to origin
                    RealVector3D_t volumeCoord = coordinate.template cast<real_t>().array() + 0.5;

                    RealVector4D_t homogeneousVoxelCoord;
                    homogeneousVoxelCoord << volumeCoord, 1;

                    // Project voxel center onto detector using camera matrix
                    RealVector3D_t voxelCenterOnDetectorHomogenous =
                        (projMatrix * homogeneousVoxelCoord);
                    voxelCenterOnDetectorHomogenous.block(0, 0, 2, 1) /=
                        voxelCenterOnDetectorHomogenous[2];

                    // correct origin shift
                    RealVector2D_t detectorCoord =
                        voxelCenterOnDetectorHomogenous.head(2).array() - 0.5;

                    // compute source voxel distance
                    RealVector3D_t voxelInCameraSpace = (extMatrix * homogeneousVoxelCoord);
                    auto distance = voxelInCameraSpace.norm();

                    // compute scaling using intercept theorem
                    auto scaling = geometry.getSourceDetectorDistance() / distance;

                    // find all detector pixels that are hit
                    auto radiusOnDetector = voxelRadius * scaling;

                    // constrain to detector size
                    index_t lowerX =
                        std::max((index_t) 0,
                                 static_cast<index_t>(ceil(detectorCoord[0] - radiusOnDetector)));
                    index_t upperX =
                        std::min(upperDetectorX,
                                 static_cast<index_t>(floor(detectorCoord[0] + radiusOnDetector)));
                    index_t lowerY =
                        std::max((index_t) 0,
                                 static_cast<index_t>(ceil(detectorCoord[1] - radiusOnDetector)));
                    index_t upperY =
                        std::min(upperDetectorY,
                                 static_cast<index_t>(floor(detectorCoord[1] + radiusOnDetector)));

                    // initialize variables for performance
                    RealVector2D_t currentCoord{
                        {static_cast<real_t>(lowerX), static_cast<real_t>(lowerY)}};
                    index_t currentIndex =
                        lowerX + lowerY * detectorXStride + geomIndex * detectorYStride;
                    index_t iStride = upperX - lowerX + 1;
                    index_t jStride = detectorXStride - iStride;

                    for (index_t j = lowerY; j <= upperY; j++) {
                        for (index_t i = lowerX; i <= upperX; i++) {
                            const RealVector2D_t distanceVec = (detectorCoord - currentCoord);
                            const auto distance = distanceVec.norm();
                            // first axis always is always the differential axis TODO
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

        BlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor, data_t radius, data_t alpha,
                           index_t order);

        BlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return lut_.radius(); }

        data_t weight(data_t distance) const { return lut_(std::abs(distance)); }
        data_t weight(data_t distance, data_t primDistance) const
        {
            (void) primDistance;
            return weight(distance);
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

        PhaseContrastBlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                        const DetectorDescriptor& rangeDescriptor, data_t radius,
                                        data_t alpha, index_t order);

        PhaseContrastBlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                        const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return lut_.radius(); }

        data_t weight(data_t distance) const
        {
            return lut_(std::abs(distance)) * math::sgn(distance);
        }
        data_t weight(data_t distance, data_t primDistance) const
        {
            return lut3D_(distance) * primDistance;
        }

        /// implement the polymorphic clone operation
        PhaseContrastBlobVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBlobDerivativeLut<data_t, 100> lut_;
        ProjectedBlobNormalizedGradientLut<data_t, 100> lut3D_;

        using Base = VoxelProjector<data_t, PhaseContrastBlobVoxelProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class BSplineVoxelProjector : public VoxelProjector<data_t, BSplineVoxelProjector<data_t>>
    {
    public:
        using self_type = BSplineVoxelProjector<data_t>;

        BSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                              const DetectorDescriptor& rangeDescriptor, index_t order);

        BSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                              const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return lut_.radius(); }

        data_t weight(data_t distance) const { return lut_(std::abs(distance)); }
        data_t weight(data_t distance, data_t primDistance) const
        {
            return lut_(std::abs(distance));
        }

        /// implement the polymorphic clone operation
        BSplineVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBSplineLut<data_t, 100> lut_;

        using Base = VoxelProjector<data_t, BSplineVoxelProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class PhaseContrastBSplineVoxelProjector
        : public VoxelProjector<data_t, PhaseContrastBSplineVoxelProjector<data_t>>
    {
    public:
        using self_type = PhaseContrastBSplineVoxelProjector<data_t>;

        PhaseContrastBSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                           const DetectorDescriptor& rangeDescriptor,
                                           index_t order);

        PhaseContrastBSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                                           const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return lut_.radius(); }

        data_t weight(data_t distance) const
        {
            return lut_(std::abs(distance)) * math::sgn(distance);
        }
        data_t weight(data_t distance, data_t primDistance) const
        {
            return lut3D_(distance) * primDistance;
        }

        /// implement the polymorphic clone operation
        PhaseContrastBSplineVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBSplineDerivativeLut<data_t, 100> lut_;
        ProjectedBSplineNormalizedGradientLut<data_t, 100> lut3D_;

        using Base = VoxelProjector<data_t, PhaseContrastBSplineVoxelProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };

}; // namespace elsa
