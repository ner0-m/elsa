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
                forwardVoxel<2>(x, Ax);
            else if (this->_rangeDescriptor->getNumberOfDimensions() == 3)
                forwardVoxel<3>(x, Ax);
            else
                throw InvalidArgumentError(
                    "VoxelProjector: can only handle 1 or 2 dimensional Detectors");
        }

        void backward(const BoundingBox aabb, const DataContainer<data_t>& y,
                      DataContainer<data_t>& Aty) const
        {
            (void) aabb;
            if (this->_rangeDescriptor->getNumberOfDimensions() == 2)
                backwardVoxel<2>(y, Aty);
            else if (this->_rangeDescriptor->getNumberOfDimensions() == 3)
                backwardVoxel<3>(y, Aty);
            else
                throw InvalidArgumentError(
                    "VoxelProjector: can only handle 1 or 2 dimensional Detectors");
        }

        template <index_t dim, class Fn>
        void visitDetector(index_t domainIndex, index_t geomIndex, Geometry geometry, real_t radius,
                           Eigen::Matrix<index_t, dim - 1, 1> detectorDims,
                           Eigen::Matrix<index_t, dim, 1> volumeStrides, Fn apply) const
        {
            using StaticIndexVectorVolume_t = Eigen::Matrix<index_t, dim, 1>;
            using StaticIndexVectorDetector_t = Eigen::Matrix<index_t, dim - 1, 1>;
            using StaticRealVectorVolume_t = Eigen::Matrix<real_t, dim, 1>;
            using StaticRealVectorDetector_t = Eigen::Matrix<real_t, dim - 1, 1>;
            using StaticRealVectorHom_t = Eigen::Matrix<real_t, dim + 1, 1>;

            // compute coordinate from index
            StaticIndexVectorVolume_t coordinate = detail::idx2Coord(domainIndex, volumeStrides);

            // Cast to real_t and shift to center of voxel according to origin
            StaticRealVectorVolume_t coordinateShifted =
                coordinate.template cast<real_t>().array() + 0.5;
            StaticRealVectorHom_t homogenousVoxelCoord = coordinateShifted.homogeneous();

            // Project voxel center onto detector
            StaticRealVectorDetector_t center =
                (geometry.getProjectionMatrix() * homogenousVoxelCoord).hnormalized();
            center = center.array() - 0.5;

            auto sourceVoxelDistance =
                (geometry.getExtrinsicMatrix() * homogenousVoxelCoord).norm();

            auto scaling = geometry.getSourceDetectorDistance() / sourceVoxelDistance;

            auto radiusOnDetector = scaling * radius;
            StaticIndexVectorDetector_t detector_max = detectorDims.array() - 1;
            index_t detectorZeroIndex = (detector_max[0] + 1) * geomIndex;

            StaticIndexVectorDetector_t min_corner =
                (center.array() - radiusOnDetector).ceil().template cast<index_t>();
            min_corner = min_corner.cwiseMax(StaticIndexVectorDetector_t::Zero());
            StaticIndexVectorDetector_t max_corner =
                (center.array() + radiusOnDetector).floor().template cast<index_t>();
            max_corner = max_corner.cwiseMin(detector_max);

            StaticRealVectorDetector_t current = min_corner.template cast<real_t>();
            index_t currentIndex{0}, iStride{0}, jStride{0};
            if constexpr (dim == 2) {
                currentIndex = detectorZeroIndex + min_corner[0];
            } else {
                currentIndex = detectorZeroIndex * (detector_max[1] + 1)
                               + min_corner[1] * (detector_max[0] + 1) + min_corner[0];

                iStride = max_corner[0] - min_corner[0] + 1;
                jStride = (detector_max[0] + 1) - iStride;
            }
            for (index_t i = min_corner[0]; i <= max_corner[0]; i++) {
                if constexpr (dim == 2) {
                    // traverse detector pixel in voxel footprint
                    const data_t distance = (center[0] - i) / scaling;
                    apply(detectorZeroIndex + i, distance);
                } else {
                    for (index_t j = min_corner[1]; j <= max_corner[1]; j++) {
                        const RealVector2D_t distanceVec = (center - current) / scaling;
                        apply(currentIndex, distanceVec);
                        currentIndex += 1;
                        current[0] += 1;
                    }
                    currentIndex += jStride;
                    current[0] -= iStride;
                    current[1] += 1;
                }
            }
        }

        template <int dim>
        void forwardVoxel(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const
        {
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(Ax.getDataDescriptor());

            auto& volume = x.getDataDescriptor();
            const Eigen::Matrix<index_t, dim, 1>& volumeStrides =
                volume.getProductOfCoefficientsPerDimension();
            const Eigen::Matrix<index_t, dim - 1, 1>& detectorDims =
                detectorDesc.getNumberOfCoefficientsPerDimension().head(dim - 1);

            // loop over geometries/poses in parallel
#pragma omp parallel for
            for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                 geomIndex++) {
                auto& geometry = detectorDesc.getGeometry()[asUnsigned(geomIndex)];
                // loop over voxels
                for (index_t domainIndex = 0; domainIndex < volume.getNumberOfCoefficients();
                     ++domainIndex) {
                    auto voxelWeight = x[domainIndex];

                    visitDetector<dim>(
                        domainIndex, geomIndex, geometry, this->self().radius(), detectorDims,
                        volumeStrides,
                        [&](const index_t index, Eigen::Matrix<real_t, dim - 1, 1> distance) {
                            auto wght = this->self().weight(distance);
                            Ax[index] += voxelWeight * wght;
                        });
                }
            }
        }

        template <int dim>
        void backwardVoxel(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const
        {
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(y.getDataDescriptor());

            auto& volume = Aty.getDataDescriptor();
            const Eigen::Matrix<index_t, dim, 1>& volumeStrides =
                volume.getProductOfCoefficientsPerDimension();
            const Eigen::Matrix<index_t, dim - 1, 1>& detectorDims =
                detectorDesc.getNumberOfCoefficientsPerDimension().head(dim - 1);

#pragma omp parallel for
            // loop over voxels in parallel
            for (index_t domainIndex = 0; domainIndex < volume.getNumberOfCoefficients();
                 ++domainIndex) {
                // loop over geometries/poses in parallel
                for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                     geomIndex++) {

                    auto& geometry = detectorDesc.getGeometry()[asUnsigned(geomIndex)];

                    visitDetector<dim>(
                        domainIndex, geomIndex, geometry, this->self().radius(), detectorDims,
                        volumeStrides,
                        [&](const index_t index, Eigen::Matrix<real_t, dim - 1, 1> distance) {
                            auto wght = this->self().weight(distance);
                            Aty[domainIndex] += wght * y[index];
                        });
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
            (void) primDistance;
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
