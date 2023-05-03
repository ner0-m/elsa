#pragma once

#include "elsaDefines.h"
#include "Luts.hpp"
#include "LinearOperator.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "DataContainer.h"
#include "Logger.h"
#include "Blobs.h"
#include "BSplines.h"

#include "XrayProjector.h"
#include "VoxelComputation.h"
#include "Math.hpp"

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

                    voxel::visitDetector<dim>(
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

                    voxel::visitDetector<dim>(
                        domainIndex, geomIndex, geometry, this->self().radius(), detectorDims,
                        volumeStrides,
                        [&](const index_t index, Eigen::Matrix<real_t, dim - 1, 1> distance) {
                            auto wght = this->self().weight(distance.norm());
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

        data_t radius() const { return blob_.radius(); }

        data_t weight(data_t distance) const { return blob_.get_lut()(distance); }
        data_t weight(RealMatrix_t distance) const { return blob_.get_lut()(distance.norm()); }
        data_t weight(RealMatrix_t distance, data_t primDistance) const
        {
            (void) primDistance;
            return weight(distance);
        }

        /// implement the polymorphic clone operation
        BlobVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBlob<data_t> blob_;

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

        data_t radius() const { return blob_.radius(); }

        data_t weight(data_t distance) const
        {
            return blob_.get_derivative_lut().operator()(distance) * math::sgn(distance);
        }
        data_t weight(RealMatrix_t distance) const
        {
            return blob_.get_derivative_lut().operator()(distance.norm()) * math::sgn(distance(0));
        }
        data_t weight(data_t distance, data_t primDistance) const
        {
            return blob_.get_normalized_gradient_lut().operator()(distance) * primDistance;
        }

        /// implement the polymorphic clone operation
        PhaseContrastBlobVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBlob<data_t> blob_;

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

        data_t radius() const { return bspline_.radius(); }

        data_t weight(data_t distance) const
        {
            return bspline_.get_lut().operator()(std::abs(distance));
        }
        data_t weight(RealMatrix_t distance) const
        {
            return bspline_.get_lut().operator()(distance.norm());
        }
        data_t weight(RealMatrix_t distance, data_t primDistance) const
        {
            (void) primDistance;
            return weight(distance);
        }

        /// implement the polymorphic clone operation
        BSplineVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBSpline<data_t> bspline_;

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

        data_t radius() const { return bspline_.radius(); }

        data_t weight(data_t distance) const
        {
            return bspline_.get_derivative_lut().operator()(std::abs(distance))
                   * math::sgn(distance);
        }
        data_t weight(RealMatrix_t distance) const { return weight(distance.norm()); }
        data_t weight(data_t distance, data_t primDistance) const
        {
            return bspline_.get_normalized_gradient_lut().operator()(distance) * primDistance;
        }

        /// implement the polymorphic clone operation
        PhaseContrastBSplineVoxelProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBSpline<data_t> bspline_;

        using Base = VoxelProjector<data_t, PhaseContrastBSplineVoxelProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };
}; // namespace elsa
