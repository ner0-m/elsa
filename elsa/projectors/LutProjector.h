#pragma once

#include "elsaDefines.h"
#include "Timer.h"
#include "SliceTraversal.h"
#include "LinearOperator.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "DataContainer.h"
#include "BoundingBox.h"
#include "Logger.h"
#include "Blobs.h"
#include "BSplines.h"
#include "CartesianIndices.h"

#include "XrayProjector.h"

#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ostr.h"

namespace elsa
{
    template <typename data_t, typename Derived>
    class LutProjector;

    template <typename data_t = real_t>
    class BlobProjector;

    template <typename data_t = real_t>
    class BSplineProjector;

    template <typename data_t>
    struct XrayProjectorInnerTypes<BlobProjector<data_t>> {
        using value_type = data_t;
        using forward_tag = ray_driven_tag;
        using backward_tag = ray_driven_tag;
    };

    template <typename data_t>
    struct XrayProjectorInnerTypes<BSplineProjector<data_t>> {
        using value_type = data_t;
        using forward_tag = ray_driven_tag;
        using backward_tag = ray_driven_tag;
    };

    template <typename data_t, typename Derived>
    class LutProjector : public XrayProjector<Derived>
    {
    public:
        using self_type = LutProjector<data_t, Derived>;
        using base_type = XrayProjector<Derived>;
        using value_type = typename base_type::value_type;
        using forward_tag = typename base_type::forward_tag;
        using backward_tag = typename base_type::backward_tag;

        LutProjector(const VolumeDescriptor& domainDescriptor,
                     const DetectorDescriptor& rangeDescriptor)
            : base_type(domainDescriptor, rangeDescriptor)
        {
            // sanity checks
            auto dim = domainDescriptor.getNumberOfDimensions();
            if (dim < 2 || dim > 3) {
                throw InvalidArgumentError("LutProjector: only supporting 2d/3d operations");
            }

            if (dim != rangeDescriptor.getNumberOfDimensions()) {
                throw InvalidArgumentError(
                    "LutProjector: domain and range dimension need to match");
            }

            if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
                throw InvalidArgumentError("LutProjector: rangeDescriptor without any geometry");
            }
        }

        /// default destructor
        ~LutProjector() override = default;

    private:
        /// apply the binary method (i.e. forward projection)
        data_t traverseRayForward(const BoundingBox& boundingbox, const RealRay_t& ray,
                                  const DataContainer<data_t>& x) const
        {
            const IndexVector_t lower = boundingbox.min().template cast<index_t>();
            const IndexVector_t upper = boundingbox.max().template cast<index_t>();
            const auto support = this->self().support();

            index_t leadingdir = 0;
            ray.direction().array().cwiseAbs().maxCoeff(&leadingdir);

            IndexVector_t distvec = IndexVector_t::Constant(lower.size(), support);
            distvec[leadingdir] = 0;

            auto rangeVal = data_t(0);

            // Expand bounding box as rays have larger support now
            auto aabb = boundingbox;
            aabb.min().array() -= static_cast<real_t>(support);
            aabb.min()[leadingdir] += static_cast<real_t>(support);

            aabb.max().array() += static_cast<real_t>(support);
            aabb.max()[leadingdir] -= static_cast<real_t>(support);

            // Keep this here, as it saves us a couple of allocations on clang
            CartesianIndices neighbours(upper);

            // --> setup traversal algorithm
            SliceTraversal traversal(boundingbox, ray);

            for (const auto& curVoxel : traversal) {
                neighbours = neighbours_in_slice(curVoxel, distvec, lower, upper);
                for (auto neighbour : neighbours) {
                    // Correct position, such that the distance is still correct
                    const auto correctedPos = neighbour.template cast<real_t>().array() + 0.5;
                    const auto distance = ray.distance(correctedPos);
                    const auto weight = this->self().weight(distance);

                    rangeVal += weight * x(neighbour);
                }
            }

            return rangeVal;
        }

        void traverseRayBackward(const BoundingBox& boundingbox, const RealRay_t& ray,
                                 const value_type& detectorValue, DataContainer<data_t>& Aty) const
        {
            const IndexVector_t lower = boundingbox.min().template cast<index_t>();
            const IndexVector_t upper = boundingbox.max().template cast<index_t>();
            const auto support = this->self().support();

            index_t leadingdir = 0;
            ray.direction().array().cwiseAbs().maxCoeff(&leadingdir);

            IndexVector_t distvec = IndexVector_t::Constant(lower.size(), support);
            distvec[leadingdir] = 0;

            // Expand bounding box as rays have larger support now
            auto aabb = boundingbox;
            aabb.min().array() -= static_cast<real_t>(support);
            aabb.min()[leadingdir] += static_cast<real_t>(support);

            aabb.max().array() += static_cast<real_t>(support);
            aabb.max()[leadingdir] -= static_cast<real_t>(support);

            // Keep this here, as it saves us a couple of allocations on clang
            CartesianIndices neighbours(upper);

            // --> setup traversal algorithm
            SliceTraversal traversal(aabb, ray);

            for (const auto& curVoxel : traversal) {
                neighbours = neighbours_in_slice(curVoxel, distvec, lower, upper);
                for (auto neighbour : neighbours) {
                    // Correct position, such that the distance is still correct
                    const auto correctedPos = neighbour.template cast<real_t>().array() + 0.5;
                    const auto distance = ray.distance(correctedPos);
                    const auto weight = this->self().weight(distance);

#pragma omp atomic
                    Aty(neighbour) += weight * detectorValue;
                }
            }
        }

        /// implement the polymorphic clone operation
        LutProjector<data_t, Derived>* _cloneImpl() const
        {
            return new LutProjector(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                                    downcast<DetectorDescriptor>(*this->_rangeDescriptor));
        }

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const
        {
            if (!LinearOperator<data_t>::isEqual(other))
                return false;

            auto otherOp = downcast_safe<LutProjector>(&other);
            return static_cast<bool>(otherOp);
        }

        friend class XrayProjector<Derived>;
    };

    template <typename data_t>
    class BlobProjector : public LutProjector<data_t, BlobProjector<data_t>>
    {
    public:
        using self_type = BlobProjector<data_t>;

        BlobProjector(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor,
                      data_t radius = blobs::DEFAULT_RADIUS, data_t alpha = blobs::DEFAULT_ALPHA,
                      index_t order = blobs::DEFAULT_ORDER);

        data_t weight(data_t distance) const { return blob_.get_lut()(distance); }

        index_t support() const { return static_cast<index_t>(std::ceil(blob_.radius())); }

        /// implement the polymorphic clone operation
        BlobProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBlob<data_t> blob_;

        using Base = LutProjector<data_t, BlobProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class BSplineProjector : public LutProjector<data_t, BSplineProjector<data_t>>
    {
    public:
        using self_type = BlobProjector<data_t>;

        BSplineProjector(const VolumeDescriptor& domainDescriptor,
                         const DetectorDescriptor& rangeDescriptor,
                         index_t order = bspline::DEFAULT_ORDER);

        data_t weight(data_t distance) const;

        index_t support() const;

        /// implement the polymorphic clone operation
        BSplineProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBSpline<data_t> bspline_;

        using Base = LutProjector<data_t, BSplineProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };
} // namespace elsa
