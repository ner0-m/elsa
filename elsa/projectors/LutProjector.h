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

#include <stack>

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

    /// View interface for the LutProjector
    template <typename Callable>
    class LutProjectorView
    {
    public:
        struct LutProjectorSentinel;

        class LutProjectorIterator
        {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = DomainElement<real_t>; // TODO: this shouldn't be real_t fixed
            using difference_type = std::ptrdiff_t;
            using pointer = value_type*;
            using reference = value_type&;

            LutProjectorIterator(SliceTraversal::Iter iter, SliceTraversal::Iter end,
                                 const RealRay_t& ray, const IndexVector_t strides, Callable fn,
                                 const IndexVector_t& lower, const IndexVector_t& upper,
                                 const IndexVector_t& distvec)
                : fn_(fn),
                  ray_(ray),
                  iter_(iter),
                  end_(end),
                  strides_(strides),
                  lower_(lower),
                  upper_(upper),
                  distvec_(distvec)
            {
                advance();
            }

            /// Dereference operation for iterator
            value_type operator*()
            {
                auto val = stack_.top();
                stack_.pop();
                return val;
            }

            /// Advance iterator
            LutProjectorIterator& operator++()
            {
                advance();
                return *this;
            }

            /// Advance iterator
            LutProjectorIterator operator++(int)
            {
                auto copy = *this;
                advance();
                return copy;
            }

            friend bool operator==(const LutProjectorIterator& lhs, const LutProjectorIterator& rhs)
            {
                return lhs.iter_ == rhs.iter && lhs.stack_.size() == rhs.stack_.size();
            }

            friend bool operator!=(const LutProjectorIterator& lhs, const LutProjectorIterator& rhs)
            {
                return !(lhs == rhs);
            }

            /// Comparison to sentinel/end
            friend bool operator==(const LutProjectorIterator& iter, LutProjectorSentinel sentinel)
            {
                return iter.stack_.empty() && iter.iter_ == iter.end_;
            }

        private:
            /// Visit all neighbours of the current voxel in slice (i.e. all voxels that are
            /// perpendicular to the leading direction of the current ray in a given radius)
            void visit_neighbours()
            {
                auto curVoxel = *iter_;
                const auto neighbours = neighbours_in_slice(curVoxel, distvec_, lower_, upper_);
                for (const auto& neighbour : neighbours) {
                    // Correct position, such that the distance is still correct
                    const auto correctedPos = neighbour.template cast<real_t>().array() + 0.5;
                    const auto distance = ray_.distance(correctedPos);
                    const auto weight = fn_(distance);

                    auto idx = ravelIndex(neighbour, strides_);

                    stack_.emplace(weight, idx);
                }
            }

            /// Advance iterator implementation. Only check neighbours if the stack is empty
            void advance()
            {
                if (stack_.empty() && iter_ != end_) {
                    visit_neighbours();
                    ++iter_;
                }
            }

            Callable fn_;
            RealRay_t ray_;
            SliceTraversal::Iter iter_;
            SliceTraversal::Iter end_;
            IndexVector_t strides_;

            IndexVector_t lower_;
            IndexVector_t upper_;
            IndexVector_t distvec_;

            std::stack<value_type> stack_;
        };

        /// End sentinel
        struct LutProjectorSentinel {
            friend bool operator!=(const LutProjectorIterator& lhs, LutProjectorSentinel rhs)
            {
                return !(lhs == rhs);
            }

            friend bool operator==(const LutProjectorSentinel& lhs, const LutProjectorIterator& rhs)
            {
                return rhs == lhs;
            }

            friend bool operator!=(const LutProjectorSentinel& lhs, const LutProjectorIterator& rhs)
            {
                return !(lhs == rhs);
            }
        };

        /// Construct the View
        LutProjectorView(const BoundingBox& aabb, const RealRay_t& ray, Callable fn,
                         const IndexVector_t& lower, const IndexVector_t& upper,
                         const IndexVector_t& distvec)
            : fn_(fn),
              ray_(ray),
              traversal_(aabb, ray),
              strides_(aabb.strides()),
              lower_(lower),
              upper_(upper),
              distvec_(distvec)
        {
        }

        /// Return the begin iterator
        LutProjectorIterator begin()
        {
            return LutProjectorIterator{traversal_.begin(),
                                        traversal_.end(),
                                        ray_,
                                        strides_,
                                        fn_,
                                        lower_,
                                        upper_,
                                        distvec_};
        }

        /// Return the end sentinel
        LutProjectorSentinel end() { return LutProjectorSentinel{}; }

    private:
        Callable fn_;
        RealRay_t ray_;
        SliceTraversal traversal_;
        IndexVector_t strides_;

        IndexVector_t lower_;
        IndexVector_t upper_;
        IndexVector_t distvec_;
    };

    // additional deduction guide
    template <class Callable>
    LutProjectorView(const BoundingBox& aabb, const RealRay_t& ray, Callable fn)
        -> LutProjectorView<Callable>;

    template <class Callable>
    bool operator==(const typename LutProjectorView<Callable>::LutProjectorIterator& iter,
                    const typename LutProjectorView<Callable>::LutProjectorSentinel& sentinel)
    {
        return iter.iter_ == sentinel.end_;
    }

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
        auto traverseRay(BoundingBox boundingbox, const RealRay_t& ray) const
        {
            const auto support = this->self().support();

            index_t leadingdir = 0;
            ray.direction().array().cwiseAbs().maxCoeff(&leadingdir);

            const IndexVector_t lower = boundingbox.min().template cast<index_t>();
            const IndexVector_t upper = boundingbox.max().template cast<index_t>();

            IndexVector_t distvec = IndexVector_t::Constant(lower.size(), support);
            distvec[leadingdir] = 0;

            // Expand bounding box as rays have larger support now
            auto aabb = boundingbox;
            aabb.min().array() -= static_cast<real_t>(support);
            aabb.min()[leadingdir] += static_cast<real_t>(support);

            aabb.max().array() += static_cast<real_t>(support);
            aabb.max()[leadingdir] -= static_cast<real_t>(support);

            // With the lambda is still a little weird, but that's how I got it running for now :D
            return LutProjectorView(
                aabb, ray, [&](auto dist) { return this->self().weight(dist); }, lower, upper,
                distvec);
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

        BlobProjector(data_t radius, data_t alpha, data_t order,
                      const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor);

        BlobProjector(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor);

        data_t weight(data_t distance) const { return lut_(distance); }

        index_t support() const { return static_cast<index_t>(std::ceil(lut_.radius())); }

        /// implement the polymorphic clone operation
        BlobProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBlobLut<data_t, 100> lut_;

        using Base = LutProjector<data_t, BlobProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class BSplineProjector : public LutProjector<data_t, BSplineProjector<data_t>>
    {
    public:
        using self_type = BlobProjector<data_t>;

        BSplineProjector(data_t degree, const VolumeDescriptor& domainDescriptor,
                         const DetectorDescriptor& rangeDescriptor);

        BSplineProjector(const VolumeDescriptor& domainDescriptor,
                         const DetectorDescriptor& rangeDescriptor);

        data_t weight(data_t distance) const;

        index_t support() const;

        /// implement the polymorphic clone operation
        BSplineProjector<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

    private:
        ProjectedBSplineLut<data_t, 100> lut_;

        using Base = LutProjector<data_t, BSplineProjector<data_t>>;

        friend class XrayProjector<self_type>;
    };

} // namespace elsa
