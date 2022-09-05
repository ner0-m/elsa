#include "SiddonsMethod.h"
#include "Timer.h"
#include "DDA.h"
#include "TypeCasts.hpp"

#include <stdexcept>
#include <type_traits>

namespace elsa
{
    template <typename data_t>
    SiddonsMethod<data_t>::SiddonsMethod(const VolumeDescriptor& domainDescriptor,
                                         const DetectorDescriptor& rangeDescriptor)
        : base_type(domainDescriptor, rangeDescriptor)
    {
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw LogicError("SiddonsMethod: domain and range dimension need to match");
        }

        if (dim != 2 && dim != 3) {
            throw LogicError("SiddonsMethod: only supporting 2d/3d operations");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw LogicError("SiddonsMethod: geometry list was empty");
        }
    }

    template <typename data_t>
    SiddonsMethod<data_t>* SiddonsMethod<data_t>::_cloneImpl() const
    {
        return new self_type(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                             downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t>
    bool SiddonsMethod<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherSM = downcast_safe<SiddonsMethod>(&other);
        return static_cast<bool>(otherSM);
    }

    template <typename data_t>
    SiddonsView<data_t> SiddonsMethod<data_t>::traverseRay(const BoundingBox& aabb,
                                                           const RealRay_t& ray) const
    {
        return SiddonsView<data_t>(aabb, ray);
    }

    // ------------------------------------------
    // Implementation of SiddonsView
    template <typename data_t>
    SiddonsView<data_t>::SiddonsView(const BoundingBox& aabb, const RealRay_t& ray)
        : dda_(aabb, ray), strides_(aabb.strides())
    {
    }

    template <typename data_t>
    SiddonsView<data_t>::SiddonsIterator::SiddonsIterator(DDAView::DDAIterator iter,
                                                          const IndexVector_t strides)
        : iter_(iter), strides_(strides)
    {
    }

    template <typename data_t>
    typename SiddonsView<data_t>::SiddonsIterator::value_type
        SiddonsView<data_t>::SiddonsIterator::operator*() const
    {
        auto [weight, voxel] = *iter_;
        return {weight, ravelIndex(voxel, strides_)};
    }

    template <typename data_t>
    typename SiddonsView<data_t>::SiddonsIterator&
        SiddonsView<data_t>::SiddonsIterator::operator++()
    {
        ++iter_;
        return *this;
    }

    template <typename data_t>
    typename SiddonsView<data_t>::SiddonsIterator
        SiddonsView<data_t>::SiddonsIterator::operator++(int)
    {
        auto copy = *this;
        ++iter_;
        return copy;
    }

    template <typename data_t>
    bool operator==(const typename SiddonsView<data_t>::SiddonsIterator& lhs,
                    typename SiddonsView<data_t>::SiddonsIterator rhs)
    {
        return lhs.iter_ == rhs.iter_;
    }

    template <typename data_t>
    bool operator!=(const typename SiddonsView<data_t>::SiddonsIterator& lhs,
                    typename SiddonsView<data_t>::SiddonsIterator rhs)
    {
        return !(lhs == rhs);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SiddonsMethod<float>;
    template class SiddonsMethod<double>;

    template class SiddonsView<float>;
    template class SiddonsView<double>;
} // namespace elsa
