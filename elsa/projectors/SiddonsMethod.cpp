#include "SiddonsMethod.h"
#include "Timer.h"
#include "TraverseAABB.h"
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
    data_t SiddonsMethod<data_t>::traverseRayForward(BoundingBox aabb, const RealRay_t& ray,
                                                     const DataContainer<data_t>& x) const
    {
        const auto& domain = x.getDataDescriptor();

        // --> setup traversal algorithm
        TraverseAABB traverse(aabb, ray);

        data_t accumulator = data_t(0);

        // --> initial index to access the data vector
        auto dataIndexForCurrentVoxel = domain.getIndexFromCoordinate(traverse.getCurrentVoxel());

        while (traverse.isInBoundingBox()) {

            auto weight = traverse.updateTraverseAndGetDistance();
            // --> update result depending on the operation performed
            accumulator += x[dataIndexForCurrentVoxel] * weight;

            dataIndexForCurrentVoxel = domain.getIndexFromCoordinate(traverse.getCurrentVoxel());
        }

        return accumulator;
    }

    template <typename data_t>
    void SiddonsMethod<data_t>::traverseRayBackward(BoundingBox aabb, const RealRay_t& ray,
                                                    const value_type& detectorValue,
                                                    DataContainer<data_t>& Aty) const
    {
        const auto& domain = Aty.getDataDescriptor();

        // --> setup traversal algorithm
        TraverseAABB traverse(aabb, ray);

        // --> initial index to access the data vector
        auto dataIndexForCurrentVoxel = domain.getIndexFromCoordinate(traverse.getCurrentVoxel());

        while (traverse.isInBoundingBox()) {

            auto weight = traverse.updateTraverseAndGetDistance();

#pragma omp atomic
            Aty[dataIndexForCurrentVoxel] += detectorValue * weight;

            dataIndexForCurrentVoxel = domain.getIndexFromCoordinate(traverse.getCurrentVoxel());
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SiddonsMethod<float>;
    template class SiddonsMethod<double>;
} // namespace elsa
