#include "SiddonsMethodBranchless.h"
#include "Timer.h"
#include "TraverseAABBBranchless.h"
#include "TypeCasts.hpp"

#include <stdexcept>
#include <type_traits>

namespace elsa
{
    template <typename data_t>
    SiddonsMethodBranchless<data_t>::SiddonsMethodBranchless(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
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
    SiddonsMethodBranchless<data_t>* SiddonsMethodBranchless<data_t>::_cloneImpl() const
    {
        return new self_type(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                             downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t>
    bool SiddonsMethodBranchless<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherSM = downcast_safe<SiddonsMethodBranchless>(&other);
        return static_cast<bool>(otherSM);
    }

    template <typename data_t>
    data_t SiddonsMethodBranchless<data_t>::traverseRayForward(BoundingBox aabb,
                                                               const RealRay_t& ray,
                                                               const DataContainer<data_t>& x) const
    {
        const auto& domain = x.getDataDescriptor();

        if (domain.getNumberOfDimensions() == 2) {
            return doTraverseRayForward<2>(aabb, ray, x, domain);
        } else if (domain.getNumberOfDimensions() == 3) {
            return doTraverseRayForward<3>(aabb, ray, x, domain);
        }

        return data_t(0);
    }

    template <typename data_t>
    void SiddonsMethodBranchless<data_t>::traverseRayBackward(BoundingBox aabb,
                                                              const RealRay_t& ray,
                                                              const value_type& detectorValue,
                                                              DataContainer<data_t>& Aty) const
    {
        const auto& domain = Aty.getDataDescriptor();

        if (domain.getNumberOfDimensions() == 2) {
            doTraverseRayBackward<2>(aabb, ray, detectorValue, Aty);
        } else if (domain.getNumberOfDimensions() == 3) {
            doTraverseRayBackward<3>(aabb, ray, detectorValue, Aty);
        }
    }

    template <typename data_t>
    template <int dim>
    data_t SiddonsMethodBranchless<data_t>::doTraverseRayForward(BoundingBox aabb,
                                                                 const RealRay_t& ray,
                                                                 const DataContainer<data_t>& x,
                                                                 const DataDescriptor& domain) const
    {
        // --> setup traversal algorithm
        TraverseAABBBranchless<dim> traverse(aabb, ray,
                                             domain.getProductOfCoefficientsPerDimension());

        data_t accumulator = data_t(0);

        // --> initial index to access the data vector
        auto dataIndexForCurrentVoxel = traverse.getCurrentIndex();

        while (traverse.isInBoundingBox()) {

            auto weight = traverse.updateTraverseAndGetDistance();
            // --> update result depending on the operation performed
            accumulator += x[dataIndexForCurrentVoxel] * weight;

            dataIndexForCurrentVoxel = traverse.getCurrentIndex();
        }

        return accumulator;
    }

    template <typename data_t>
    template <int dim>
    void SiddonsMethodBranchless<data_t>::doTraverseRayBackward(BoundingBox aabb,
                                                                const RealRay_t& ray,
                                                                const value_type& detectorValue,
                                                                DataContainer<data_t>& Aty) const
    {
        const auto& domain = Aty.getDataDescriptor();

        // --> setup traversal algorithm
        TraverseAABBBranchless<dim> traverse(aabb, ray,
                                             domain.getProductOfCoefficientsPerDimension());
        // --> initial index to access the data vector
        auto dataIndexForCurrentVoxel = traverse.getCurrentIndex();

        while (traverse.isInBoundingBox()) {

            auto weight = traverse.updateTraverseAndGetDistance();

#pragma omp atomic
            Aty[dataIndexForCurrentVoxel] += detectorValue * weight;

            dataIndexForCurrentVoxel = traverse.getCurrentIndex();
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SiddonsMethodBranchless<float>;
    template class SiddonsMethodBranchless<double>;
} // namespace elsa
