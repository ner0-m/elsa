#include "JosephsMethodBranchless.h"
#include "Timer.h"
#include "DrivingDirectionTraversalBranchless.h"
#include "Error.h"
#include "TypeCasts.hpp"

#include <type_traits>

namespace elsa
{
    template <typename data_t>
    JosephsMethodBranchless<data_t>::JosephsMethodBranchless(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : base_type(domainDescriptor, rangeDescriptor)
    {
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim != 2 && dim != 3) {
            throw InvalidArgumentError("JosephsMethodBranchless:only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError(
                "JosephsMethodBranchless: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("JosephsMethodBranchless: geometry list was empty");
        }
    }

    template <int dim>
    bool isInAABB(const IndexArray_t<dim>& indices, const IndexArray_t<dim>& aabbMin,
                  const IndexArray_t<dim>& aabbMax)
    {
        return (indices >= aabbMin && indices < aabbMax).all();
    }

    template <int dim>
    std::pair<RealArray_t<dim>, RealArray_t<dim>>
        getLinearInterpolationWeights(const RealArray_t<dim>& currentPos,
                                      const IndexArray_t<dim>& voxelFloor,
                                      const index_t drivingDirection)
    {
        // subtract 0.5 because the weight calculation assumes indices that refer to the center of
        // the voxels, while elsa works with the lower corners of the indices.
        RealArray_t<dim> complement_weight = currentPos - voxelFloor.template cast<real_t>() - 0.5f;
        RealArray_t<dim> weight = RealArray_t<dim>{1} - complement_weight;
        // set weights along drivingDirection to 1 so that the interpolation does not have to handle
        // the drivingDirection as a special case
        weight(drivingDirection) = 1;
        complement_weight(drivingDirection) = 1;
        return std::make_pair(weight, complement_weight);
    }

    template <typename data_t, int dim, class Fn>
    void doInterpolation(const IndexArray_t<dim>& voxelFloor, const IndexArray_t<dim>& voxelCeil,
                         const RealArray_t<dim>& weight, const RealArray_t<dim>& complement_weight,
                         const IndexArray_t<dim>& aabbMin, const IndexArray_t<dim>& aabbMax, Fn fn)
    {
        auto clip = [](auto coord, auto lower, auto upper) { return coord.min(upper).max(lower); };
        IndexArray_t<dim> tempIndices;
        if constexpr (dim == 2) {
            auto interpol = [&](auto v1, auto v2, auto w1, auto w2) {
                tempIndices << v1, v2;

                bool is_in_aab = isInAABB(tempIndices, aabbMin, aabbMax);
                tempIndices = clip(tempIndices, aabbMin, (aabbMax - 1));
                auto weight = is_in_aab * w1 * w2;
                fn(tempIndices, weight);
            };

            interpol(voxelFloor[0], voxelCeil[1], weight[0], complement_weight[1]);
            interpol(voxelCeil[0], voxelFloor[1], complement_weight[0], weight[1]);
        } else {
            auto interpol = [&](auto v1, auto v2, auto v3, auto w1, auto w2, auto w3) {
                tempIndices << v1, v2, v3;

                bool is_in_aab = isInAABB(tempIndices, aabbMin, aabbMax);
                tempIndices = clip(tempIndices, aabbMin, (aabbMax - 1));
                auto weight = is_in_aab * w1 * w2 * w3;
                fn(tempIndices, weight);
            };

            interpol(voxelFloor[0], voxelFloor[1], voxelFloor[2], weight[0], weight[1], weight[2]);
            interpol(voxelFloor[0], voxelCeil[1], voxelCeil[2], weight[0], complement_weight[1],
                     complement_weight[2]);
            interpol(voxelCeil[0], voxelFloor[1], voxelCeil[2], complement_weight[0], weight[1],
                     complement_weight[2]);
            interpol(voxelCeil[0], voxelCeil[1], voxelFloor[2], complement_weight[0],
                     complement_weight[1], weight[2]);
        }
    }

    template <typename data_t>
    void JosephsMethodBranchless<data_t>::forward(const BoundingBox& aabb,
                                                  const DataContainer<data_t>& x,
                                                  DataContainer<data_t>& Ax) const
    {
        Timer timeguard("JosephsMethodBranchless", "apply");
        if (aabb.dim() == 2) {
            traverseVolume<false, 2>(aabb, x, Ax);
        } else if (aabb.dim() == 3) {
            traverseVolume<false, 3>(aabb, x, Ax);
        }
    }

    template <typename data_t>
    void JosephsMethodBranchless<data_t>::backward(const BoundingBox& aabb,
                                                   const DataContainer<data_t>& y,
                                                   DataContainer<data_t>& Aty) const
    {
        Timer timeguard("JosephsMethodBranchless", "applyAdjoint");
        if (aabb.dim() == 2) {
            traverseVolume<true, 2>(aabb, y, Aty);
        } else if (aabb.dim() == 3) {
            traverseVolume<true, 3>(aabb, y, Aty);
        }
    }

    template <typename data_t>
    JosephsMethodBranchless<data_t>* JosephsMethodBranchless<data_t>::_cloneImpl() const
    {
        return new self_type(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                             downcast<DetectorDescriptor>(*this->_rangeDescriptor));
    }

    template <typename data_t>
    bool JosephsMethodBranchless<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherJM = downcast_safe<JosephsMethodBranchless>(&other);
        return static_cast<bool>(otherJM);
    }

    template <typename data_t>
    template <bool adjoint, int dim>
    void JosephsMethodBranchless<data_t>::traverseVolume(const BoundingBox& aabb,
                                                         const DataContainer<data_t>& vector,
                                                         DataContainer<data_t>& result) const
    {
        if constexpr (adjoint)
            result = 0;

        const auto& domain = adjoint ? result.getDataDescriptor() : vector.getDataDescriptor();
        const auto& range = downcast<DetectorDescriptor>(adjoint ? vector.getDataDescriptor()
                                                                 : result.getDataDescriptor());

        const auto sizeOfRange = range.getNumberOfCoefficients();

        const IndexArray_t<dim> aabbMin = aabb.min().template cast<index_t>();
        const IndexArray_t<dim> aabbMax = aabb.max().template cast<index_t>();

        // iterate over all rays
#pragma omp parallel for
        for (index_t ir = 0; ir < sizeOfRange; ir++) {
            const auto ray = range.computeRayFromDetectorCoord(ir);

            // --> setup traversal algorithm

            DrivingDirectionTraversalBranchless<dim> traverse(aabb, ray);
            const index_t drivingDirection = traverse.getDrivingDirection();
            const data_t intersection = traverse.getIntersectionLength();

            if constexpr (!adjoint)
                result[ir] = 0;

            // Make steps through the volume
            while (traverse.isInBoundingBox()) {
                const IndexArray_t<dim> voxelFloor = traverse.getCurrentVoxelFloor();
                const IndexArray_t<dim> voxelCeil = traverse.getCurrentVoxelCeil();
                const auto [weight, complement_weight] = getLinearInterpolationWeights(
                    traverse.getCurrentPos(), voxelFloor, drivingDirection);

                if constexpr (adjoint) {
                    doInterpolation<data_t>(voxelFloor, voxelCeil, weight, complement_weight,
                                            aabbMin, aabbMax, [&](const auto& coord, auto wght) {
#pragma omp atomic
                                                result(coord) += vector[ir] * intersection * wght;
                                            });
                } else {
                    doInterpolation<data_t>(voxelFloor, voxelCeil, weight, complement_weight,
                                            aabbMin, aabbMax, [&](const auto& coord, auto wght) {
                                                result[ir] += vector(coord) * intersection * wght;
                                            });
                }
                // update Traverse
                traverse.updateTraverse();
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class JosephsMethodBranchless<float>;
    template class JosephsMethodBranchless<double>;

} // namespace elsa
