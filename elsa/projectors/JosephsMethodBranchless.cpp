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

            if constexpr (!adjoint)
                result[ir] = 0;

            // Make steps through the volume
            while (traverse.isInBoundingBox()) {

                IndexArray_t<dim> currentVoxel = traverse.getCurrentVoxel();
                float intersection = traverse.getIntersectionLength();

                // to avoid code duplicates for apply and applyAdjoint
                auto [to, from] = [&]() {
                    const auto curVoxIndex = domain.getIndexFromCoordinate(currentVoxel);
                    return adjoint ? std::pair{curVoxIndex, ir} : std::pair{ir, curVoxIndex};
                }();

                // #dfrank: This suppresses a clang-tidy warning:
                // "error: reference to local binding 'to' declared in enclosing
                // context [clang-diagnostic-error]", which seems to be based on
                // clang-8, and a bug in the standard, as structured bindings are not
                // "real" variables, but only references. I'm not sure why a warning is
                // generated though
                auto tmpTo = to;
                auto tmpFrom = from;

                linear<adjoint, dim>(result, vector, traverse.getCurrentPos(),
                                     traverse.getCurrentVoxelFloor(),
                                     traverse.getCurrentVoxelCeil(), traverse.getIgnoreDirection(),
                                     to, ir, intersection, aabbMin, aabbMax);
                // update Traverse
                traverse.updateTraverse();
            }
        }
    }

    template <typename data_t>
    template <bool adjoint, int dim>
    void JosephsMethodBranchless<data_t>::linear(
        DataContainer<data_t>& result, const DataContainer<data_t>& vector,
        RealArray_t<dim> currentPos, IndexArray_t<dim> voxelFloor, IndexArray_t<dim> voxelCeil,
        index_t drivingDirection, index_t to, index_t ir, real_t intersection,
        const IndexArray_t<dim> aabbMin, const IndexArray_t<dim> aabbMax) const
    {
        RealArray_t<dim> complement_weight = currentPos - voxelFloor.template cast<real_t>() - 0.5f;
        RealArray_t<dim> weight = RealArray_t<dim>{1} - complement_weight;
        weight(drivingDirection) = 1;
        complement_weight(drivingDirection) = 1;
        IndexArray_t<dim> indices;
        if constexpr (dim == 2) {
            if constexpr (adjoint) {
                indices << voxelFloor[0], voxelCeil[1];
                bool is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent ouf of bounds access
                indices << (is_in_aabb ? indices[0] : 0), (is_in_aabb ? indices[1] : 0);
                to = is_in_aabb * (to >= 0)
                     * (to < (aabbMax[0] - aabbMin[0]) * (aabbMax[1] - aabbMin[1])) * to;
#pragma omp atomic
                result(indices) +=
                    is_in_aabb * vector[ir] * intersection * weight[0] * complement_weight[1];

                indices << voxelCeil[0], voxelFloor[1];
                is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent out of bounds access
                indices << (is_in_aabb ? indices[0] : 0), (is_in_aabb ? indices[1] : 0);
                to = is_in_aabb * (to >= 0)
                     * (to < (aabbMax[0] - aabbMin[0]) * (aabbMax[1] - aabbMin[1])) * to;
#pragma omp atomic
                result(indices) +=
                    is_in_aabb * vector[ir] * intersection * complement_weight[0] * weight[1];
            } else {
                indices << voxelFloor[0], voxelCeil[1];
                bool is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent ouf of bounds access
                indices << (is_in_aabb ? indices[0] : 0), (is_in_aabb ? indices[1] : 0);
                result[to] +=
                    is_in_aabb * vector(indices) * intersection * weight[0] * complement_weight[1];

                indices << voxelCeil[0], voxelFloor[1];
                is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent out of bounds access
                indices << (is_in_aabb ? indices[0] : 0), (is_in_aabb ? indices[1] : 0);
                result[to] +=
                    is_in_aabb * vector(indices) * intersection * complement_weight[0] * weight[1];
            }
        } else {
            if constexpr (adjoint) {
                bool is_in_aabb = (voxelFloor >= aabbMin && voxelFloor < aabbMax).all();
                // prevent out of bounds access
                indices << (is_in_aabb ? voxelFloor : IndexArray_t<dim>(0));
                result(indices) +=
                    is_in_aabb * vector[ir] * intersection * weight[0] * weight[1] * weight[2];

                indices << voxelFloor[0], voxelCeil[1], voxelCeil[2];
                is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent out ouf bounds access
                indices << (is_in_aabb ? indices : IndexArray_t<dim>(0));
                result(indices) += is_in_aabb * vector[ir] * intersection * weight[0]
                                   * complement_weight[1] * complement_weight[2];

                indices << voxelCeil[0], voxelFloor[1], voxelCeil[2];
                is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent out of bounds access
                indices << (is_in_aabb ? indices : IndexArray_t<dim>(0));
                result(indices) += is_in_aabb * vector[ir] * intersection * complement_weight[0]
                                   * weight[1] * complement_weight[2];

                indices << voxelCeil[0], voxelCeil[1], voxelFloor[2];
                is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent out of bounds access
                indices << (is_in_aabb ? indices : IndexArray_t<dim>(0));
                result(indices) += is_in_aabb * vector[ir] * intersection * complement_weight[0]
                                   * complement_weight[1] * weight[2];
            } else {
                bool is_in_aabb = (voxelFloor >= aabbMin && voxelFloor < aabbMax).all();
                // prevent out of bounds access
                indices << (is_in_aabb ? voxelFloor : IndexArray_t<dim>(0));
                result[to] +=
                    is_in_aabb * vector(indices) * intersection * weight[0] * weight[1] * weight[2];

                indices << voxelFloor[0], voxelCeil[1], voxelCeil[2];
                is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent out ouf bounds access
                indices << (is_in_aabb ? indices : IndexArray_t<dim>(0));
                result[to] += is_in_aabb * vector(indices) * intersection * weight[0]
                              * complement_weight[1] * complement_weight[2];

                indices << voxelCeil[0], voxelFloor[1], voxelCeil[2];
                is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent out of bounds access
                indices << (is_in_aabb ? indices : IndexArray_t<dim>(0));
                result[to] += is_in_aabb * vector(indices) * intersection * complement_weight[0]
                              * weight[1] * complement_weight[2];

                indices << voxelCeil[0], voxelCeil[1], voxelFloor[2];
                is_in_aabb = (indices >= aabbMin && indices < aabbMax).all();
                // prevent out of bounds access
                indices << (is_in_aabb ? indices : IndexArray_t<dim>(0));
                result[to] += is_in_aabb * vector(indices) * intersection * complement_weight[0]
                              * complement_weight[1] * weight[2];
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class JosephsMethodBranchless<float>;
    template class JosephsMethodBranchless<double>;

} // namespace elsa
