#include "JosephsMethod.h"
#include "Timer.h"
#include "TraverseAABBJosephsMethod.h"
#include "Error.h"
#include "TypeCasts.hpp"

#include <type_traits>

namespace elsa
{
    template <typename data_t>
    JosephsMethod<data_t>::JosephsMethod(const VolumeDescriptor& domainDescriptor,
                                         const DetectorDescriptor& rangeDescriptor,
                                         Interpolation interpolation)
        : base_type(domainDescriptor, rangeDescriptor), _interpolation{interpolation}
    {
        auto dim = domainDescriptor.getNumberOfDimensions();
        if (dim != 2 && dim != 3) {
            throw InvalidArgumentError("JosephsMethod:only supporting 2d/3d operations");
        }

        if (dim != rangeDescriptor.getNumberOfDimensions()) {
            throw InvalidArgumentError("JosephsMethod: domain and range dimension need to match");
        }

        if (rangeDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("JosephsMethod: geometry list was empty");
        }
    }

    template <typename data_t>
    void JosephsMethod<data_t>::forward(const BoundingBox& aabb, const DataContainer<data_t>& x,
                                        DataContainer<data_t>& Ax) const
    {
        Timer timeguard("JosephsMethod", "apply");
        traverseVolume<false>(aabb, x, Ax);
    }

    template <typename data_t>
    void JosephsMethod<data_t>::backward(const BoundingBox& aabb, const DataContainer<data_t>& y,
                                         DataContainer<data_t>& Aty) const
    {
        Timer timeguard("JosephsMethod", "applyAdjoint");
        traverseVolume<true>(aabb, y, Aty);
    }

    template <typename data_t>
    JosephsMethod<data_t>* JosephsMethod<data_t>::_cloneImpl() const
    {
        return new self_type(downcast<VolumeDescriptor>(*this->_domainDescriptor),
                             downcast<DetectorDescriptor>(*this->_rangeDescriptor), _interpolation);
    }

    template <typename data_t>
    bool JosephsMethod<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherJM = downcast_safe<JosephsMethod>(&other);
        return static_cast<bool>(otherJM);
    }

    template <typename data_t>
    template <bool adjoint>
    void JosephsMethod<data_t>::traverseVolume(const BoundingBox& aabb,
                                               const DataContainer<data_t>& vector,
                                               DataContainer<data_t>& result) const
    {
        if constexpr (adjoint)
            result = 0;

        const auto& domain = adjoint ? result.getDataDescriptor() : vector.getDataDescriptor();
        const auto& range = downcast<DetectorDescriptor>(adjoint ? vector.getDataDescriptor()
                                                                 : result.getDataDescriptor());

        const auto sizeOfRange = range.getNumberOfCoefficients();
        const auto rangeDim = range.getNumberOfDimensions();

        // iterate over all rays
#pragma omp parallel for
        for (index_t ir = 0; ir < sizeOfRange; ir++) {
            const auto ray = range.computeRayFromDetectorCoord(ir);

            // --> setup traversal algorithm
            TraverseAABBJosephsMethod traverse(aabb, ray);

            if constexpr (!adjoint)
                result[ir] = 0;

            // Make steps through the volume
            while (traverse.isInBoundingBox()) {

                IndexVector_t currentVoxel = traverse.getCurrentVoxel();
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

                switch (_interpolation) {
                    case Interpolation::LINEAR:
                        linear<adjoint>(aabb, vector, result, traverse.getFractionals(), rangeDim,
                                        currentVoxel, intersection, from, to,
                                        traverse.getIgnoreDirection());
                        break;
                    case Interpolation::NN:
                        if constexpr (adjoint) {
#pragma omp atomic
                            // NOLINTNEXTLINE
                            result[tmpTo] += intersection * vector[tmpFrom];
                        } else {
                            result[tmpTo] += intersection * vector[tmpFrom];
                        }
                        break;
                }

                // update Traverse
                traverse.updateTraverse();
            }
        }
    }

    template <typename data_t>
    template <bool adjoint>
    void JosephsMethod<data_t>::linear(const BoundingBox& aabb, const DataContainer<data_t>& vector,
                                       DataContainer<data_t>& result,
                                       const RealVector_t& fractionals, index_t domainDim,
                                       const IndexVector_t& currentVoxel, float intersection,
                                       index_t from, index_t to, int mainDirection) const
    {
        data_t weight = intersection;
        IndexVector_t interpol = currentVoxel;

        // handle diagonal if 3D
        if (domainDim == 3) {
            for (int i = 0; i < domainDim; i++) {
                if (i != mainDirection) {
                    weight *= std::abs(fractionals(i));
                    interpol(i) += (fractionals(i) < 0.0) ? -1 : 1;
                    // mirror values at border if outside the volume
                    auto interpolVal = static_cast<real_t>(interpol(i));
                    if (interpolVal < aabb.min()(i) || interpolVal > aabb.max()(i))
                        interpol(i) = static_cast<index_t>(aabb.min()(i));
                    else if (interpolVal == aabb.max()(i))
                        interpol(i) = static_cast<index_t>(aabb.max()(i)) - 1;
                }
            }
            if constexpr (adjoint) {
#pragma omp atomic
                result(interpol) += weight * vector[from];
            } else {
                result[to] += weight * vector(interpol);
            }
        }

        // handle current voxel
        weight = intersection * (1 - fractionals.array().abs()).prod()
                 / (1 - std::abs(fractionals(mainDirection)));
        if constexpr (adjoint) {
#pragma omp atomic
            result[to] += weight * vector[from];
        } else {
            result[to] += weight * vector[from];
        }

        // handle neighbors not along the main direction
        for (int i = 0; i < domainDim; i++) {
            if (i != mainDirection) {
                data_t weightn = weight * std::abs(fractionals(i)) / (1 - std::abs(fractionals(i)));
                interpol = currentVoxel;
                interpol(i) += (fractionals(i) < 0.0) ? -1 : 1;

                // mirror values at border if outside the volume
                auto interpolVal = static_cast<real_t>(interpol(i));
                if (interpolVal < aabb.min()(i) || interpolVal > aabb.max()(i))
                    interpol(i) = static_cast<index_t>(aabb.min()(i));
                else if (interpolVal == aabb.max()(i))
                    interpol(i) = static_cast<index_t>(aabb.max()(i)) - 1;

                if constexpr (adjoint) {
#pragma omp atomic
                    result(interpol) += weightn * vector[from];
                } else {
                    result[to] += weightn * vector(interpol);
                }
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class JosephsMethod<float>;
    template class JosephsMethod<double>;

} // namespace elsa
