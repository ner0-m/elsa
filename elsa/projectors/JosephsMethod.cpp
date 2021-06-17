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
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _boundingBox{domainDescriptor.getNumberOfCoefficientsPerDimension()},
          _detectorDescriptor(static_cast<DetectorDescriptor&>(*_rangeDescriptor)),
          _volumeDescriptor(static_cast<VolumeDescriptor&>(*_domainDescriptor)),
          _interpolation{interpolation}
    {
        auto dim = _domainDescriptor->getNumberOfDimensions();
        if (dim != 2 && dim != 3) {
            throw InvalidArgumentError("JosephsMethod:only supporting 2d/3d operations");
        }

        if (dim != _rangeDescriptor->getNumberOfDimensions()) {
            throw InvalidArgumentError("JosephsMethod: domain and range dimension need to match");
        }

        if (_detectorDescriptor.getNumberOfGeometryPoses() == 0) {
            throw InvalidArgumentError("JosephsMethod: geometry list was empty");
        }
    }

    template <typename data_t>
    void JosephsMethod<data_t>::applyImpl(const DataContainer<data_t>& x,
                                          DataContainer<data_t>& Ax) const
    {
        Timer<> timeguard("JosephsMethod", "apply");
        traverseVolume<false>(x, Ax);
    }

    template <typename data_t>
    void JosephsMethod<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                 DataContainer<data_t>& Aty) const
    {
        Timer<> timeguard("JosephsMethod", "applyAdjoint");
        traverseVolume<true>(y, Aty);
    }

    template <typename data_t>
    JosephsMethod<data_t>* JosephsMethod<data_t>::cloneImpl() const
    {
        return new JosephsMethod(_volumeDescriptor, _detectorDescriptor, _interpolation);
    }

    template <typename data_t>
    bool JosephsMethod<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherJM = downcast_safe<JosephsMethod>(&other);
        if (!otherJM)
            return false;

        return true;
    }

    template <typename data_t>
    template <bool adjoint>
    void JosephsMethod<data_t>::traverseVolume(const DataContainer<data_t>& vector,
                                               DataContainer<data_t>& result) const
    {
        if constexpr (adjoint)
            result = 0;

        const auto sizeOfRange = _rangeDescriptor->getNumberOfCoefficients();
        const auto rangeDim = _rangeDescriptor->getNumberOfDimensions();

        // iterate over all rays
#pragma omp parallel for
        for (index_t ir = 0; ir < sizeOfRange; ir++) {
            const auto ray = _detectorDescriptor.computeRayFromDetectorCoord(ir);

            // --> setup traversal algorithm
            TraverseAABBJosephsMethod traverse(_boundingBox, ray);

            if constexpr (!adjoint)
                result[ir] = 0;

            // Make steps through the volume
            while (traverse.isInBoundingBox()) {

                IndexVector_t currentVoxel = traverse.getCurrentVoxel();
                float intersection = traverse.getIntersectionLength();

                // to avoid code duplicates for apply and applyAdjoint
                auto [to, from] = [&]() {
                    const auto curVoxIndex =
                        _domainDescriptor->getIndexFromCoordinate(currentVoxel);
                    return adjoint ? std::pair{curVoxIndex, ir} : std::pair{ir, curVoxIndex};
                }();

                // #dfrank: This supresses a clang-tidy warning:
                // "error: reference to local binding 'to' declared in enclosing
                // context [clang-diagnostic-error]", which seems to be based on
                // clang-8, and a bug in the standard, as structured bindings are not
                // "real" variables, but only references. I'm not sure why a warning is
                // generated though
                auto tmpTo = to;
                auto tmpFrom = from;

                switch (_interpolation) {
                    case Interpolation::LINEAR:
                        linear<adjoint>(vector, result, traverse.getFractionals(), rangeDim,
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
    void JosephsMethod<data_t>::linear(const DataContainer<data_t>& vector,
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
                    if (interpolVal < _boundingBox._min(i) || interpolVal > _boundingBox._max(i))
                        interpol(i) = static_cast<index_t>(_boundingBox._min(i));
                    else if (interpolVal == _boundingBox._max(i))
                        interpol(i) = static_cast<index_t>(_boundingBox._max(i)) - 1;
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
                if (interpolVal < _boundingBox._min(i) || interpolVal > _boundingBox._max(i))
                    interpol(i) = static_cast<index_t>(_boundingBox._min(i));
                else if (interpolVal == _boundingBox._max(i))
                    interpol(i) = static_cast<index_t>(_boundingBox._max(i)) - 1;

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
