#include "JosephsMethod.h"
#include "Timer.h"
#include "TraverseAABBJosephsMethod.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    JosephsMethod<data_t>::JosephsMethod(const DataDescriptor& domainDescriptor,
                                         const DataDescriptor& rangeDescriptor,
                                         const std::vector<Geometry>& geometryList,
                                         Interpolation interpolation)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _geometryList{geometryList},
          _boundingBox{domainDescriptor.getNumberOfCoefficientsPerDimension()},
          _interpolation{interpolation}
    {
        auto dim = _domainDescriptor->getNumberOfDimensions();
        if (dim != 2 && dim != 3) {
            throw std::invalid_argument("JosephsMethod:only supporting 2d/3d operations");
        }

        if (dim != _rangeDescriptor->getNumberOfDimensions()) {
            throw std::invalid_argument("JosephsMethod: domain and range dimension need to match");
        }

        if (_geometryList.empty()) {
            throw std::invalid_argument("JosephsMethod: geometry list was empty");
        }
    }

    template <typename data_t>
    void JosephsMethod<data_t>::_apply(const DataContainer<data_t>& x,
                                       DataContainer<data_t>& Ax) const
    {
        Timer<> timeguard("JosephsMethod", "apply");
        traverseVolume<false>(x, Ax);
    }

    template <typename data_t>
    void JosephsMethod<data_t>::_applyAdjoint(const DataContainer<data_t>& y,
                                              DataContainer<data_t>& Aty) const
    {
        Timer<> timeguard("JosephsMethod", "applyAdjoint");
        traverseVolume<true>(y, Aty);
    }

    template <typename data_t>
    JosephsMethod<data_t>* JosephsMethod<data_t>::cloneImpl() const
    {
        return new JosephsMethod(*_domainDescriptor, *_rangeDescriptor, _geometryList);
    }

    template <typename data_t>
    bool JosephsMethod<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherJM = dynamic_cast<const JosephsMethod*>(&other);
        if (!otherJM)
            return false;

        if (_geometryList != otherJM->_geometryList || _interpolation != otherJM->_interpolation)
            return false;

        return true;
    }

    template <typename data_t>
    template <bool adjoint>
    void JosephsMethod<data_t>::traverseVolume(const DataContainer<data_t>& vector,
                                               DataContainer<data_t>& result) const
    {
        if (adjoint)
            result = 0;

        const index_t sizeOfRange = _rangeDescriptor->getNumberOfCoefficients();
        const auto rangeDim = _rangeDescriptor->getNumberOfDimensions();

        // iterate over all rays
#pragma omp parallel for
        for (index_t ir = 0; ir < sizeOfRange; ir++) {
            Ray ray = computeRayToDetector(ir, rangeDim);

            // --> setup traversal algorithm
            TraverseAABBJosephsMethod traverse(_boundingBox, ray);

            if (!adjoint)
                result[ir] = 0;

            // Make steps through the volume
            while (traverse.isInBoundingBox()) {

                IndexVector_t currentVoxel = traverse.getCurrentVoxel();
                float intersection = traverse.getIntersectionLength();

                // to avoid code duplicates for apply and applyAdjoint
                index_t from;
                index_t to;
                if (adjoint) {
                    to = _domainDescriptor->getIndexFromCoordinate(currentVoxel);
                    from = ir;
                } else {
                    to = ir;
                    from = _domainDescriptor->getIndexFromCoordinate(currentVoxel);
                }

                switch (_interpolation) {
                    case Interpolation::LINEAR:
                        LINEAR(vector, result, traverse.getFractionals(), adjoint, rangeDim,
                               currentVoxel, intersection, from, to, traverse.getIgnoreDirection());
                        break;
                    case Interpolation::NN:
                        if (adjoint) {
#pragma omp atomic
                            result[to] += intersection * vector[from];
                        } else {
                            result[to] += intersection * vector[from];
                        }
                        break;
                }

                // update Traverse
                traverse.updateTraverse();
            }
        }
    }

    template <typename data_t>
    typename JosephsMethod<data_t>::Ray
        JosephsMethod<data_t>::computeRayToDetector(index_t detectorIndex, index_t dimension) const
    {
        auto detectorCoord = _rangeDescriptor->getCoordinateFromIndex(detectorIndex);

        // center of detector pixel is 0.5 units away from the corresponding detector coordinates
        auto geometry = _geometryList.at(detectorCoord(dimension - 1));
        auto [ro, rd] = geometry.computeRayTo(
            detectorCoord.block(0, 0, dimension - 1, 1).template cast<real_t>().array() + 0.5);

        return Ray(ro, rd);
    }

    template <typename data_t>
    void JosephsMethod<data_t>::LINEAR(const DataContainer<data_t>& vector,
                                       DataContainer<data_t>& result,
                                       const RealVector_t& fractionals, bool adjoint, int domainDim,
                                       const IndexVector_t& currentVoxel, float intersection,
                                       index_t from, index_t to, int mainDirection) const
    {
        float weight = intersection;
        IndexVector_t interpol = currentVoxel;

        // handle diagonal if 3D
        if (domainDim == 3) {
            for (int i = 0; i < domainDim; i++) {
                if (i != mainDirection) {
                    weight *= fabs(fractionals(i));
                    interpol(i) += (fractionals(i) < 0.0) ? -1 : 1;
                    // mirror values at border if outside the volume
                    if (interpol(i) < _boundingBox._min(i) || interpol(i) > _boundingBox._max(i))
                        interpol(i) = _boundingBox._min(i);
                    else if (interpol(i) == _boundingBox._max(i))
                        interpol(i) = _boundingBox._max(i) - 1;
                }
            }
            if (adjoint) {
#pragma omp atomic
                result(interpol) += weight * vector[from];
            } else {
                result[to] += weight * vector(interpol);
            }
        }

        // handle current voxel
        weight = intersection * (1 - fractionals.array().abs()).prod()
                 / (1 - fabs(fractionals(mainDirection)));
        if (adjoint) {
#pragma omp atomic
            result[to] += weight * vector[from];
        } else {
            result[to] += weight * vector[from];
        }

        // handle neighbors not along the main direction
        for (int i = 0; i < domainDim; i++) {
            if (i != mainDirection) {
                float weightn = weight * fabs(fractionals(i)) / (1 - fabs(fractionals(i)));
                interpol = currentVoxel;
                interpol(i) += (fractionals(i) < 0.0) ? -1 : 1;

                // mirror values at border if outside the volume
                if (interpol(i) < _boundingBox._min(i) || interpol(i) > _boundingBox._max(i))
                    interpol(i) = _boundingBox._min(i);
                else if (interpol(i) == _boundingBox._max(i))
                    interpol(i) = _boundingBox._max(i) - 1;

                if (adjoint) {
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
