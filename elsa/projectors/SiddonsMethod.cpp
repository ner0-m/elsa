#include "SiddonsMethod.h"
#include "Timer.h"
#include "TraverseAABB.h"

#include <stdexcept>
#include <type_traits>

namespace elsa
{
    template <typename data_t>
    SiddonsMethod<data_t>::SiddonsMethod(const VolumeDescriptor& domainDescriptor,
                                         const DetectorDescriptor& rangeDescriptor)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _boundingBox(domainDescriptor.getNumberOfCoefficientsPerDimension()),
          _detectorDescriptor(static_cast<DetectorDescriptor&>(*_rangeDescriptor)),
          _volumeDescriptor(static_cast<VolumeDescriptor&>(*_domainDescriptor))
    {
        auto dim = _domainDescriptor->getNumberOfDimensions();
        if (dim != _rangeDescriptor->getNumberOfDimensions()) {
            throw LogicError("SiddonsMethod: domain and range dimension need to match");
        }

        if (dim != 2 && dim != 3) {
            throw LogicError("SiddonsMethod: only supporting 2d/3d operations");
        }

        if (_detectorDescriptor.getNumberOfGeometryPoses() == 0) {
            throw LogicError("SiddonsMethod: geometry list was empty");
        }
    }

    template <typename data_t>
    void SiddonsMethod<data_t>::applyImpl(const DataContainer<data_t>& x,
                                          DataContainer<data_t>& Ax) const
    {
        Timer t("SiddonsMethod", "apply");
        traverseVolume<false>(x, Ax);
    }

    template <typename data_t>
    void SiddonsMethod<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                 DataContainer<data_t>& Aty) const
    {
        Timer t("SiddonsMethod", "applyAdjoint");
        traverseVolume<true>(y, Aty);
    }

    template <typename data_t>
    SiddonsMethod<data_t>* SiddonsMethod<data_t>::cloneImpl() const
    {
        return new SiddonsMethod(_volumeDescriptor, _detectorDescriptor);
    }

    template <typename data_t>
    bool SiddonsMethod<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherSM = dynamic_cast<const SiddonsMethod*>(&other);
        if (!otherSM)
            return false;

        return true;
    }

    template <typename data_t>
    template <bool adjoint>
    void SiddonsMethod<data_t>::traverseVolume(const DataContainer<data_t>& vector,
                                               DataContainer<data_t>& result) const
    {
        const index_t maxIterations = adjoint ? vector.getSize() : result.getSize();

        if constexpr (adjoint) {
            result = 0; // initialize volume to 0, because we are not going to hit every voxel!
        }

        // --> loop either over every voxel that should  updated or every detector
        // cell that should be calculated
#pragma omp parallel for
        for (index_t rangeIndex = 0; rangeIndex < maxIterations; ++rangeIndex) {
            // --> get the current ray to the detector center
            const auto ray = _detectorDescriptor.computeRayFromDetectorCoord(rangeIndex);

            // --> setup traversal algorithm
            TraverseAABB traverse(_boundingBox, ray);

            if constexpr (!adjoint)
                result[rangeIndex] = 0;

            // --> initial index to access the data vector
            auto dataIndexForCurrentVoxel =
                _domainDescriptor->getIndexFromCoordinate(traverse.getCurrentVoxel());

            while (traverse.isInBoundingBox()) {

                auto weight = traverse.updateTraverseAndGetDistance();
                // --> update result depending on the operation performed
                if constexpr (adjoint)
#pragma omp atomic
                    result[dataIndexForCurrentVoxel] += vector[rangeIndex] * weight;
                else
                    result[rangeIndex] += vector[dataIndexForCurrentVoxel] * weight;

                dataIndexForCurrentVoxel =
                    _domainDescriptor->getIndexFromCoordinate(traverse.getCurrentVoxel());
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SiddonsMethod<float>;
    template class SiddonsMethod<double>;
} // namespace elsa
