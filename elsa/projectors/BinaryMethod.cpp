#include "BinaryMethod.h"
#include "Timer.h"
#include "TraverseAABB.h"

#include <stdexcept>
#include <type_traits>

namespace elsa
{
    template <typename data_t>
    BinaryMethod<data_t>::BinaryMethod(const VolumeDescriptor& domainDescriptor,
                                       const DetectorDescriptor& rangeDescriptor)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _boundingBox{domainDescriptor.getNumberOfCoefficientsPerDimension()},
          _detectorDescriptor(static_cast<DetectorDescriptor&>(*_rangeDescriptor)),
          _volumeDescriptor(static_cast<VolumeDescriptor&>(*_domainDescriptor))
    {
        // sanity checks
        auto dim = _domainDescriptor->getNumberOfDimensions();
        if (dim < 2 || dim > 3)
            throw std::invalid_argument("BinaryMethod: only supporting 2d/3d operations");

        if (dim != _rangeDescriptor->getNumberOfDimensions())
            throw std::invalid_argument("BinaryMethod: domain and range dimension need to match");

        if (_detectorDescriptor.getNumberOfGeometryPoses() == 0)
            throw std::invalid_argument("BinaryMethod: rangeDescriptor without any geometry");
    }

    template <typename data_t>
    void BinaryMethod<data_t>::applyImpl(const DataContainer<data_t>& x,
                                         DataContainer<data_t>& Ax) const
    {
        Timer t("BinaryMethod", "apply");
        traverseVolume<false>(x, Ax);
    }

    template <typename data_t>
    void BinaryMethod<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                DataContainer<data_t>& Aty) const
    {
        Timer t("BinaryMethod", "applyAdjoint");
        traverseVolume<true>(y, Aty);
    }

    template <typename data_t>
    BinaryMethod<data_t>* BinaryMethod<data_t>::cloneImpl() const
    {
        return new BinaryMethod(_volumeDescriptor, _detectorDescriptor);
    }

    template <typename data_t>
    bool BinaryMethod<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherBM = dynamic_cast<const BinaryMethod*>(&other);
        if (!otherBM)
            return false;

        return true;
    }

    template <typename data_t>
    template <bool adjoint>
    void BinaryMethod<data_t>::traverseVolume(const DataContainer<data_t>& vector,
                                              DataContainer<data_t>& result) const
    {
        const index_t maxIterations = adjoint ? vector.getSize() : result.getSize();

        if constexpr (adjoint) {
            result = 0; // initialize volume to 0, because we are not going to hit every voxel!
        }

        // --> loop either over every voxel that should be updated or every detector
        // cell that should be calculated
#pragma omp parallel for
        for (index_t rangeIndex = 0; rangeIndex < maxIterations; ++rangeIndex) {

            // --> get the current ray to the detector center (from reference to DetectorDescriptor)
            auto ray = _detectorDescriptor.computeRayFromDetectorCoord(rangeIndex);

            // --> setup traversal algorithm
            TraverseAABB traverse(_boundingBox, ray);

            if constexpr (!adjoint)
                result[rangeIndex] = 0;

            while (traverse.isInBoundingBox()) {
                // --> initial index to access the data vector
                auto dataIndexForCurrentVoxel =
                    _domainDescriptor->getIndexFromCoordinate(traverse.getCurrentVoxel());

                // --> update result depending on the operation performed
                if constexpr (adjoint)
#pragma omp atomic
                    result[dataIndexForCurrentVoxel] += vector[rangeIndex];
                else
                    result[rangeIndex] += vector[dataIndexForCurrentVoxel];

                traverse.updateTraverse();
            }
        } // end for
    }

    // ------------------------------------------
    // explicit template instantiation
    template class BinaryMethod<float>;
    template class BinaryMethod<double>;

} // namespace elsa
