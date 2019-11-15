#include "BinaryMethod.h"
#include "Timer.h"
#include "TraverseAABB.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    BinaryMethod<data_t>::BinaryMethod(const DataDescriptor& domainDescriptor,
                                       const DataDescriptor& rangeDescriptor,
                                       const std::vector<Geometry>& geometryList)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _geometryList{geometryList},
          _boundingBox{domainDescriptor.getNumberOfCoefficientsPerDimension()}
    {
        // sanity checks
        auto dim = _domainDescriptor->getNumberOfDimensions();
        if (dim < 2 || dim > 3)
            throw std::invalid_argument("BinaryMethod: only supporting 2d/3d operations");

        if (dim != _rangeDescriptor->getNumberOfDimensions())
            throw std::invalid_argument("BinaryMethod: domain and range dimension need to match");

        if (_geometryList.empty())
            throw std::invalid_argument("BinaryMethod: geometry list was empty");
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
        return new BinaryMethod(*_domainDescriptor, *_rangeDescriptor, _geometryList);
    }

    template <typename data_t>
    bool BinaryMethod<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherBM = dynamic_cast<const BinaryMethod*>(&other);
        if (!otherBM)
            return false;

        if (_geometryList != otherBM->_geometryList)
            return false;

        return true;
    }

    template <typename data_t>
    template <bool adjoint>
    void BinaryMethod<data_t>::traverseVolume(const DataContainer<data_t>& vector,
                                              DataContainer<data_t>& result) const
    {
        index_t maxIterations{0};
        if (adjoint) {
            maxIterations = vector.getSize();
            result = 0; // initialize volume to 0, because we are not going to hit every voxel!
        } else
            maxIterations = result.getSize();

        const auto rangeDim = _rangeDescriptor->getNumberOfDimensions();

        // --> loop either over every voxel that should be updated or every detector
        // cell that should be calculated
#pragma omp parallel for
        for (size_t rangeIndex = 0; rangeIndex < maxIterations; ++rangeIndex) {

            // --> get the current ray to the detector center
            auto ray = computeRayToDetector(rangeIndex, rangeDim);

            // --> setup traversal algorithm
            TraverseAABB traverse(_boundingBox, ray);

            if (!adjoint)
                result[rangeIndex] = 0;

            while (traverse.isInBoundingBox()) {
                // --> initial index to access the data vector
                auto dataIndexForCurrentVoxel =
                    _domainDescriptor->getIndexFromCoordinate(traverse.getCurrentVoxel());

                // --> update result depending on the operation performed
                if (adjoint)
#pragma omp atomic
                    result[dataIndexForCurrentVoxel] += vector[rangeIndex];
                else
                    result[rangeIndex] += vector[dataIndexForCurrentVoxel];

                traverse.updateTraverse();
            }
        } // end for
    }

    template <typename data_t>
    typename BinaryMethod<data_t>::Ray
        BinaryMethod<data_t>::computeRayToDetector(index_t detectorIndex, index_t dimension) const
    {
        auto detectorCoord = _rangeDescriptor->getCoordinateFromIndex(detectorIndex);

        // center of detector pixel is 0.5 units away from the corresponding detector coordinates
        auto geometry = _geometryList.at(detectorCoord(dimension - 1));
        auto [ro, rd] = geometry.computeRayTo(
            detectorCoord.block(0, 0, dimension - 1, 1).template cast<real_t>().array() + 0.5);

        return Ray(ro, rd);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class BinaryMethod<float>;
    template class BinaryMethod<double>;

} // namespace elsa
