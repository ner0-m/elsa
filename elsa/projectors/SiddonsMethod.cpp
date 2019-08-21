#include "SiddonsMethod.h"
#include "Timer.h"
#include "TraverseAABB.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    SiddonsMethod<data_t>::SiddonsMethod(const DataDescriptor& domainDescriptor, const DataDescriptor& rangeDescriptor,
                               const std::vector<Geometry>& geometryList)
            : LinearOperator<data_t>(domainDescriptor, rangeDescriptor), _geometryList{geometryList},
              _boundingBox(domainDescriptor.getNumberOfCoefficientsPerDimension())
    {
        auto dim = _domainDescriptor->getNumberOfDimensions();
        if (dim != _rangeDescriptor->getNumberOfDimensions()) {
            throw std::logic_error("SiddonsMethod: domain and range dimension need to match");
        }

        if (dim!=2 && dim != 3) {
            throw std::logic_error("SiddonsMethod: only supporting 2d/3d operations");
        }

        if (_geometryList.empty()) {
            throw std::logic_error("SiddonsMethod: geometry list was empty");
        }
    }

    template <typename data_t>
    void SiddonsMethod<data_t>::_apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax)
    {
        Timer t("SiddonsMethod", "apply");
        traverseVolume<false>(x, Ax);
    }

    template <typename data_t>
    void SiddonsMethod<data_t>::_applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty)
    {
        Timer t("SiddonsMethod", "applyAdjoint");
        traverseVolume<true>(y, Aty);
    }

    template <typename data_t>
    SiddonsMethod<data_t>* SiddonsMethod<data_t>::cloneImpl() const
    {
        return new SiddonsMethod(*_domainDescriptor, *_rangeDescriptor, _geometryList);
    }

    template <typename data_t>
    bool SiddonsMethod<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherSM = dynamic_cast<const SiddonsMethod*>(&other);
        if (!otherSM)
            return false;

        if (_geometryList != otherSM->_geometryList)
            return false;

        return true;
    }

    template<typename data_t>
    template<bool adjoint>
    void SiddonsMethod<data_t>::traverseVolume(const DataContainer<data_t>& vector, DataContainer<data_t>& result) const
    {
        index_t maxIterations{0};
        if (adjoint) {
            maxIterations = vector.getSize();
            result = 0; // initialize volume to 0, because we are not going to hit every voxel!
        } else
            maxIterations = result.getSize();

        const auto rangeDim = _rangeDescriptor->getNumberOfDimensions();

        // --> loop either over every voxel that should  updated or every detector
        // cell that should be calculated
#pragma omp parallel for
        for (size_t rangeIndex = 0; rangeIndex < maxIterations; ++rangeIndex)
        {
            // --> get the current ray to the detector center
            auto ray = computeRayToDetector(rangeIndex, rangeDim);

            // --> setup traversal algorithm
            TraverseAABB traverse(_boundingBox, ray);

            if(!adjoint) 
                result[rangeIndex] = 0;

            // --> initial index to access the data vector
            auto dataIndexForCurrentVoxel = _domainDescriptor->getIndexFromCoordinate(traverse.getCurrentVoxel());

            while (traverse.isInBoundingBox())
            {
                
                auto weight = traverse.updateTraverseAndGetDistance();
                // --> update result depending on the operation performed
                if (adjoint)
#pragma omp atomic
                    result[dataIndexForCurrentVoxel] += vector[rangeIndex] * weight;
                else
                    result[rangeIndex] += vector[dataIndexForCurrentVoxel] * weight;
                
                dataIndexForCurrentVoxel = _domainDescriptor->getIndexFromCoordinate(traverse.getCurrentVoxel());
            }
        }
    }

    template <typename data_t>
    typename SiddonsMethod<data_t>::Ray SiddonsMethod<data_t>::computeRayToDetector(index_t detectorIndex, index_t dimension) const
    {
        auto detectorCoord = _rangeDescriptor->getCoordinateFromIndex(detectorIndex);

        //center of detector pixel is 0.5 units away from the corresponding detector coordinates
        auto geometry = _geometryList.at(detectorCoord(dimension - 1));
        auto [ro, rd] = geometry.computeRayTo(detectorCoord.block(0, 0, dimension-1, 1).template cast<real_t>().array() + 0.5);

        return Ray(ro, rd);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SiddonsMethod<float>;
    template class SiddonsMethod<double>;
}
