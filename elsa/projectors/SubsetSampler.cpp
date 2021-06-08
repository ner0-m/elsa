#include <set>
#include "SubsetSampler.h"
#include "PartitionDescriptor.h"
#include "SiddonsMethod.h"

namespace elsa
{
    template <typename DetectorDescriptor_t, typename data_t>
    SubsetSampler<DetectorDescriptor_t, data_t>::SubsetSampler(
        const VolumeDescriptor& volumeDescriptor, const DetectorDescriptor_t& detectorDescriptor,
        index_t nSubsets, SamplingStrategy samplingStrategy)
        : _indexMapping(static_cast<std::size_t>(nSubsets)),
          _volumeDescriptor(volumeDescriptor),
          _fullDetectorDescriptor(detectorDescriptor),
          _nSubsets{nSubsets}
    {
        if (nSubsets <= 1) {
            throw std::invalid_argument("SubsetSampler: nSubsets must be >= 2");
        }

        // the mapping of data indices to subsets

        if (samplingStrategy == SamplingStrategy::ROUND_ROBIN) {
            _indexMapping = sampleRoundRobin(detectorDescriptor, _nSubsets);
        } else if (samplingStrategy == SamplingStrategy::EQUI_ROTATION) {
            _indexMapping = sampleEquiRotation(detectorDescriptor, _nSubsets);
        } else {
            throw std::invalid_argument("SubsetSampler: unsupported sampling strategy");
        }

        // create the detector descriptors that correspond to each subset
        for (const auto& indices : _indexMapping) {
            std::vector<Geometry> geometry;
            for (auto index : indices) {
                auto geo = detectorDescriptor.getGeometryAt(index);
                if (geo.has_value()) {
                    geometry.emplace_back(*geo);
                }
            }
            IndexVector_t numOfCoeffsPerDim =
                detectorDescriptor.getNumberOfCoefficientsPerDimension();
            numOfCoeffsPerDim[numOfCoeffsPerDim.size() - 1] = static_cast<index_t>(indices.size());

            _detectorDescriptors.emplace_back(DetectorDescriptor_t(numOfCoeffsPerDim, geometry));
        }
    }

    template <typename DetectorDescriptor_t, typename data_t>
    std::vector<std::vector<index_t>> SubsetSampler<DetectorDescriptor_t, data_t>::sampleRoundRobin(
        const DetectorDescriptor_t& detectorDescriptor, index_t nSubsets)
    {
        std::vector<std::vector<index_t>> subsetIndices(static_cast<std::size_t>(nSubsets));

        const auto numCoeffsPerDim = detectorDescriptor.getNumberOfCoefficientsPerDimension();
        const index_t nDimensions = detectorDescriptor.getNumberOfDimensions();
        const auto numElements = numCoeffsPerDim[nDimensions - 1];

        // determine the mapping of indices to subsets
        for (index_t i = 0; i < numElements; ++i) {
            const auto subset = i % nSubsets;
            subsetIndices[static_cast<std::size_t>(subset)].template emplace_back(i);
        }

        return subsetIndices;
    }

    template <typename DetectorDescriptor_t, typename data_t>
    std::vector<std::vector<index_t>>
        SubsetSampler<DetectorDescriptor_t, data_t>::sampleEquiRotation(
            const DetectorDescriptor_t& detectorDescriptor, index_t nSubsets)
    {
        std::vector<std::vector<index_t>> subsetIndices(static_cast<std::size_t>(nSubsets));

        const auto numCoeffsPerDim = detectorDescriptor.getNumberOfCoefficientsPerDimension();
        const index_t nDimensions = detectorDescriptor.getNumberOfDimensions();

        subsetIndices[0].template emplace_back(0);
        std::size_t currentSubset = 1;
        Geometry currGeometry = detectorDescriptor.getGeometryAt(0).value();
        auto nElements = static_cast<index_t>(numCoeffsPerDim[nDimensions - 1]);
        std::set<index_t> taken = {0};

        // angle between two rotation matrices used as a distance measure
        auto dist = [](auto& g1, auto& g2) {
            return std::acos(
                ((g1.getRotationMatrix() * g2.getRotationMatrix().transpose()).trace() - 1.0)
                / 2.0);
        };

        // loop to find the closest (rotationally) point to the current one, putting it in the next
        // subset until all points are distributed
        while (taken.size() < static_cast<std::size_t>(nElements)) {
            index_t closest = -1;
            double minAngle = std::numeric_limits<double>::max();
            for (index_t i = 0; i < nElements; ++i) {
                if (taken.find(i) == taken.end()) {
                    const Geometry geo = detectorDescriptor.getGeometryAt(i).value();
                    auto angle = dist(currGeometry, geo);
                    if (angle < minAngle) {
                        closest = i;
                        minAngle = angle;
                    }
                }
            }

            if (closest < 0) {
                throw std::runtime_error(
                    "SubsetSampler: Error finding smallest rotational difference");
            }

            currGeometry = detectorDescriptor.getGeometryAt(closest).value();
            taken.emplace(closest);
            subsetIndices[currentSubset].template emplace_back(closest);
            currentSubset = (currentSubset + 1) % static_cast<std::size_t>(nSubsets);
        }

        return subsetIndices;
    }

    template <typename DetectorDescriptor_t, typename data_t>
    SubsetSampler<DetectorDescriptor_t, data_t>::SubsetSampler(
        const SubsetSampler<DetectorDescriptor_t, data_t>& other)
        : _indexMapping{other._indexMapping},
          _volumeDescriptor(other._volumeDescriptor),
          _fullDetectorDescriptor(other._fullDetectorDescriptor),
          _nSubsets{other._nSubsets}
    {
        for (const auto& detectorDescriptor : other._detectorDescriptors) {
            _detectorDescriptors.emplace_back(detectorDescriptor);
        }
    }

    template <typename DetectorDescriptor_t, typename data_t>
    DataContainer<data_t> SubsetSampler<DetectorDescriptor_t, data_t>::getPartitionedData(
        const DataContainer<data_t>& sinogram)
    {
        // save the number of entries per subset
        IndexVector_t slicesInBlock(_indexMapping.size());
        for (unsigned long i = 0; i < _indexMapping.size(); i++) {
            slicesInBlock[static_cast<index_t>(i)] = static_cast<index_t>(_indexMapping[i].size());
        }
        PartitionDescriptor dataDescriptor(sinogram.getDataDescriptor(), slicesInBlock);

        const auto numCoeffsPerDim =
            sinogram.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        auto partitionedData = DataContainer<data_t>(dataDescriptor, sinogram.getDataHandlerType());

        // resort the actual measurement partitionedData
        index_t coeffsPerRow = numCoeffsPerDim.head(numCoeffsPerDim.size() - 1).prod();
        for (index_t i = 0; i < _nSubsets; i++) {
            // the indices of the partitionedData rows belonging to this subset
            std::vector<index_t> indices = _indexMapping[static_cast<std::size_t>(i)];

            auto block = partitionedData.getBlock(i);

            for (std::size_t j = 0; j < indices.size(); j++) {
                for (int k = 0; k < coeffsPerRow; k++) {
                    block[static_cast<index_t>(j) * coeffsPerRow + k] =
                        sinogram[indices[j] * coeffsPerRow + k];
                }
            }
        }
        return partitionedData;
    }

    template <typename DetectorDescriptor_t, typename data_t>
    SubsetSampler<DetectorDescriptor_t, data_t>*
        SubsetSampler<DetectorDescriptor_t, data_t>::cloneImpl() const
    {
        return new SubsetSampler<DetectorDescriptor_t, data_t>(*this);
    }

    template <typename DetectorDescriptor_t, typename data_t>
    bool SubsetSampler<DetectorDescriptor_t, data_t>::isEqual(
        const SubsetSampler<DetectorDescriptor_t, data_t>& other) const
    {
        if (typeid(*this) != typeid(other))
            return false;

        if (_indexMapping != other._indexMapping)
            return false;

        if (_volumeDescriptor != other._volumeDescriptor)
            return false;

        if (_fullDetectorDescriptor != other._fullDetectorDescriptor)
            return false;

        if (_nSubsets != other._nSubsets)
            return false;

        // we do not need to check if the vector of detector descriptors is equal as this is
        // implied by the equality of _data in combination with the full detector descriptor

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SubsetSampler<PlanarDetectorDescriptor, float>;
    template class SubsetSampler<PlanarDetectorDescriptor, double>;
    template class SubsetSampler<PlanarDetectorDescriptor, std::complex<float>>;
    template class SubsetSampler<PlanarDetectorDescriptor, std::complex<double>>;
} // namespace elsa