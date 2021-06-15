#include <set>
#include <numeric>
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

        const auto numCoeffsPerDim = detectorDescriptor.getNumberOfCoefficientsPerDimension();
        const index_t nDimensions = detectorDescriptor.getNumberOfDimensions();
        const auto numElements = numCoeffsPerDim[nDimensions - 1];
        if (samplingStrategy == SamplingStrategy::ROUND_ROBIN) {
            std::vector<index_t> indices(static_cast<std::size_t>(numElements));
            std::iota(indices.begin(), indices.end(), 0);
            _indexMapping = splitRoundRobin(indices, _nSubsets);
        } else if (samplingStrategy == SamplingStrategy::ROTATIONAL_CLUSTERING) {
            _indexMapping = splitRotationalClustering(detectorDescriptor, _nSubsets);
        } else {
            throw std::invalid_argument("SubsetSampler: unsupported sampling strategy");
        }

        // create the detector descriptors that correspond to each subset
        for (const auto& blockIndices : _indexMapping) {
            std::vector<Geometry> geometry;
            for (auto index : blockIndices) {
                auto geo = detectorDescriptor.getGeometryAt(index);
                if (geo.has_value()) {
                    geometry.emplace_back(*geo);
                }
            }
            IndexVector_t numOfCoeffsPerDim =
                detectorDescriptor.getNumberOfCoefficientsPerDimension();
            numOfCoeffsPerDim[numOfCoeffsPerDim.size() - 1] =
                static_cast<index_t>(blockIndices.size());

            _detectorDescriptors.emplace_back(DetectorDescriptor_t(numOfCoeffsPerDim, geometry));
        }
    }

    template <typename DetectorDescriptor_t, typename data_t>
    std::vector<std::vector<index_t>> SubsetSampler<DetectorDescriptor_t, data_t>::splitRoundRobin(
        const std::vector<index_t>& indices, index_t nSubsets)
    {
        std::vector<std::vector<index_t>> subsetIndices(static_cast<std::size_t>(nSubsets));

        // determine the mapping of indices to subsets
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto subset = i % static_cast<std::size_t>(nSubsets);
            subsetIndices[subset].template emplace_back(indices[i]);
        }

        return subsetIndices;
    }

    template <typename DetectorDescriptor_t, typename data_t>
    std::vector<std::vector<index_t>>
        SubsetSampler<DetectorDescriptor_t, data_t>::splitRotationalClustering(
            const DetectorDescriptor_t& detectorDescriptor, index_t nSubsets)
    {

        const auto numCoeffsPerDim = detectorDescriptor.getNumberOfCoefficientsPerDimension();
        const index_t nDimensions = detectorDescriptor.getNumberOfDimensions();
        const auto numElements = numCoeffsPerDim[nDimensions - 1];
        std::vector<index_t> indices(static_cast<std::size_t>(numElements));
        std::iota(indices.begin(), indices.end(), 0);
        const auto geometry = detectorDescriptor.getGeometry();

        // angle between two rotation matrices used as a distance measure
        auto dist = [nDimensions](auto& g1, auto& g2) {
            const auto& r1 = g1.getRotationMatrix();
            const auto& r2 = g2.getRotationMatrix();
            auto product = r1 * r2.transpose();
            if (nDimensions == 2) { // the 2D case
                return static_cast<double>(std::atan2(product(1, 0), product(0, 0)));
            } else { // the 3D case
                return std::acos((product.trace() - 1.0) / 2.0);
            }
        };

        const auto first = geometry.front();
        std::sort(std::begin(indices), std::end(indices),
                  [dist, first, &geometry](auto lhs, auto rhs) {
                      return dist(first, geometry[static_cast<std::size_t>(lhs)])
                             < dist(first, geometry[static_cast<std::size_t>(rhs)]);
                  });

        return splitRoundRobin(indices, nSubsets);
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