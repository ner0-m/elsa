#include "SubsetSampler.h"
#include "PartitionDescriptor.h"
#include "PlanarDetectorDescriptor.h"
#include "SiddonsMethod.h"
#include "JosephsMethod.h"

namespace elsa
{
    template <typename detectorDescriptor_t, typename data_t>
    SubsetSampler<detectorDescriptor_t, data_t>::SubsetSampler(
        const VolumeDescriptor& volumeDescriptor, const detectorDescriptor_t& detectorDescriptor,
        const DataContainer<data_t>& sinogram, index_t nSubsets)
        : _volumeDescriptor(volumeDescriptor),
          _fullDetectorDescriptor(detectorDescriptor),
          _nSubsets{nSubsets}
    {

        // the individual data descriptors for each block
        std::vector<IndexVector_t> subsetIndices;

        index_t nDimensions = sinogram.getDataDescriptor().getNumberOfDimensions();
        // determine the mapping of indices to subsets
        auto subsetSize = static_cast<index_t>(
            sinogram.getDataDescriptor().getNumberOfCoefficientsPerDimension()[nDimensions - 1]
            / _nSubsets);
        for (index_t i = 0; i < _nSubsets - 1; i++) {
            IndexVector_t indices(subsetSize);
            for (index_t j = 0; j < subsetSize; j++) {
                indices[j] = i + j * _nSubsets;
            }
            subsetIndices.emplace_back(indices);
        }
        index_t lastSubsetSize =
            sinogram.getDataDescriptor().getNumberOfCoefficientsPerDimension()[nDimensions - 1]
            - subsetSize * (_nSubsets - 1);
        IndexVector_t lastSubsetIndices(lastSubsetSize);
        for (index_t j = 0; j < subsetSize; j++) {
            lastSubsetIndices[j] = _nSubsets - 1 + j * _nSubsets;
        }

        // TODO: this is not quite right, better would be for the first subset to be larger
        for (index_t j = subsetSize; j < lastSubsetSize; j++) {
            lastSubsetIndices[j] =
                sinogram.getDataDescriptor().getNumberOfCoefficientsPerDimension()[nDimensions - 1]
                - (lastSubsetSize - j);
        }
        subsetIndices.emplace_back(lastSubsetIndices);

        // save the number of entries per subset
        IndexVector_t slicesInBlock(subsetIndices.size());
        for (unsigned long i = 0; i < subsetIndices.size(); i++) {
            slicesInBlock[static_cast<index_t>(i)] = subsetIndices[i].size();
        }

        for (index_t i = 0; i < _nSubsets; i++) {
            IndexVector_t indices = subsetIndices[static_cast<unsigned long>(
                i)]; // the indices of the data rows belonging to this subset
            std::vector<Geometry> geometry;
            for (auto index : indices) {
                auto geo = detectorDescriptor.getGeometryAt(index);
                if (geo.has_value()) {
                    geometry.emplace_back(*geo);
                }
            }
            IndexVector_t numOfCoeffsPerDim =
                detectorDescriptor.getNumberOfCoefficientsPerDimension();
            numOfCoeffsPerDim[numOfCoeffsPerDim.size() - 1] = indices.size();

            // TODO: maybe move this logic to the detector descriptor class (but how?)

            _detectorDescriptors.emplace_back(detectorDescriptor_t(numOfCoeffsPerDim, geometry));
        }

        PartitionDescriptor dataDescriptor(sinogram.getDataDescriptor(), slicesInBlock);
        // TODO: make this initialization better to not initialize _data twice (if neccessary)
        _data =
            std::make_unique<DataContainer<data_t>>(dataDescriptor, sinogram.getDataHandlerType());

        // resort the actual measurement data
        IndexVector_t numOfCoeffsPerDim =
            sinogram.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        index_t coeffsPerRow = numOfCoeffsPerDim.head(numOfCoeffsPerDim.size() - 1).prod();
        for (index_t i = 0; i < _nSubsets; i++) {
            // the indices of the data rows belonging to this subset
            IndexVector_t indices = subsetIndices[static_cast<unsigned long>(i)];

            auto block = (*_data).getBlock(i);

            for (int j = 0; j < indices.size(); j++) {
                for (int k = 0; k < coeffsPerRow; k++) {
                    block[j * coeffsPerRow + k] = sinogram[indices[j] * coeffsPerRow + k];
                }
            }
        }
    }

    template <typename detectorDescriptor_t, typename data_t>
    SubsetSampler<detectorDescriptor_t, data_t>::SubsetSampler(
        const SubsetSampler<detectorDescriptor_t, data_t>& other)
        : _volumeDescriptor(other._volumeDescriptor),
          _fullDetectorDescriptor(other._fullDetectorDescriptor),
          _nSubsets{other._nSubsets}
    {
        _data = std::make_unique<DataContainer<data_t>>(*other._data);
        for (const auto& detectorDescriptor : other._detectorDescriptors) {
            _detectorDescriptors.emplace_back(detectorDescriptor);
        }
    }

    template <typename detectorDescriptor_t, typename data_t>
    DataContainer<data_t> SubsetSampler<detectorDescriptor_t, data_t>::getData()
    {
        return (*_data);
    }

    template <typename detectorDescriptor_t, typename data_t>
    SubsetSampler<detectorDescriptor_t, data_t>*
        SubsetSampler<detectorDescriptor_t, data_t>::cloneImpl() const
    {
        return new SubsetSampler<detectorDescriptor_t, data_t>(*this);
    }

    template <typename detectorDescriptor_t, typename data_t>
    bool SubsetSampler<detectorDescriptor_t, data_t>::isEqual(
        const SubsetSampler<detectorDescriptor_t, data_t>& other) const
    {
        if (typeid(*this) != typeid(other))
            return false;

        if (*_data != *(other._data))
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