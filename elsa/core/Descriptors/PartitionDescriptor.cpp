#include "PartitionDescriptor.h"
#include "Error.h"

#include <unordered_map>
#include <type_traits>

namespace elsa
{
    PartitionDescriptor::PartitionDescriptor(const DataDescriptor& dataDescriptor,
                                             index_t numberOfBlocks)
        : BlockDescriptor{dataDescriptor},
          _indexMap(numberOfBlocks),
          _blockDescriptors(0),
          _blockOffsets(numberOfBlocks)
    {
        if (numberOfBlocks < 2)
            throw InvalidArgumentError(
                "PartitionDescriptor: number of blocks must be greater than one");

        index_t lastDimSize = _numberOfCoefficientsPerDimension[_numberOfDimensions - 1];

        if (numberOfBlocks > lastDimSize)
            throw InvalidArgumentError(
                "PartitionDescriptor: number of blocks too large for given descriptor");

        index_t rest = lastDimSize % numberOfBlocks;

        auto blockDesc = generateDescriptorOfPartition(lastDimSize / numberOfBlocks);
        _blockDescriptors.push_back(std::move(blockDesc));
        _indexMap.head(numberOfBlocks - rest).setZero();
        for (index_t i = 0; i < numberOfBlocks && i <= numberOfBlocks - rest; i++)
            _blockOffsets[i] = i * _blockDescriptors[0]->getNumberOfCoefficients();

        if (rest > 0) {
            blockDesc = generateDescriptorOfPartition(lastDimSize / numberOfBlocks + 1);
            _blockDescriptors.push_back(std::move(blockDesc));
            _indexMap.tail(rest).array().setConstant(1);
            auto numCoeffs = _blockDescriptors[1]->getNumberOfCoefficients();
            for (index_t i = numberOfBlocks - rest + 1; i < numberOfBlocks; i++)
                _blockOffsets[i] = _blockOffsets[i - 1] + numCoeffs;
        }
    }

    PartitionDescriptor::PartitionDescriptor(const DataDescriptor& dataDescriptor,
                                             IndexVector_t slicesInBlock)
        : BlockDescriptor{dataDescriptor},
          _indexMap(slicesInBlock.size()),
          _blockDescriptors(0),
          _blockOffsets(slicesInBlock.size())
    {
        if (slicesInBlock.size() < 2)
            throw InvalidArgumentError(
                "PartitionDescriptor: number of blocks must be greater than one");

        if ((slicesInBlock.array() <= 0).any())
            throw InvalidArgumentError(
                "PartitionDescriptor: non-positive number of coefficients not allowed");

        if (slicesInBlock.sum() != _numberOfCoefficientsPerDimension[_numberOfDimensions - 1])
            throw InvalidArgumentError("PartitionDescriptor: cumulative size of partitioned "
                                       "descriptor does not match size of original descriptor");

        std::unordered_map<index_t, index_t> sizeToIndex;
        _blockOffsets[0] = 0;
        for (index_t i = 0; i < getNumberOfBlocks(); i++) {
            auto it = sizeToIndex.find(slicesInBlock[i]);
            index_t numCoeffs;

            if (it != sizeToIndex.end()) {
                _indexMap[i] = it->second;
                auto index = std::make_unsigned_t<index_t>(it->second);
                numCoeffs = _blockDescriptors[index]->getNumberOfCoefficients();
            } else {
                sizeToIndex.insert({slicesInBlock[i], _blockDescriptors.size()});
                _indexMap[i] = std::make_signed_t<long>(_blockDescriptors.size());
                _blockDescriptors.push_back(generateDescriptorOfPartition(slicesInBlock[i]));
                numCoeffs = _blockDescriptors.back()->getNumberOfCoefficients();
            }

            if (i != getNumberOfBlocks() - 1)
                _blockOffsets[i + 1] = _blockOffsets[i] + numCoeffs;
        }
    }

    PartitionDescriptor::PartitionDescriptor(const PartitionDescriptor& other)
        : BlockDescriptor(other), _indexMap(other._indexMap), _blockOffsets{other._blockOffsets}
    {
        for (const auto& blockDesc : other._blockDescriptors)
            _blockDescriptors.push_back(blockDesc->clone());
    }

    index_t PartitionDescriptor::getNumberOfBlocks() const { return _indexMap.size(); }

    const DataDescriptor& PartitionDescriptor::getDescriptorOfBlock(index_t i) const
    {
        if (i < 0 || i >= getNumberOfBlocks())
            throw InvalidArgumentError("BlockDescriptor: index i is out of bounds");

        auto index = std::make_unsigned_t<long>(_indexMap[i]);
        return *_blockDescriptors[index];
    }

    index_t PartitionDescriptor::getOffsetOfBlock(index_t i) const
    {
        if (i < 0 || i >= getNumberOfBlocks())
            throw InvalidArgumentError("BlockDescriptor: index i is out of bounds");

        return _blockOffsets[i];
    }

    PartitionDescriptor* PartitionDescriptor::cloneImpl() const
    {
        return new PartitionDescriptor(*this);
    }

    bool PartitionDescriptor::isEqual(const DataDescriptor& other) const
    {
        if (!BlockDescriptor::isEqual(other))
            return false;

        // static cast as type checked in base comparison
        auto otherBlock = static_cast<const PartitionDescriptor*>(&other);

        return _blockOffsets == otherBlock->_blockOffsets;
    }

    std::unique_ptr<VolumeDescriptor>
        PartitionDescriptor::generateDescriptorOfPartition(index_t numberOfSlices) const
    {
        auto coeffsPerDim = getNumberOfCoefficientsPerDimension();
        coeffsPerDim[_numberOfDimensions - 1] = numberOfSlices;
        return std::make_unique<VolumeDescriptor>(coeffsPerDim, getSpacingPerDimension());
    }
} // namespace elsa
