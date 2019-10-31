#include "BlockDescriptor.h"

#include <stdexcept>

namespace elsa
{

    BlockDescriptor::BlockDescriptor(index_t numberOfBlocks, const DataDescriptor& dataDescriptor)
        : DataDescriptor(dataDescriptor), _blockDescriptors{}, _blockOffsets(numberOfBlocks)
    {
        // sanity check
        if (numberOfBlocks <= 0)
            throw std::invalid_argument("BlockDescriptor: number of blocks has to be positive");

        for (index_t i = 0; i < numberOfBlocks; ++i) {
            _blockDescriptors.emplace_back(dataDescriptor.clone());
            _blockOffsets(i) = dataDescriptor.getNumberOfCoefficients() * i;
        }

        // update the base class DataDescriptor with additional dimension/size
        _numberOfDimensions++;

        _numberOfCoefficientsPerDimension.conservativeResize(_numberOfDimensions);
        _numberOfCoefficientsPerDimension(_numberOfDimensions - 1) = numberOfBlocks;

        _spacingPerDimension.conservativeResize(_numberOfDimensions);
        _spacingPerDimension(_numberOfDimensions - 1) = 1.0;

        _productOfCoefficientsPerDimension.conservativeResize(_numberOfDimensions);
        _productOfCoefficientsPerDimension(_numberOfDimensions - 1) =
            _numberOfCoefficientsPerDimension.head(_numberOfDimensions - 1).prod();
    }

    index_t BlockDescriptor::getNumberOfBlocks() const { return _blockDescriptors.size(); }

    const DataDescriptor& BlockDescriptor::getIthDescriptor(index_t i) const
    {
        return *_blockDescriptors.at(i);
    }

    index_t BlockDescriptor::getIthBlockOffset(elsa::index_t i) const
    {
        if (i < 0 || i >= _blockOffsets.size())
            throw std::invalid_argument("BlockDescriptor: index i is out of bounds");

        return _blockOffsets.coeff(i);
    }

    BlockDescriptor::BlockDescriptor(const BlockDescriptor& blockDescriptor)
        : DataDescriptor(blockDescriptor),
          _blockDescriptors{},
          _blockOffsets{blockDescriptor._blockOffsets}
    {
        for (const auto& descriptor : blockDescriptor._blockDescriptors)
            _blockDescriptors.emplace_back(descriptor->clone());
    }

    BlockDescriptor* BlockDescriptor::cloneImpl() const { return new BlockDescriptor(*this); }

    bool BlockDescriptor::isEqual(const DataDescriptor& other) const
    {
        if (!DataDescriptor::isEqual(other))
            return false;

        auto otherBlock = dynamic_cast<const BlockDescriptor*>(&other);
        if (!otherBlock)
            return false;

        if (_blockDescriptors.size() != otherBlock->_blockDescriptors.size())
            return false;

        for (index_t i = 0; i < _blockDescriptors.size(); ++i)
            if (*_blockDescriptors.at(i) != *otherBlock->_blockDescriptors.at(i))
                return false;

        if (_blockOffsets != otherBlock->_blockOffsets)
            return false;

        return true;
    }

} // namespace elsa
