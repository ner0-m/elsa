#include "IdenticalBlocksDescriptor.h"
#include "VolumeDescriptor.h"
#include "Error.h"

namespace elsa
{
    IdenticalBlocksDescriptor::IdenticalBlocksDescriptor(index_t numberOfBlocks,
                                                         const DataDescriptor& dataDescriptor)
        : BlockDescriptor{initBase(numberOfBlocks, dataDescriptor)},
          _blockDescriptor{dataDescriptor.clone()},
          _numberOfBlocks{numberOfBlocks}

    {
    }

    index_t IdenticalBlocksDescriptor::getNumberOfBlocks() const
    {
        return _numberOfBlocks;
    }

    const DataDescriptor& IdenticalBlocksDescriptor::getDescriptorOfBlock(index_t i) const
    {
        if (i < 0 || i >= _numberOfBlocks)
            throw InvalidArgumentError("BlockDescriptor: index i is out of bounds");

        return *_blockDescriptor;
    }

    index_t IdenticalBlocksDescriptor::getOffsetOfBlock(index_t i) const
    {
        if (i < 0 || i >= _numberOfBlocks)
            throw InvalidArgumentError("BlockDescriptor: index i is out of bounds");

        return i * _blockDescriptor->getNumberOfCoefficients();
    }

    IdenticalBlocksDescriptor* IdenticalBlocksDescriptor::cloneImpl() const
    {
        return new IdenticalBlocksDescriptor(_numberOfBlocks, *_blockDescriptor);
    }

    bool IdenticalBlocksDescriptor::isEqual(const DataDescriptor& other) const
    {
        if (!BlockDescriptor::isEqual(other))
            return false;

        // static cast as type checked in base comparison
        auto otherBlock = static_cast<const IdenticalBlocksDescriptor*>(&other);

        if (*_blockDescriptor != *otherBlock->_blockDescriptor)
            return false;

        return true;
    }

    VolumeDescriptor IdenticalBlocksDescriptor::initBase(index_t numberOfBlocks,
                                                         const DataDescriptor& dataDescriptor)
    {
        if (numberOfBlocks < 1)
            throw InvalidArgumentError(
                "IdenticalBlockDescriptor: number of blocks has to be positive");

        auto numberOfCoeffs = dataDescriptor.getNumberOfCoefficientsPerDimension();
        index_t numDim = numberOfCoeffs.size() + 1;
        auto spacingOfCoeffs = dataDescriptor.getSpacingPerDimension();
        numberOfCoeffs.conservativeResize(numDim);
        spacingOfCoeffs.conservativeResize(numDim);
        numberOfCoeffs[numDim - 1] = numberOfBlocks;
        spacingOfCoeffs[numDim - 1] = 1;

        return VolumeDescriptor(numberOfCoeffs, spacingOfCoeffs);
    }
} // namespace elsa
