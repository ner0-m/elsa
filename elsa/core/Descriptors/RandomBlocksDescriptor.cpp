#include "RandomBlocksDescriptor.h"
#include "VolumeDescriptor.h"

namespace elsa
{
    RandomBlocksDescriptor::RandomBlocksDescriptor(
        const std::vector<std::unique_ptr<DataDescriptor>>& blockDescriptors)
        : BlockDescriptor{VolumeDescriptor{determineSize(blockDescriptors)}},
          _blockDescriptors(0),
          _blockOffsets{blockDescriptors.size()}
    {
        index_t offset = 0;

        for (std::size_t i = 0; i < blockDescriptors.size(); i++) {
            _blockOffsets[static_cast<index_t>(i)] = offset;
            _blockDescriptors.emplace_back(blockDescriptors[i]->clone());
            offset += blockDescriptors[i]->getNumberOfCoefficients();
        }
    }

    RandomBlocksDescriptor::RandomBlocksDescriptor(
        std::vector<std::unique_ptr<DataDescriptor>>&& blockDescriptors)
        : BlockDescriptor{VolumeDescriptor{determineSize(blockDescriptors)}},
          _blockDescriptors{std::move(blockDescriptors)},
          _blockOffsets{_blockDescriptors.size()}
    {
        index_t offset = 0;

        for (std::size_t i = 0; i < _blockDescriptors.size(); i++) {
            _blockOffsets[static_cast<index_t>(i)] = offset;
            offset += _blockDescriptors[i]->getNumberOfCoefficients();
        }
    }

    index_t RandomBlocksDescriptor::getNumberOfBlocks() const
    {
        return static_cast<index_t>(_blockDescriptors.size());
    }

    const DataDescriptor& RandomBlocksDescriptor::getDescriptorOfBlock(index_t i) const
    {
        if (i < 0 || i >= _blockOffsets.size())
            throw std::invalid_argument("BlockDescriptor: index i is out of bounds");

        return *_blockDescriptors[static_cast<std::size_t>(i)];
    }

    index_t RandomBlocksDescriptor::getOffsetOfBlock(index_t i) const
    {
        if (i < 0 || i >= _blockOffsets.size())
            throw std::invalid_argument("BlockDescriptor: index i is out of bounds");

        return _blockOffsets[i];
    }

    RandomBlocksDescriptor* RandomBlocksDescriptor::cloneImpl() const
    {
        return new RandomBlocksDescriptor(_blockDescriptors);
    }

    bool RandomBlocksDescriptor::isEqual(const DataDescriptor& other) const
    {
        if (!BlockDescriptor::isEqual(other))
            return false;

        // static_cast as type checked in base comparison
        auto otherBlock = static_cast<const RandomBlocksDescriptor*>(&other);

        if (_blockDescriptors.size() != otherBlock->_blockDescriptors.size())
            return false;

        for (std::size_t i = 0; i < _blockDescriptors.size(); ++i)
            if (*_blockDescriptors[i] != *otherBlock->_blockDescriptors[i])
                return false;

        return true;
    }

    IndexVector_t RandomBlocksDescriptor::determineSize(
        const std::vector<std::unique_ptr<DataDescriptor>>& blockDescriptors)
    {
        if (blockDescriptors.empty())
            throw std::invalid_argument(
                "RandomBlockDescriptor: list of block descriptors cannot be empty");

        index_t size = 0;

        for (const auto& desc : blockDescriptors)
            size += desc->getNumberOfCoefficients();

        return IndexVector_t::Constant(1, size);
    }

} // namespace elsa
