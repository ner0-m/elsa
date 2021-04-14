#pragma once

#include "BlockDescriptor.h"
#include "VolumeDescriptor.h"

namespace elsa
{
    /**
     * \brief Class representing a series of identical descriptors concatenated along a new
     * dimension (the last dimension of the full descriptor).
     *
     * \author Nikola Dinev
     *
     * The blocks are, essentially, slices (though not necessarily two-dimensional) of the full
     * descriptor along its last dimension. The last dimension of the full descriptor serves solely
     * for the indexing of the different blocks, and will always have a spacing of one and a number
     * of coefficients corresponding to the number of blocks.
     *
     * This descriptor should be the preferred choice when dealing with vector fields.
     */
    class IdenticalBlocksDescriptor : public BlockDescriptor
    {
    public:
        /**
         *  \brief Create a new descriptor, replicating the dataDescriptor numberOfBlocks times
         * along a new dimension
         *
         *  \param[in] numberOfBlocks is the desired number of blocks
         *  \param[in] dataDescriptor is the descriptor that will be replicated numberOfBlocks
         *  times
         * along a new dimension
         *
         *  \throw InvalidArgumentError if numberOfBlocks is non-positive
         */
        IdenticalBlocksDescriptor(index_t numberOfBlocks, const DataDescriptor& dataDescriptor);

        /// make copy constructor deletion explicit
        IdenticalBlocksDescriptor(const IdenticalBlocksDescriptor&) = delete;

        /// default destructor
        ~IdenticalBlocksDescriptor() override = default;

        /// return the number of blocks
        index_t getNumberOfBlocks() const override;

        /// return the DataDescriptor of the i-th block
        const DataDescriptor& getDescriptorOfBlock(index_t i) const override;

        /// return the offset to access the data of the i-th block
        index_t getOffsetOfBlock(index_t i) const override;

    protected:
        /// descriptor of a single block
        std::unique_ptr<DataDescriptor> _blockDescriptor;

        /// the total number of identical blocks
        index_t _numberOfBlocks;

        /// implement the polymorphic clone operation
        IdenticalBlocksDescriptor* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataDescriptor& other) const override;

    private:
        /// generates the
        VolumeDescriptor initBase(index_t numberOfBlocks, const DataDescriptor& dataDescriptor);
    };
} // namespace elsa
