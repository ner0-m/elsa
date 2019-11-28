#pragma once

#include "elsa.h"
#include "Cloneable.h"
#include "DataDescriptor.h"

#include <memory>
#include <vector>

namespace elsa
{

    /**
     *  \brief Class representing metadata for blocked, linearized n-dimensional signal stored in
     * memory
     *
     *  \author Matthias Wieczorek - initial code
     *  \author David Frank - rewrite
     *  \author Nikola Dinev - various enhancements
     *  \author Tobias Lasser - rewrite, modularization, modernization
     *
     *  This class provides metadata about a signal that is stored in memory (typically a
     * DataContainer). This signal can be n-dimensional, and will be stored in memory in a
     * linearized fashion in blocks. The blocks can be used to support various operations (like
     * blocked operators or ordered subsets), however, the blocks have to lie in memory one after
     * the other (i.e. no stride is supported).
     */
    class BlockDescriptor : public DataDescriptor
    {
    public:
        /// delete default constructor (having no metadata is invalid)
        BlockDescriptor() = delete;

        /// default destructor
        ~BlockDescriptor() override = default;

        /**
         *  \brief Create a new descriptor, replicating the dataDescriptor numberOfBlocks times
         * along a new dimension
         *
         *  \param[in] numberOfBlocks is the desired number of blocks
         *  \param[in] dataDescriptor is the descriptor that will be replicated numberOfBlocks times
         * along a new dimension
         *
         *  \throw std::invalid_argument if numberOfBlocks is non-positive
         */
        explicit BlockDescriptor(index_t numberOfBlocks, const DataDescriptor& dataDescriptor);

        /// return the number of blocks
        index_t getNumberOfBlocks() const;

        /// return the DataDescriptor of the i-th block
        const DataDescriptor& getDescriptorOfBlock(index_t i) const;

        /// return the offset to access the data of the i-th block
        index_t getOffsetOfBlock(index_t i) const;

    protected:
        /// vector of DataDescriptors describing the individual blocks
        std::vector<std::unique_ptr<DataDescriptor>> _blockDescriptors;

        /// vector of the individual block data offsets
        IndexVector_t _blockOffsets;

        /// protected copy constructor
        BlockDescriptor(const BlockDescriptor& blockDescriptor);

        /// implement the polymorphic clone operation
        BlockDescriptor* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataDescriptor& other) const override;
    };
} // namespace elsa
