#include "BlockDescriptor.h"
namespace elsa
{
    /**
     * \brief Class representing a descriptor obtained after the partition of a normal data
     * descriptor into blocks.
     *
     * \author Nikola Dinev
     *
     * A single block of the PartitionDescriptor represents a linear segment containing one or more
     * slices of the original descriptor, taken along its last dimension. This also means that the
     * number of coefficients per block can only vary in the last dimension.
     *
     * The PartitionDescriptor has the same number of coefficients and spacing per dimension as the
     * original.
     */
    class PartitionDescriptor : public BlockDescriptor
    {
    public:
        /**
         * \brief Construct a PartitionDescriptor by partitioning a given descriptor into blocks of
         * fairly equal sizes
         *
         * \param[in] dataDescriptor the descriptor to be partitioned
         * \param[in] numberOfBlocks the number of blocks
         *
         * \throw std::invalid_argument if numberOfBlocks is less than 2 or greater than the number
         * of coefficients in the last dimension
         *
         * If the given descriptor has a size of \f$ N \f$ in its last dimension, when dividing it
         * into \f$ m \f$ blocks and \f$ N \f$ is not evenly divisible by \f$ m \f$, the last \f$ N
         * \bmod m \f$ blocks will have a size of the last dimension one bigger than that of the
         * others.
         *
         * Note: if the passed in DataDescriptor is a block descriptor, the block information
         * is ignored when generating the new PartitionDescriptor.
         */
        PartitionDescriptor(const DataDescriptor& dataDescriptor, index_t numberOfBlocks);

        /**
         * \brief Construct a PartitionDescriptor by specifying the number of slices contained in
         * each block
         *
         * \param[in] dataDescriptor the descriptor to be partitioned
         * \param[in] slicesInBlock the number of slices in each block
         *
         * \throw std::invalid_argument if slicesInBlock does not specify a valid partition scheme
         * for the given descriptor
         *
         * Note: if the passed in DataDescriptor is a block descriptor, the block information
         * is ignored when generating the new PartitionDescriptor.
         */
        PartitionDescriptor(const DataDescriptor& dataDescriptor, IndexVector_t slicesInBlock);

        /// default destructor
        ~PartitionDescriptor() override = default;

        /// return the number of blocks
        index_t getNumberOfBlocks() const override;

        /// return the DataDescriptor of the i-th block
        const DataDescriptor& getDescriptorOfBlock(index_t i) const override;

        /// return the offset to access the data of the i-th block
        index_t getOffsetOfBlock(index_t i) const override;

    protected:
        /// maps a block index to the index of the corresponding descriptor in _blockDescriptors
        IndexVector_t _indexMap;

        /// vector of unique DataDescriptors describing the individual blocks
        std::vector<std::unique_ptr<DataDescriptor>> _blockDescriptors;

        /// vector of the individual block data offsets
        IndexVector_t _blockOffsets;

        /// protected copy constructor; used for cloning
        PartitionDescriptor(const PartitionDescriptor& other);

        /// implement the polymorphic clone operation
        PartitionDescriptor* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataDescriptor& other) const override;

    private:
        /// generates the descriptor of a partition containing numberOfSlices slices
        std::unique_ptr<DataDescriptor> generateDescriptorOfPartition(index_t numberOfSlices) const;
    };
} // namespace elsa
