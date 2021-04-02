#pragma once

#include "BlockDescriptor.h"

#include <vector>

#include "spdlog/fmt/fmt.h"

namespace elsa
{
    /**
     * @brief Class representing a block descriptor whose different blocks may have completely
     * different descriptors.
     *
     * @author Matthias Wieczorek - initial code
     * @author David Frank - rewrite
     * @author Nikola Dinev - various enhancements
     * @author Tobias Lasser - rewrite, modularization, modernization
     *
     * There are no restrictions whatsoever imposed on the descriptors of different blocks.
     * Different blocks may even have different number of dimensions.
     *
     * The full descriptor will always be one-dimensional, and with a spacing of one. The size of it
     * will be the sum of the sizes of all the descriptors, i.e. the sizes returned by
     * DataDescriptor::getNumberOfCoefficients() for each descriptor in the list.
     */
    class RandomBlocksDescriptor : public BlockDescriptor
    {
    public:
        /**
         * @brief Construct a RandomBlocksDescriptor from a list of descriptors
         *
         * @param[in] blockDescriptors the list of descriptors of each block
         *
         * @throw InvalidArgumentError if the list is empty
         */
        RandomBlocksDescriptor(
            const std::vector<std::unique_ptr<DataDescriptor>>& blockDescriptors);

        /**
         * @brief Construct a RandomBlocksDescriptor from a list of descriptors
         *
         * @param[in] blockDescriptors the list of descriptors of each block
         *
         * @throw InvalidArgumentError if the list is empty
         */
        RandomBlocksDescriptor(std::vector<std::unique_ptr<DataDescriptor>>&& blockDescriptors);

        /// make copy constructor deletion explicit
        RandomBlocksDescriptor(const RandomBlocksDescriptor&) = delete;

        /// default desctructor
        ~RandomBlocksDescriptor() override = default;

        /// return the number of blocks
        index_t getNumberOfBlocks() const override;

        /// return the DataDescriptor of the i-th block
        const DataDescriptor& getDescriptorOfBlock(index_t i) const override;

        /// return the offset to access the data of the i-th block
        index_t getOffsetOfBlock(index_t i) const override;

    protected:
        /// vector of DataDescriptors describing the individual blocks
        std::vector<std::unique_ptr<DataDescriptor>> _blockDescriptors;

        /// vector of the individual block data offsets
        IndexVector_t _blockOffsets;

        /// implement the polymorphic clone operation
        RandomBlocksDescriptor* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataDescriptor& other) const override;

    private:
        /// return the total number of coefficients of the descriptors in the list as an IndexVector
        IndexVector_t
            determineSize(const std::vector<std::unique_ptr<DataDescriptor>>& blockDescriptors);
    };
} // namespace elsa

std::ostream& operator<<(std::ostream& os, const elsa::RandomBlocksDescriptor& desc);

template <>
struct fmt::formatter<elsa::RandomBlocksDescriptor> : fmt::formatter<std::string> {
    auto format(const elsa::RandomBlocksDescriptor& dd, fmt::format_context& ctx)
        -> decltype(ctx.out())
    {
        std::ostringstream os;
        os << dd;
        return formatter<std::string>::format(os.str(), ctx);
    }
};
