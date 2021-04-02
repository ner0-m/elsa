#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"
#include "DataDescriptor.h"

#include "spdlog/fmt/fmt.h"

namespace elsa
{

    /**
     *  @brief Abstract class defining the interface of all block descriptors.
     *
     *  @author Matthias Wieczorek - initial code
     *  @author David Frank - rewrite
     *  @author Tobias Lasser - rewrite, modularization, modernization
     *  @author Nikola Dinev - rework into abstract class
     *
     * A block descriptor provides metadata about a signal that is stored in memory (typically a
     * DataContainer). This signal can be n-dimensional, and will be stored in memory in a
     * linearized fashion in blocks. The blocks can be used to support various operations (like
     * blocked operators or ordered subsets), however, the blocks have to lie in memory one after
     * the other (i.e. no stride is supported).
     */
    class BlockDescriptor : public DataDescriptor
    {
    public:
        /// default destructor
        ~BlockDescriptor() override = default;

        /// return the number of blocks
        virtual index_t getNumberOfBlocks() const = 0;

        /// return the DataDescriptor of the i-th block
        virtual const DataDescriptor& getDescriptorOfBlock(index_t i) const = 0;

        /// return the offset to access the data of the i-th block
        virtual index_t getOffsetOfBlock(index_t i) const = 0;

    protected:
        /// used by derived classes to initialize the DataDescriptor base
        BlockDescriptor(DataDescriptor&& base) : DataDescriptor{std::move(base)} {}

        /// used by derived classes to initialize the DataDescriptor base
        BlockDescriptor(const DataDescriptor& base) : DataDescriptor{base} {}
    };
} // namespace elsa

std::ostream& operator<<(std::ostream& os, const elsa::BlockDescriptor& desc);

template <>
struct fmt::formatter<elsa::BlockDescriptor> : fmt::formatter<std::string> {
    auto format(const elsa::BlockDescriptor& dd, fmt::format_context& ctx) -> decltype(ctx.out())
    {
        std::ostringstream os;
        os << dd;
        return formatter<std::string>::format(os.str(), ctx);
    }
};
