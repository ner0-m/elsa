#pragma once

#include "DataDescriptor.h"

namespace elsa
{
    /**
     * @brief Finds the descriptor with the same number of coefficients as the descriptors in
     * the list that retains as much information as possible
     *
     * @param[in] descriptorList a vector of plain pointers to DataDescriptor
     *
     * @return std::unique_ptr<DataDescriptor> the best common descriptor
     *
     * @throw InvalidArgumentError if the vector is empty or the descriptors in the vector
     * don't all have the same size
     *
     * If all descriptors are equal, a clone of the first descriptor in the list is returned.
     * If all descriptors have a common base descriptor, that data descriptor is returned.
     * If the base descriptors only differ in spacing, the base descriptor with a uniform
     * spacing of 1 is returned.
     * Otherwise, the linearized descriptor with a spacing of 1 is returned.
     */
    std::unique_ptr<DataDescriptor>
        bestCommon(const std::vector<const DataDescriptor*>& descriptorList);

    /// convenience overload for invoking bestCommon() with a number of const DataDescriptor&
    template <
        typename... DescriptorType,
        typename = std::enable_if_t<(std::is_base_of_v<DataDescriptor, DescriptorType> && ...)>>
    std::unique_ptr<DataDescriptor> bestCommon(const DescriptorType&... descriptors)
    {
        return bestCommon(std::vector{static_cast<const DataDescriptor*>(&descriptors)...});
    }
} // namespace elsa