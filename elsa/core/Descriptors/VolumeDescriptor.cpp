#include "VolumeDescriptor.h"

#include <algorithm>

namespace elsa
{
    VolumeDescriptor::VolumeDescriptor(IndexVector_t numberOfCoefficientsPerDimension)
        : DataDescriptor(numberOfCoefficientsPerDimension)
    {
    }

    VolumeDescriptor::VolumeDescriptor(IndexVector_t numberOfCoefficientsPerDimension,
                                       RealVector_t spacingPerDimension)
        : DataDescriptor(numberOfCoefficientsPerDimension, spacingPerDimension)
    {
    }

    VolumeDescriptor::VolumeDescriptor(
        std::initializer_list<index_t> numberOfCoefficientsPerDimension)
        : DataDescriptor(IndexVector_t{numberOfCoefficientsPerDimension})
    {
    }

    VolumeDescriptor::VolumeDescriptor(
        std::initializer_list<index_t> numberOfCoefficientsPerDimension,
        std::initializer_list<real_t> spacingPerDimension)
        : DataDescriptor(IndexVector_t{numberOfCoefficientsPerDimension},
                         RealVector_t{spacingPerDimension})
    {
    }

    VolumeDescriptor* VolumeDescriptor::cloneImpl() const
    {
        return new VolumeDescriptor(_numberOfCoefficientsPerDimension, _spacingPerDimension);
    }

    bool VolumeDescriptor::isEqual(const DataDescriptor& other) const
    {
        return DataDescriptor::isEqual(other);
    }

} // namespace elsa
