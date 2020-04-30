#include "VolumeDescriptor.h"

#include <stdexcept>
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

    VolumeDescriptor* VolumeDescriptor::cloneImpl() const
    {
        return new VolumeDescriptor(_numberOfCoefficientsPerDimension, _spacingPerDimension);
    }

    bool VolumeDescriptor::isEqual(const DataDescriptor& other) const
    {
        return DataDescriptor::isEqual(other);
    }

} // namespace elsa
