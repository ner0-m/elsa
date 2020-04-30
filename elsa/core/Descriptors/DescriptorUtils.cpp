#include "DescriptorUtils.h"
#include "DataDescriptor.h"
#include "VolumeDescriptor.h"

namespace elsa
{
    std::unique_ptr<DataDescriptor> bestCommon(const std::vector<const DataDescriptor*>& descList)
    {
        if (descList.empty())
            throw std::invalid_argument("DataDescriptor::bestCommon: descriptor list empty");

        const auto& firstDesc = *descList[0];
        auto coeffs = firstDesc.getNumberOfCoefficientsPerDimension();
        auto size = firstDesc.getNumberOfCoefficients();
        auto spacing = firstDesc.getSpacingPerDimension();

        bool allSame =
            std::all_of(descList.begin(), descList.end(),
                        [&firstDesc](const DataDescriptor* d) { return *d == firstDesc; });
        if (allSame)
            return firstDesc.clone();

        bool allSameCoeffs =
            std::all_of(descList.begin(), descList.end(), [&coeffs](const DataDescriptor* d) {
                return d->getNumberOfCoefficientsPerDimension().size() == coeffs.size()
                       && d->getNumberOfCoefficientsPerDimension() == coeffs;
            });

        if (allSameCoeffs) {
            bool allSameSpacing =
                std::all_of(descList.begin(), descList.end(), [&spacing](const DataDescriptor* d) {
                    return d->getSpacingPerDimension() == spacing;
                });
            if (allSameSpacing) {
                return std::make_unique<VolumeDescriptor>(coeffs, spacing);
            } else {
                return std::make_unique<VolumeDescriptor>(coeffs);
            }
        }

        bool allSameSize =
            std::all_of(descList.begin(), descList.end(), [size](const DataDescriptor* d) {
                return d->getNumberOfCoefficients() == size;
            });

        if (!allSameSize)
            throw std::invalid_argument(
                "DataDescriptor::bestCommon: descriptor sizes do not match");

        return std::make_unique<VolumeDescriptor>(IndexVector_t::Constant(1, size));
    }
} // namespace elsa