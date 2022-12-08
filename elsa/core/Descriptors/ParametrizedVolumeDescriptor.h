#pragma once
#include "IdenticalBlocksDescriptor.h"

namespace elsa
{
/**
 * \brief Represents a volume of functions or vectors under a certain basis
 *
 * \author Nikola Dinev (nikola.dinev@tum.de)
 */
class ParametrizedVolumeDescriptor : public IdenticalBlocksDescriptor
{
//private:
//    const IndexVector_t determineCoefficientsPerDimension(
//        const DataDescriptor& volumeDescriptor, const DataDescriptor& basisDescriptor) const;
//
//    const RealVector_t determineSpacingPerDimension(
//        const DataDescriptor& volumeDescriptor, const DataDescriptor& basisDescriptor) const;

public:
    /**
     * \brief Construct a new ParametrizedVolumeDescriptor object
     *
     * For a basis of dimension \$N\$, concatenates \$N\$ volume descriptors in a BlockDescriptor.
     * The \$i\$-th descriptor holds the volume parameters w.r.t. the \$i\$-th basis function/vector.
     *
     * \param volumeDesciptor descriptor for the volume
     * \param basisDescriptor a descriptor for the basis functions/vectors
     */
    ParametrizedVolumeDescriptor(const DataDescriptor& volumeDescriptor, const DataDescriptor& basisDescriptor);

    virtual ParametrizedVolumeDescriptor* cloneImpl() const override;
    const DataDescriptor& getBasisDescriptor() const;

protected:
    std::unique_ptr<DataDescriptor> _basisDescriptor;
};
} // namespace elsa