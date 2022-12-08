#include "ParametrizedVolumeDescriptor.h"

namespace elsa {
    ParametrizedVolumeDescriptor::ParametrizedVolumeDescriptor(
        const DataDescriptor& volumeDescriptor, const DataDescriptor& basisDesciptor):
          IdenticalBlocksDescriptor(basisDesciptor.getNumberOfCoefficients(), volumeDescriptor),
          _basisDescriptor(basisDesciptor.clone())
    {
        if (basisDesciptor.getNumberOfDimensions() != 1)
            throw std::invalid_argument("ParametrizedVolumeDescriptor: Basis should be exactly one-dimensional.");
        
    }


//    const IndexVector_t ParametrizedVolumeDescriptor::determineCoefficientsPerDimension(
//        const DataDescriptor& volumeDescriptor, const DataDescriptor& basisDesciptor) const
//    {
//        auto volumeDimensions = volumeDescriptor.getNumberOfCoefficientsPerDimension();
//        volumeDimensions.conservativeResize(volumeDescriptor.getNumDim()+1);
//        volumeDimensions[volumeDimensions.size()-1] = basisDesciptor.getNumberOfCoefficients();;
//        return volumeDimensions;
//    }
//
//    const RealVector_t ParametrizedVolumeDescriptor::determineSpacingPerDimension(
//        const DataDescriptor& volumeDescriptor, const DataDescriptor& basisDesciptor) const
//    {
//        auto spacingPerDim = volumeDescriptor.getSpacingOfCoefficientsPerDimension();
//        spacingPerDim.conservativeResize(volumeDescriptor.getNumDim()+1);
//        spacingPerDim[spacingPerDim.size()-1] = 1.0;
//        return spacingPerDim;
//    }

    ParametrizedVolumeDescriptor* ParametrizedVolumeDescriptor::cloneImpl() const {
        return new ParametrizedVolumeDescriptor(IdenticalBlocksDescriptor::getDescriptorOfBlock(0), *_basisDescriptor);
    }

    const DataDescriptor& ParametrizedVolumeDescriptor::getBasisDescriptor() const {
        return *_basisDescriptor;
    }
}