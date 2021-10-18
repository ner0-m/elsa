#include "TrajectoryGenerator.h"
#include "VolumeDescriptor.h"

namespace elsa
{

    std::pair<IndexVector_t, RealVector_t>
        TrajectoryGenerator::calculateSizeAndSpacingPerGeometry(const DataDescriptor& volDescr,
                                                                index_t numberOfPoses)
    {
        const auto dim = volDescr.getNumberOfDimensions();

        IndexVector_t coeffs(dim);
        RealVector_t spacing(dim);

        // Scale coeffsPerDim by sqrt(2), this reduces undersampling of the corners, as the
        // detector is larger than the volume. Cast back and forthe to reduce warnings...
        // This has to be a RealVector_t, most likely that the cast happens, anyway we get
        // errors down the line see #86 in Gitlab
        const RealVector_t coeffsPerDim =
            volDescr.getNumberOfCoefficientsPerDimension().template cast<real_t>();
        const real_t sqrt2 = std::sqrt(2.f);
        const auto coeffsPerDimScaled = (coeffsPerDim * sqrt2).template cast<index_t>();

        const auto spacingPerDim = volDescr.getSpacingPerDimension();

        coeffs.head(dim - 1) = coeffsPerDimScaled.head(dim - 1);
        coeffs[dim - 1] = numberOfPoses; // TODO: with eigen 3.4: `coeffs(Eigen::last) = 1`

        spacing.head(dim - 1) = spacingPerDim.head(dim - 1);
        spacing[dim - 1] = 1; // TODO: same as coeffs

        // return a pair, then split it using structured bindings
        return std::pair{coeffs, spacing};
    }
} // namespace elsa
