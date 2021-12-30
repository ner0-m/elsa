#include "Trajectories.h"

#include "VolumeDescriptor.h"
#include "CircleTrajectoryGenerator.h"
#include "PlanarDetectorDescriptor.h"
#include "StrongTypes.h"

namespace elsa::trajectories
{
    PlanarDetectorDescriptor circular(const DataDescriptor& volumeDescriptor, SampledArc arc,
                                      Distance distance)
    {
        const auto [segment, numAngles] = arc();
        const auto [ctr2det, src2ctr] = distance();

        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, volumeDescriptor, segment, src2ctr, ctr2det);

        // TODO: Does this create a correct copy?
        return downcast<PlanarDetectorDescriptor>(*sinoDescriptor);
    }

    PlanarDetectorDescriptor circular(const DataDescriptor& volumeDescriptor, SampledArc arc)
    {
        const auto shape = volumeDescriptor.getNumberOfCoefficientsPerDimension();
        return circular(volumeDescriptor, arc, Distance{static_cast<real_t>(shape(0))});
    }

    PlanarDetectorDescriptor fullCircle(const DataDescriptor& volDesc) { return circular(volDesc); }

    PlanarDetectorDescriptor halfCircle(const DataDescriptor& volDesc)
    {
        return circular(volDesc, SampledArc{geometry::Degree{180}});
    }

    PlanarDetectorDescriptor sparseCircle(const DataDescriptor& volDesc, index_t numSamples)
    {
        return circular(volDesc, SampledArc{geometry::Degree(360), numSamples});
    }
} // namespace elsa::trajectories
