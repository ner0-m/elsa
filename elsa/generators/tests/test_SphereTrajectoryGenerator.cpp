/**
 * @file test_SphereTrajectoryGenerator.cpp
 *
 * @brief Test for SphereTrajectoryGenerator class
 *
 * @author Michael Loipführer - initial code
 */

#include "doctest/doctest.h"

#include "SphereTrajectoryGenerator.h"

#include "Logger.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

SCENARIO("Create a Spherical Trajectory")
{
    using namespace geometry;

    const index_t s = 64;

    // Detector size is the volume size scalled by the square root of 2
    const auto expectedDetectorSize = static_cast<index_t>(s * std::sqrt(2));

    GIVEN("A 3D descriptor and 256 poses")
    {
        index_t numberOfPoses = 256;
        IndexVector_t volSize(3);
        volSize << s, s, s;
        VolumeDescriptor desc{volSize};

        WHEN("We create a spherical trajectory with 3 circular trajectories and 256 poses for this "
             "scenario")
        {
            auto diffCenterSource = static_cast<real_t>(s * 100);
            auto diffCenterDetector = static_cast<real_t>(s);

            auto sdesc = SphereTrajectoryGenerator::createTrajectory(
                numberOfPoses, desc, 5, diffCenterSource, diffCenterDetector);

            // Check that the detector size is correct
            REQUIRE(sdesc->getNumberOfCoefficientsPerDimension()[0] == expectedDetectorSize);
            REQUIRE(sdesc->getNumberOfCoefficientsPerDimension()[1] == expectedDetectorSize);
        }
    }
}
