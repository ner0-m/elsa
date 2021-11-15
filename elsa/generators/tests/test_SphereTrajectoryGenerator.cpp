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

TEST_CASE("SphereTrajectoryGenerator: Create a Spherical Trajectory")
{
    using namespace geometry;

    const index_t s = 64;

    // Detector size is the volume size scalled by the square root of 2
    const auto expectedDetectorSize = static_cast<index_t>(s * std::sqrt(2));

    GIVEN("A 2D descriptor and 256 poses")
    {
        index_t numberOfPoses = 256;
        VolumeDescriptor desc({s, s});

        WHEN("Trying to create a 2D spherical trajectory")
        {
            geometry::SourceToCenterOfRotation diffCenterSource(s * 100);
            geometry::CenterOfRotationToDetector diffCenterDetector(s);

            REQUIRE_THROWS_AS(SphereTrajectoryGenerator::createTrajectory(
                                  numberOfPoses, desc, 5, diffCenterSource, diffCenterDetector),
                              InvalidArgumentError);
        }
    }

    GIVEN("A 3D descriptor and 256 poses")
    {
        index_t numberOfPoses = 256;
        VolumeDescriptor desc{{s, s, s}};

        WHEN("We create a spherical trajectory with 3 circular trajectories and 256 poses for this "
             "scenario")
        {
            geometry::SourceToCenterOfRotation diffCenterSource(s * 100);
            geometry::CenterOfRotationToDetector diffCenterDetector(s);

            auto sdesc = SphereTrajectoryGenerator::createTrajectory(
                numberOfPoses, desc, 5, diffCenterSource, diffCenterDetector);

            // Check that the detector size is correct
            REQUIRE_EQ(sdesc->getNumberOfCoefficientsPerDimension()[0], expectedDetectorSize);
            REQUIRE_EQ(sdesc->getNumberOfCoefficientsPerDimension()[1], expectedDetectorSize);
        }
    }
}
