/**
 * @file test_LimitedAngleTrajectoryGenerator.cpp
 *
 * @brief Tests for the LimitedAngleTrajectoryGenerator class
 *
 * @author Andi Braimllari
 */

#include "LimitedAngleTrajectoryGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

#include "testHelpers.h"
#include "doctest/doctest.h"

using namespace elsa;
using namespace doctest;

TEST_CASE("LimitedAngleTrajectoryGenerator: Create a Limited Angle Trajectory")
{
    using namespace geometry;

    const index_t s = 64;

    // Detector size is the volume size scaled by the square root of 2
    const auto expectedDetectorSize = static_cast<index_t>(s * std::sqrt(2));

    GIVEN("A 2D descriptor and 256 angles")
    {
        index_t numberOfAngles = 256;
        IndexVector_t volSize(2);
        volSize << s, s;
        VolumeDescriptor desc{volSize};

        WHEN("We create a half limited angle trajectory for this scenario")
        {
            THEN("?") {}
        }
    }
}
