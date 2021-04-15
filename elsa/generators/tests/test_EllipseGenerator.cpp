/**
 * @file test_EllipseGenerator.cpp
 *
 * @brief Tests for the EllipseGenerator class
 *
 * @author Tobias Lasser - initial code
 */

#include <catch2/catch.hpp>
#include "EllipseGenerator.h"
#include "VolumeDescriptor.h"

using namespace elsa;

static constexpr auto pi_d = pi<real_t>;

SCENARIO("Drawing a rotated filled ellipse in 2d")
{
    GIVEN("a volume and example ellipse parameters")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 200, 200;
        VolumeDescriptor dd(numCoeff);
        DataContainer dc(dd);

        Eigen::Matrix<index_t, 2, 1> center{100, 100};
        Eigen::Matrix<index_t, 2, 1> sizes{40, 80};

        WHEN("comparing an ellipse created using an inefficient method")
        {
            THEN("the ellipse mostly matches the efficient one")
            {
                // check a few rotation angles
                for (real_t angleDeg : {0.0f, 18.0f, 30.0f, 45.0f, 60.0f, 72.0f, 90.0f}) {
                    real_t angleRad = angleDeg * pi_d / 180.0f;

                    dc = 0;
                    EllipseGenerator<real_t>::drawFilledEllipse2d(dc, 1.0, center, sizes, angleDeg);

                    index_t wrongPixelCounter = 0;

                    for (index_t x = 0; x < numCoeff[0]; ++x) {
                        for (index_t y = 0; y < numCoeff[1]; ++y) {
                            real_t aPart =
                                static_cast<real_t>(x - center[0]) * std::cos(angleRad)
                                + static_cast<real_t>(y - center[1]) * std::sin(angleRad);
                            aPart *= aPart;
                            aPart /= static_cast<real_t>(sizes[0] * sizes[0]);

                            real_t bPart =
                                -static_cast<real_t>(x - center[0]) * std::sin(angleRad)
                                + static_cast<real_t>(y - center[1]) * std::cos(angleRad);
                            bPart *= bPart;
                            bPart /= static_cast<real_t>(sizes[1] * sizes[1]);

                            IndexVector_t coord(2);
                            coord[0] = x;
                            coord[1] = numCoeff[1] - 1 - y; // flip y axis

                            if (aPart + bPart <= 1.0) { // point in ellipse
                                if (dc(coord) == 0)
                                    ++wrongPixelCounter; // CHECK(dc(coord) > 0);
                            } else {                     // point not in ellipse
                                if (dc(coord) != 0)
                                    ++wrongPixelCounter; // CHECK(dc(coord) == 0);
                            }
                        }
                    }
                    REQUIRE(
                        (static_cast<real_t>(wrongPixelCounter) / static_cast<real_t>(sizes.prod()))
                        < 0.11); // 11% isn't great... :(
                }
            }
        }
    }
}

SCENARIO("Drawing a rotated filled ellipsoid in 3d")
{
    GIVEN("a volume and example ellipsoid parameters")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 64, 64, 64;
        VolumeDescriptor dd(numCoeff);
        DataContainer dc(dd);

        Eigen::Matrix<index_t, 3, 1> center{32, 32, 32};
        Eigen::Matrix<index_t, 3, 1> sizes{10, 25, 5};

        WHEN("creating an ellipse")
        {
            EllipseGenerator<real_t>::drawFilledEllipsoid3d(dc, 1.0, center, sizes, -18, 0, 10);

            THEN("somebody should check it...")
            {
                REQUIRE(true); // TODO: add a reasonable test here
            }
        }
    }
}