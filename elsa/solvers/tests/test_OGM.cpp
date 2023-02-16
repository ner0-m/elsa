/**
 * @file test_OGM.cpp
 *
 * @brief Tests for the Optimized Gradient Method class
 *
 * @author Michael Loipführer - initial code
 */

#include "doctest/doctest.h"

#include <iostream>
#include "OGM.h"
#include "Scaling.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("OGM: Solving a simple linear problem", data_t, float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        VolumeDescriptor dd{{13, 15}};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Identity<data_t> op{dd};

        // using WLS problem here for ease of use
        // since OGM is very picky with the precision of the lipschitz constant of a problem we need
        // to pass it explicitly
        L2NormPow2<data_t> prob{op, b};

        WHEN("setting up an OGM solver")
        {
            OGM<data_t> solver{prob};

            THEN("the clone works correctly")
            {
                auto ogmClone = solver.clone();

                CHECK_NE(ogmClone.get(), &solver);
                CHECK_EQ(*ogmClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(200);

                    DataContainer<data_t> diff = solution - b;

                    // should have converged for the given number of iterations
                    CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0).epsilon(0.01));
                }
            }
        }

        // Something is wrong here this doesn't converge, I need to look into this
        // WHEN("setting up a preconditioned OGM solver")
        // {
        //     bVec = 1 / bVec.array();
        //     OGM<data_t> solver{prob, Scaling<data_t>{dd, DataContainer<data_t>{dd, bVec}}};
        //
        //     THEN("the clone works correctly")
        //     {
        //         auto ogmClone = solver.clone();
        //
        //         CHECK_NE(ogmClone.get(), &solver);
        //         CHECK_EQ(*ogmClone, solver);
        //
        //         AND_THEN("it works as expected")
        //         {
        //             // with a good preconditioner we should need fewer iterations than without
        //             auto solution = solver.solve(10);
        //
        //             DataContainer<data_t> diff = solution - b;
        //
        //             // should have converged for the given number of iterations
        //             CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0).epsilon(0.01));
        //         }
        //     }
        // }
    }
}

TEST_SUITE_END();
