#include "doctest/doctest.h"

#include "L2NormPow2.h"
#include <iostream>
#include "JacobiPreconditioner.h"
#include "SQS.h"
#include "SubsetSampler.h"
#include "PlanarDetectorDescriptor.h"
#include "Identity.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("SQS: Solving a simple linear problem", data_t, float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        VolumeDescriptor dd{{8, 9}};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB{dd, bVec};

        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        // using WLS problem here for ease of use
        L2NormPow2<data_t> prob{scalingOp, dcB};

        WHEN("setting up a SQS solver")
        {
            SQS<data_t> solver{prob, true};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                CHECK_NE(sqsClone.get(), &solver);
                CHECK_EQ(*sqsClone, solver);
            }

            AND_THEN("it works as expected")
            {
                auto solution = solver.solve(50);

                DataContainer<data_t> diff = scalingOp.apply(solution) - dcB;

                // should have converged for the given number of iterations
                CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0));
            }
        }

        WHEN("setting up a preconditioned SQS solver")
        {
            auto preconditioner = JacobiPreconditioner<data_t>(scalingOp, false);
            SQS<data_t> solver{prob, preconditioner, true};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE_NE(sqsClone.get(), &solver);
                REQUIRE_EQ(*sqsClone, solver);

                AND_THEN("it works as expected")
                {
                    // with a good preconditioner we should need fewer iterations than without
                    auto solution = solver.solve(200);

                    DataContainer<data_t> diff = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("SQS: Solving a linear problem with ordered subsets", data_t, float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a linear problem")
    {
        VolumeDescriptor dd{{8, 9}};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB{dd, bVec};

        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        // using WLS problem here for ease of use
        L2NormPow2<data_t> prob{scalingOp, dcB};

        WHEN("setting up a SQS solver")
        {
            SQS<data_t> solver{prob, true};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                CHECK_NE(sqsClone.get(), &solver);
                CHECK_EQ(*sqsClone, solver);
            }

            AND_THEN("it works as expected")
            {
                auto solution = solver.solve(50);

                DataContainer<data_t> diff = scalingOp.apply(solution) - dcB;

                // should have converged for the given number of iterations
                CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0));
            }
        }

        WHEN("setting up a preconditioned SQS solver")
        {
            auto preconditioner = JacobiPreconditioner<data_t>(scalingOp, false);
            SQS<data_t> solver{prob, preconditioner, true};

            THEN("the clone works correctly")
            {
                auto sqsClone = solver.clone();

                REQUIRE_NE(sqsClone.get(), &solver);
                REQUIRE_EQ(*sqsClone, solver);

                AND_THEN("it works as expected")
                {
                    // with a good preconditioner we should need fewer iterations than without
                    auto solution = solver.solve(200);

                    DataContainer<data_t> diff = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0));
                }
            }
        }
    }
}

TEST_SUITE_END();
