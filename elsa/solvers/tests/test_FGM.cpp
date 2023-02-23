/**
 * @file test_FGM.cpp
 *
 * @brief Tests for the Fast Gradient Method class
 *
 * @author Michael Loipführer - initial code
 */

#include "LeastSquares.h"
#include "doctest/doctest.h"

#include <iostream>
#include "FGM.h"
#include "Identity.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "JacobiPreconditioner.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("FGM: Solving a simple linear problem", data_t, float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    VolumeDescriptor dd{{13, 11}};
    GIVEN("a linear problem")
    {
        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Identity<data_t> A{dd};
        auto prob = LeastSquares<data_t>(A, b);

        WHEN("setting up a FGM solver")
        {
            FGM<data_t> solver{prob, epsilon};

            auto solution = solver.solve(100);

            DataContainer<data_t> diff = A.apply(solution) - b;

            // should have converged for the given number of iterations
            CHECK_EQ(diff.squaredL2Norm(), doctest::Approx(0));

            THEN("the clone works correctly")
            {
                auto clone = solver.clone();

                CHECK_NE(clone.get(), &solver);
                CHECK_EQ(*clone, solver);
            }
        }
        //
        //         WHEN("setting up a preconditioned FGM solver")
        //         {
        //             auto preconditionerInverse = JacobiPreconditioner<data_t>(scalingOp, true);
        //             TestType solver{prob, preconditionerInverse, epsilon};
        //
        //             THEN("the clone works correctly")
        //             {
        //                 auto fgmClone = solver.clone();
        //
        //                 CHECK_NE(fgmClone.get(), &solver);
        //                 CHECK_EQ(*fgmClone, solver);
        //
        //                 AND_THEN("it works as expected")
        //                 {
        //                     // with a good preconditioner we should need fewer iterations than
        //                     without auto solution = solver.solve(40);
        //
        //                     DataContainer<data_t> resultsDifference = scalingOp.apply(solution) -
        //                     b;
        //
        //                     // should have converged for the given number of iterations
        //                     CHECK_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
        //                                                 epsilon * epsilon * b.squaredL2Norm(),
        //                                                 0.1f));
        //                 }
        //             }
        //         }
    }
}

TEST_SUITE_END();
