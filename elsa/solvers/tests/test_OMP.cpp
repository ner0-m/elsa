/**
 * @file test_OMP.cpp
 *
 * @brief Tests for the OMP class
 *
 * @author Jonas Buerger
 */

#include "doctest/doctest.h"

#include <limits>
#include "Error.h"
#include "OMP.h"
#include "Dictionary.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "RepresentationProblem.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE("OMP: Solving a RepresentationProblem")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a RepresentationProblem")
    {
        VolumeDescriptor dd({2});
        const index_t nAtoms = 3;
        IdenticalBlocksDescriptor ibd(nAtoms, dd);

        RealVector_t dictVec(ibd.getNumberOfCoefficients());
        dictVec << 0, 1, 1, 0, 1, -1;
        DataContainer dcDict(ibd, dictVec);

        Dictionary dictOp(dcDict);

        RealVector_t signalVec(dd.getNumberOfCoefficients());
        signalVec << 5, 3;
        DataContainer dcSignal(dd, signalVec);

        RepresentationProblem reprProb(dictOp, dcSignal);

        WHEN("setting up a OMP solver")
        {
            OMP solver(reprProb, std::numeric_limits<real_t>::epsilon());

            THEN("cloned OMP solver equals original OMP solver")
            {
                auto ompClone = solver.clone();

                REQUIRE_NE(ompClone.get(), &solver);
                REQUIRE_EQ(*ompClone, solver);
            }

            THEN("the solution is correct")
            {
                auto solution = solver.solve(2);

                RealVector_t expected(nAtoms);
                expected << 3, 5, 0;

                REQUIRE_UNARY(isApprox(solution, expected));
            }
        }
    }
}

TEST_SUITE_END();
