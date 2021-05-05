
/**
 * \file test_SIRT.cpp
 *
 * \brief Tests for the SIRT class
 *
 * \author Maryna Shcherbak
 */

#include <catch2/catch.hpp>
#include "SIRT.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

using namespace elsa;

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEMPLATE_TEST_CASE("Scenario: Solving a simple linear problem", "", SIRT<float>, SIRT<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity<data_t> idOp(dd);
        WLSProblem prob(idOp, dcB);

        WHEN("setting up a SIRT solver with fixed step size")
        {
            TestType solver(prob);

            THEN("the clone works correctly")
            {
                auto gdClone = solver.clone();

                REQUIRE(gdClone.get() != &solver);
                REQUIRE(*gdClone == solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(1000);
                    REQUIRE(solution.squaredL2Norm() == Approx(bVec.squaredNorm()));
                }
            }
        }
    }
}