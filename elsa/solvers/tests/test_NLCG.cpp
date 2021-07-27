
/**
 * @file test_NLCG.cpp
 *
 * @brief Tests for the NLCG class
 *
 * @author Maryna Shcherbak
 */

#include "doctest/doctest.h"
#include "NLCG.h"
#include "Identity.h"
#include "LinearResidual.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include "TransmissionLogLikelihood.h"


using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TYPE_TO_STRING(NLCG<float>);
TYPE_TO_STRING(NLCG<double>);

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEST_CASE_TEMPLATE("Scenario: Solving a simple problem", TestType, NLCG<float>, NLCG<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);
    using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd{numCoeff};

        Vector y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer<data_t> dcY(dd, y);

        Vector bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        DataContainer<data_t> dcB{dd, bVec};


        //Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        TransmissionLogLikelihood func(dd, dcY, dcB);

        Problem<data_t> prob{func, dcB, true};
        //generisches Problem
        data_t epsilon = std::numeric_limits<data_t>::epsilon();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        WHEN("setting up a NLCG solver")
        {
            TestType solver{prob};

            THEN("the clone works correctly")
            {
                auto nlcgClone = solver.clone();

                REQUIRE_NE(nlcgClone.get(), &solver);
                REQUIRE_EQ(*nlcgClone, solver);

                AND_THEN("it works as expected")
                {
                    auto solution = solver.solve(dd.getNumberOfCoefficients());

                    DataContainer<data_t> resultsDifference = scalingOp.apply(solution) - dcB;

                    // should have converged for the given number of iterations
                    REQUIRE_LE((resultsDifference).squaredL2Norm(),
                               epsilon * epsilon * dcB.squaredL2Norm());
                }
            }
        }
    }
}

TEST_SUITE_END();