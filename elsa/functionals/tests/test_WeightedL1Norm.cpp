/**
 * @file test_WeightedL1Norm.cpp
 *
 * @brief Tests for the WeightedL1Norm class
 *
 * @author Andi Braimllari
 */

#include "WeightedL1Norm.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("Testing the weighted, squared l1 norm functional", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 17;
        VolumeDescriptor dd(numCoeff);

        DataContainer<TestType> scaleFactors(dd);
        scaleFactors = 1;

        WHEN("instantiating")
        {
            WeightedL1Norm<TestType> func(scaleFactors);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);
                REQUIRE(func.getWeightingOperator() == scaleFactors);

                const auto* linRes =
                    dynamic_cast<const LinearResidual<TestType>*>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasOperator() == false);
                REQUIRE(linRes->hasDataVector() == false);
            }

            THEN("a clone behaves as expected")
            {
                auto wl1Clone = func.clone();

                REQUIRE(wl1Clone.get() != &func);
                REQUIRE(*wl1Clone == func);
            }

            Vector dataVec(dd.getNumberOfCoefficients());
            dataVec.setRandom();
            DataContainer<TestType> x(dd, dataVec);

            THEN("the evaluate works as expected")
            {
                REQUIRE(func.evaluate(x) == Approx(scaleFactors.dot(abs(x))));
            }

            THEN("the gradient and Hessian throw as expected")
            {
                REQUIRE_THROWS_AS(func.getGradient(x), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(x), LogicError);
            }
        }
    }

    GIVEN("a residual with data")
    {
        // linear residual
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        VolumeDescriptor dd(numCoeff);

        Vector randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> b(dd, randomData);

        Identity<TestType> A(dd);

        LinearResidual<TestType> linRes(A, b);

        // scaling operator
        DataContainer<TestType> scaleFactors(dd);
        scaleFactors = 1;

        WHEN("instantiating")
        {
            WeightedL1Norm<TestType> func(linRes, scaleFactors);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);
                REQUIRE(func.getWeightingOperator() == scaleFactors);

                const auto* lRes =
                    dynamic_cast<const LinearResidual<TestType>*>(&func.getResidual());
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto wl1Clone = func.clone();

                REQUIRE(wl1Clone.get() != &func);
                REQUIRE(*wl1Clone == func);
            }

            THEN("the evaluate, gradient and Hessian work was expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                REQUIRE(func.evaluate(x) == Approx(scaleFactors.dot(abs(x - b))));
                REQUIRE_THROWS_AS(func.getGradient(x), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(x), LogicError);
            }
        }
    }
}

TEST_SUITE_END();
