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

#include <catch2/catch.hpp>

using namespace elsa;

SCENARIO("Testing the weighted, squared l1 norm functional")
{
    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 17;
        VolumeDescriptor dd(numCoeff);

        RealVector_t scalingData(dd.getNumberOfCoefficients());
        scalingData.setRandom();
        DataContainer scaleFactors(dd, scalingData);

        WHEN("instantiating")
        {
            WeightedL1Norm func(scaleFactors);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);
                REQUIRE(func.getWeightingOperator() == scaleFactors);

                const auto* linRes =
                    dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasOperator() == false);
                REQUIRE(linRes->hasDataVector() == false);
            }

            THEN("a clone behaves as expected")
            {
                auto wl1Clone = func.clone();
                ;

                REQUIRE(wl1Clone.get() != &func);
                REQUIRE(*wl1Clone == func);
            }

            RealVector_t dataVec(dd.getNumberOfCoefficients());
            dataVec.setRandom();
            DataContainer x(dd, dataVec);

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

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer b(dd, randomData);

        Identity A(dd);

        LinearResidual linRes(A, b);

        // scaling operator
        RealVector_t scalingData(dd.getNumberOfCoefficients());
        scalingData.setRandom();
        DataContainer scaleFactors(dd, scalingData);

        WHEN("instantiating")
        {
            WeightedL1Norm func(linRes, scaleFactors);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);
                REQUIRE(func.getWeightingOperator() == scaleFactors);

                const auto* lRes = dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
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
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                REQUIRE(func.evaluate(x) == Approx(scaleFactors.dot(abs(x - b))));

                REQUIRE_THROWS_AS(func.getGradient(x), LogicError);

                REQUIRE_THROWS_AS(func.getHessian(x), LogicError);
            }
        }
    }
}
