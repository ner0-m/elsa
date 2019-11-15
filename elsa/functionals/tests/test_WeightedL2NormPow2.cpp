/**
 * \file test_WeightedL2NormPow2.cpp
 *
 * \brief Tests for the WeightedL2NormPow2 class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite
 * \author Tobias Lasser - modernization
 */

#include <catch2/catch.hpp>
#include "WeightedL2NormPow2.h"
#include "LinearResidual.h"
#include "Identity.h"

using namespace elsa;

SCENARIO("Testing the weighted, squared l2 norm functional")
{
    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 17;
        DataDescriptor dd(numCoeff);

        RealVector_t scalingData(dd.getNumberOfCoefficients());
        scalingData.setRandom();
        DataContainer scaleFactors(dd, scalingData);

        Scaling scalingOp(dd, scaleFactors);

        WHEN("instantiating")
        {
            WeightedL2NormPow2 func(scalingOp);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);
                REQUIRE(func.getWeightingOperator() == scalingOp);

                auto* linRes = dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasOperator() == false);
                REQUIRE(linRes->hasDataVector() == false);
            }

            THEN("a clone behaves as expected")
            {
                auto wl2Clone = func.clone();
                ;

                REQUIRE(wl2Clone.get() != &func);
                REQUIRE(*wl2Clone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                RealVector_t Wx = scalingData.array() * dataVec.array();
                REQUIRE(func.evaluate(x) == Approx(0.5f * Wx.dot(dataVec)));
                DataContainer dcWx(dd, Wx);
                REQUIRE(func.getGradient(x) == dcWx);

                REQUIRE(func.getHessian(x) == leaf(scalingOp));
            }
        }
    }

    GIVEN("a residual with data")
    {
        // linear residual
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        DataDescriptor dd(numCoeff);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer b(dd, randomData);

        Identity A(dd);

        LinearResidual linRes(A, b);

        // scaling operator
        RealVector_t scalingData(dd.getNumberOfCoefficients());
        scalingData.setRandom();
        DataContainer scaleFactors(dd, scalingData);

        Scaling scalingOp(dd, scaleFactors);

        WHEN("instantiating")
        {
            WeightedL2NormPow2 func(linRes, scalingOp);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);
                REQUIRE(func.getWeightingOperator() == scalingOp);

                auto* lRes = dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto wl2Clone = func.clone();

                REQUIRE(wl2Clone.get() != &func);
                REQUIRE(*wl2Clone == func);
            }

            THEN("the evaluate, gradient and Hessian work was expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                RealVector_t WRx = scalingData.array() * (dataVec - randomData).array();
                REQUIRE(func.evaluate(x) == Approx(0.5f * WRx.dot(dataVec - randomData)));

                DataContainer dcWRx(dd, WRx);
                REQUIRE(func.getGradient(x) == dcWRx);

                auto hessian = func.getHessian(x);
                RealVector_t Wx = scalingData.array() * dataVec.array();
                DataContainer dcWx(dd, Wx);
                REQUIRE(hessian.apply(x) == dcWx);
            }
        }
    }
}