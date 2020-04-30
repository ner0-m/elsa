/**
 * \file test_L1Norm.cpp
 *
 * \brief Tests for the L1Norm class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite
 * \author Tobias Lasser - modernization
 */

#include <catch2/catch.hpp>
#include "L1Norm.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Testing the l1 norm functional")
{
    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            L1Norm func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto& residual = func.getResidual();
                auto* linRes = dynamic_cast<const LinearResidual<real_t>*>(&residual);
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);
            }

            THEN("a clone behaves as expected")
            {
                auto l1Clone = func.clone();

                REQUIRE(l1Clone.get() != &func);
                REQUIRE(*l1Clone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec << -9, -4, 0, 1;
                DataContainer dc(dd, dataVec);

                REQUIRE(func.evaluate(dc) == 14);
                REQUIRE_THROWS_AS(func.getGradient(dc), std::logic_error);
                REQUIRE_THROWS_AS(func.getHessian(dc), std::logic_error);
            }
        }
    }

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor dd(numCoeff);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer dc(dd, randomData);

        Identity idOp(dd);

        LinearResidual linRes(idOp, dc);

        WHEN("instantiating")
        {
            L1Norm func(linRes);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto& residual = func.getResidual();
                auto* lRes = dynamic_cast<const LinearResidual<real_t>*>(&residual);
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto l1Clone = func.clone();

                REQUIRE(l1Clone.get() != &func);
                REQUIRE(*l1Clone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                REQUIRE(func.evaluate(x) == Approx((dataVec - randomData).lpNorm<1>()));
                REQUIRE_THROWS_AS(func.getGradient(x), std::logic_error);
                REQUIRE_THROWS_AS(func.getHessian(x), std::logic_error);
            }
        }
    }
}
