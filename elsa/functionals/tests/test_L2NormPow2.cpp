/**
 * \file test_L2NormPow2.cpp
 *
 * \brief Tests for the L2NormPow2 class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite
 * \author Tobias Lasser - modernization
 */

#include <catch2/catch.hpp>
#include "L2NormPow2.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Testing the l2 norm (squared) functional")
{
    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 13;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            L2NormPow2 func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto* linRes = dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasOperator() == false);
                REQUIRE(linRes->hasDataVector() == false);
            }

            THEN("a clone behaves as expected")
            {
                auto l2Clone = func.clone();
                ;

                REQUIRE(l2Clone.get() != &func);
                REQUIRE(*l2Clone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                REQUIRE(func.evaluate(x) == Approx(0.5f * dataVec.squaredNorm()));
                REQUIRE(func.getGradient(x) == x);

                Identity idOp(dd);
                REQUIRE(func.getHessian(x) == leaf(idOp));
            }
        }
    }

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        VolumeDescriptor dd(numCoeff);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer b(dd, randomData);

        Identity A(dd);

        LinearResidual linRes(A, b);

        WHEN("instantiating")
        {
            L2NormPow2 func(linRes);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto* lRes = dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto l2Clone = func.clone();

                REQUIRE(l2Clone.get() != &func);
                REQUIRE(*l2Clone == func);
            }

            THEN("the evaluate, gradient and Hessian work was expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                REQUIRE(func.evaluate(x) == Approx(0.5f * (dataVec - randomData).squaredNorm()));

                DataContainer grad(dd, (dataVec - randomData).eval());
                REQUIRE(func.getGradient(x) == grad);

                auto hessian = func.getHessian(x);
                REQUIRE(hessian.apply(x) == x);
            }
        }
    }
}