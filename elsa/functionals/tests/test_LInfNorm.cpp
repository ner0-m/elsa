/**
 * \file test_LInfNorm.cpp
 *
 * \brief Tests for the LInfNorm class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite
 * \author Tobias Lasser - modernization
 */

#include <catch2/catch.hpp>
#include "LInfNorm.h"
#include "LinearResidual.h"
#include "Identity.h"

using namespace elsa;

SCENARIO("Testing the linf norm functional") {
    GIVEN("just data (no residual)") {
        IndexVector_t numCoeff(2); numCoeff << 8, 15;
        DataDescriptor dd(numCoeff);

        WHEN("instantiating") {
            LInfNorm func(dd);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto& residual = func.getResidual();
                auto* linRes = dynamic_cast<const LinearResidual<real_t>*>(&residual);
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);
            }

            THEN("a clone behaves as expected") {
                auto lInfClone = func.clone();

                REQUIRE(lInfClone.get() != &func);
                REQUIRE(*lInfClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer dc(dd, dataVec);

                REQUIRE(func.evaluate(dc) == dataVec.array().abs().maxCoeff());
                REQUIRE_THROWS_AS(func.getGradient(dc), std::logic_error);
                REQUIRE_THROWS_AS(func.getHessian(dc), std::logic_error);
            }
        }
    }

    GIVEN("a residual with data") {
        IndexVector_t numCoeff(3); numCoeff << 3, 7, 13;
        DataDescriptor dd(numCoeff);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer dc(dd, randomData);

        Identity idOp(dd);

        LinearResidual linRes(idOp, dc);

        WHEN("instantiating") {
            LInfNorm func(linRes);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto& residual = func.getResidual();
                auto* lRes = dynamic_cast<const LinearResidual<real_t>*>(&residual);
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected") {
                auto lInfClone = func.clone();

                REQUIRE(lInfClone.get() != &func);
                REQUIRE(*lInfClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                REQUIRE(func.evaluate(x) == Approx((dataVec - randomData).lpNorm<Eigen::Infinity>()));
                REQUIRE_THROWS_AS(func.getGradient(x), std::logic_error);
                REQUIRE_THROWS_AS(func.getHessian(x), std::logic_error);
            }
        }

        // TODO: add the rest with operator A=scaling, vector b=1 etc.
    }
}
