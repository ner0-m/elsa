/**
 * @file test_L0PseudoNorm.cpp
 *
 * @brief Tests for the L0PseudoNorm class
 *
 * @author Andi Braimllari
 */

#include "L0PseudoNorm.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

#include <catch2/catch.hpp>

using namespace elsa;

SCENARIO("Testing the l0 pseudo-norm functional")
{
    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating")
        {
            L0PseudoNorm<real_t> l0PseudoNorm(volDescr);

            THEN("the functional is as expected")
            {
                REQUIRE(l0PseudoNorm.getDomainDescriptor() == volDescr);

                const auto& residual = l0PseudoNorm.getResidual();
                const auto* linRes = dynamic_cast<const LinearResidual<real_t>*>(&residual);
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);
            }

            THEN("a clone behaves as expected")
            {
                auto l0Clone = l0PseudoNorm.clone();

                REQUIRE(l0Clone.get() != &l0PseudoNorm);
                REQUIRE(*l0Clone == l0PseudoNorm);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(volDescr.getNumberOfCoefficients());
                dataVec << 7, 0, 2, 5;
                DataContainer<real_t> dc(volDescr, dataVec);

                REQUIRE(l0PseudoNorm.evaluate(dc) == 3);
                REQUIRE_THROWS_AS(l0PseudoNorm.getGradient(dc), std::logic_error);
                REQUIRE_THROWS_AS(l0PseudoNorm.getHessian(dc), std::logic_error);
            }
        }
    }

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor volDescr(numCoeff);

        RealVector_t randomData(volDescr.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<real_t> dc(volDescr, randomData);

        Identity<real_t> idOp(volDescr);

        LinearResidual<real_t> linRes(idOp, dc);

        WHEN("instantiating")
        {
            L0PseudoNorm<real_t> l0PseudoNorm(linRes);

            THEN("the functional is as expected")
            {
                REQUIRE(l0PseudoNorm.getDomainDescriptor() == volDescr);

                const auto& residual = l0PseudoNorm.getResidual();
                const auto* lRes = dynamic_cast<const LinearResidual<real_t>*>(&residual);
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto l0Clone = l0PseudoNorm.clone();

                REQUIRE(l0Clone.get() != &l0PseudoNorm);
                REQUIRE(*l0Clone == l0PseudoNorm);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(volDescr.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<real_t> x(volDescr, dataVec);

                REQUIRE(l0PseudoNorm.evaluate(x)
                        == (real_t)(randomData.array().cwiseAbs()
                                    >= std::numeric_limits<real_t>::epsilon())
                               .count());
                REQUIRE_THROWS_AS(l0PseudoNorm.getGradient(x), std::logic_error);
                REQUIRE_THROWS_AS(l0PseudoNorm.getHessian(x), std::logic_error);
            }
        }
    }
}
