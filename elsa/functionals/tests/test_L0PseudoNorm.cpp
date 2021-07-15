/**
 * @file test_L0PseudoNorm.cpp
 *
 * @brief Tests for the L0PseudoNorm class
 *
 * @author Andi Braimllari
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "L0PseudoNorm.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("L0PseudoNorm: Testing without residual", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor volDescr(numCoeff);

        WHEN("instantiating")
        {
            L0PseudoNorm<TestType> l0PseudoNorm(volDescr);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(l0PseudoNorm.getDomainDescriptor(), volDescr);

                const auto& residual = l0PseudoNorm.getResidual();
                const auto* linRes = downcast_safe<LinearResidual<TestType>>(&residual);
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());
            }

            THEN("a clone behaves as expected")
            {
                auto l0Clone = l0PseudoNorm.clone();

                REQUIRE_NE(l0Clone.get(), &l0PseudoNorm);
                REQUIRE_EQ(*l0Clone, l0PseudoNorm);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(volDescr.getNumberOfCoefficients());
                dataVec << 7, 0, 2, 5;
                DataContainer<TestType> dc(volDescr, dataVec);

                REQUIRE_UNARY(checkApproxEq(l0PseudoNorm.evaluate(dc), 3));
                REQUIRE_THROWS_AS(l0PseudoNorm.getGradient(dc), std::logic_error);
                REQUIRE_THROWS_AS(l0PseudoNorm.getHessian(dc), std::logic_error);
            }
        }
    }
}

TEST_CASE_TEMPLATE("L0PseudoNorm: Testing with residual", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor volDescr(numCoeff);

        Vector randomData(volDescr.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> dc(volDescr, randomData);

        Identity<TestType> idOp(volDescr);

        LinearResidual<TestType> linRes(idOp, dc);

        WHEN("instantiating")
        {
            L0PseudoNorm<TestType> l0PseudoNorm(linRes);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(l0PseudoNorm.getDomainDescriptor(), volDescr);

                const auto& residual = l0PseudoNorm.getResidual();
                const auto* lRes = downcast_safe<LinearResidual<TestType>>(&residual);
                REQUIRE_UNARY(lRes);
                REQUIRE_EQ(*lRes, linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto l0Clone = l0PseudoNorm.clone();

                REQUIRE_NE(l0Clone.get(), &l0PseudoNorm);
                REQUIRE_EQ(*l0Clone, l0PseudoNorm);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(volDescr.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(volDescr, dataVec);

                REQUIRE_UNARY(checkApproxEq(l0PseudoNorm.evaluate(x),
                                            (TestType)(randomData.array().cwiseAbs()
                                                       >= std::numeric_limits<TestType>::epsilon())
                                                .count()));
                REQUIRE_THROWS_AS(l0PseudoNorm.getGradient(x), std::logic_error);
                REQUIRE_THROWS_AS(l0PseudoNorm.getHessian(x), std::logic_error);
            }
        }
    }
}

TEST_SUITE_END();
