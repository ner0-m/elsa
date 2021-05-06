/**
 * @file test_LInfNorm.cpp
 *
 * @brief Tests for the LInfNorm class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - modernization
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "LInfNorm.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(std::complex<float>);
TYPE_TO_STRING(std::complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("LInfNorm: Testing without residual", TestType, float, double,
                   std::complex<float>, std::complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 8, 15;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            LInfNorm<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto& residual = func.getResidual();
                auto* linRes = dynamic_cast<const LinearResidual<TestType>*>(&residual);
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());
            }

            THEN("a clone behaves as expected")
            {
                auto lInfClone = func.clone();

                REQUIRE_NE(lInfClone.get(), &func);
                REQUIRE_EQ(*lInfClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> dc(dd, dataVec);

                REQUIRE_UNARY(checkApproxEq(func.evaluate(dc), dataVec.array().abs().maxCoeff()));
                REQUIRE_THROWS_AS(func.getGradient(dc), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(dc), LogicError);
            }
        }
    }

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 7, 13;
        VolumeDescriptor dd(numCoeff);

        Vector randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> dc(dd, randomData);

        Identity<TestType> idOp(dd);

        LinearResidual<TestType> linRes(idOp, dc);

        WHEN("instantiating")
        {
            LInfNorm<TestType> func(linRes);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto& residual = func.getResidual();
                auto* lRes = dynamic_cast<const LinearResidual<TestType>*>(&residual);
                REQUIRE_UNARY(lRes);
                REQUIRE_EQ(*lRes, linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto lInfClone = func.clone();

                REQUIRE_NE(lInfClone.get(), &func);
                REQUIRE_EQ(*lInfClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                REQUIRE_UNARY(checkApproxEq(
                    func.evaluate(x), (dataVec - randomData).template lpNorm<Eigen::Infinity>()));
                REQUIRE_THROWS_AS(func.getGradient(x), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(x), LogicError);
            }
        }

        // TODO: add the rest with operator A=scaling, vector b=1 etc.
    }
}

TEST_SUITE_END();
