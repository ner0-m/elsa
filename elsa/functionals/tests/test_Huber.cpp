/**
 * @file test_Huber.cpp
 *
 * @brief Tests for the Huber class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - modernization
 */

#include <doctest/doctest.h>

#include "Huber.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("Huber: Testing without residual only data", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 7, 16, 29;
        VolumeDescriptor dd(numCoeff);

        real_t delta = 10;

        WHEN("instantiating")
        {
            Huber<TestType> func(dd, delta);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());
            }

            THEN("a clone behaves as expected")
            {
                auto huberClone = func.clone();

                REQUIRE_NE(huberClone.get(), &func);
                REQUIRE_EQ(*huberClone, func);
            }

            Vector dataVec(dd.getNumberOfCoefficients());
            dataVec.setRandom();

            // fix the first entries to be bigger/smaller than delta
            dataVec[0] = delta + 1;
            dataVec[1] = delta + 2;
            dataVec[2] = delta + 3;
            dataVec[3] = delta - 1;
            dataVec[4] = delta - 2;
            dataVec[5] = delta - 3;

            DataContainer<TestType> x(dd, dataVec);

            // compute the "true" values
            TestType trueValue = 0;
            Vector trueGrad(dd.getNumberOfCoefficients());
            for (index_t i = 0; i < dataVec.size(); ++i) {
                TestType value = dataVec[i];
                if (std::abs(value) <= delta) {
                    trueValue += 0.5f * value * value;
                    trueGrad[i] = value;
                } else {
                    trueValue += delta * (std::abs(value) - 0.5f * delta);
                    trueGrad[i] = (value > 0) ? delta : -delta;
                }
            }

            THEN("the evaluate works as expected")
            {
                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
            }
            THEN("the gradient works as expected")
            {
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(checkApproxEq(func.getGradient(x), dcTrueGrad));
            }
            THEN("the Hessian works as expected")
            {
                auto hessian = func.getHessian(x);
                auto hx = hessian.apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE_UNARY(
                        checkApproxEq(hx[i], ((std::abs(dataVec[i]) <= delta) ? x[i] : 0)));
            }
        }
    }
}

TEST_CASE_TEMPLATE("Huber<TestType>: Testing with residual", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

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

        real_t delta = 20;

        WHEN("instantiating")
        {
            Huber<TestType> func(linRes, delta);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* lRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(lRes);
                REQUIRE_EQ(*lRes, linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto huberClone = func.clone();

                REQUIRE_NE(huberClone.get(), &func);
                REQUIRE_EQ(*huberClone, func);
            }

            Vector dataVec(dd.getNumberOfCoefficients());
            dataVec.setRandom();
            DataContainer<TestType> x(dd, dataVec);

            // compute the "true" values
            TestType trueValue = 0;
            Vector trueGrad(dd.getNumberOfCoefficients());
            for (index_t i = 0; i < dataVec.size(); ++i) {
                TestType value = dataVec[i] - randomData[i];
                if (std::abs(value) <= delta) {
                    trueValue += 0.5f * value * value;
                    trueGrad[i] = value;
                } else {
                    trueValue += delta * (std::abs(value) - 0.5f * delta);
                    trueGrad[i] = (value > 0) ? delta : -delta;
                }
            }

            THEN("the evaluate works as expected")
            {
                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
            }

            THEN("the gradient works as expected")
            {
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));
            }
            THEN("the Hessian works as expected")
            {

                auto hessian = func.getHessian(x);
                auto hx = hessian.apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE(hx[i] == ((std::abs(dataVec[i] - randomData[i]) <= delta) ? x[i] : 0));
            }
        }
    }
}

TEST_SUITE_END();
