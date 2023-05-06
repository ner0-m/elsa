#include <doctest/doctest.h>

#include "testHelpers.h"
#include "AnisotropicTV.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("AnisotropicTV", TestType, float, double, complex<float>, complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("1d case")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            AnisotropicTV<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);
            }

            THEN("a clone behaves as expected")
            {
                auto ATVClone = func.clone();

                REQUIRE_NE(ATVClone.get(), &func);
                REQUIRE_EQ(*ATVClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec << 0.1, -4.0, 5.4, -0.001;
                DataContainer<TestType> dc(dd, dataVec);

                REQUIRE(checkApproxEq(func.evaluate(dc), 18.902));
                REQUIRE_THROWS_AS(func.getGradient(dc), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(dc), LogicError);
            }
        }
    }

    GIVEN("2d case")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 3, 2;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            AnisotropicTV<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);
            }

            THEN("a clone behaves as expected")
            {
                auto ATVClone = func.clone();

                REQUIRE_NE(ATVClone.get(), &func);
                REQUIRE_EQ(*ATVClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec << -2, -1, 6, 3, 10, -4;
                DataContainer<TestType> dc(dd, dataVec);

                REQUIRE(checkApproxEq(func.evaluate(dc), 82));
                REQUIRE_THROWS_AS(func.getGradient(dc), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(dc), LogicError);
            }
        }
    }

    GIVEN("3d case")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 2, 3, 2;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            AnisotropicTV<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);
            }

            THEN("a clone behaves as expected")
            {
                auto ATVClone = func.clone();

                REQUIRE_NE(ATVClone.get(), &func);
                REQUIRE_EQ(*ATVClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
                DataContainer<TestType> dc(dd, dataVec);

                REQUIRE(checkApproxEq(func.evaluate(dc), 191));
                REQUIRE_THROWS_AS(func.getGradient(dc), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(dc), LogicError);
            }
        }
    }

    GIVEN("3d square case")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 2, 2, 2;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            AnisotropicTV<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);
            }

            THEN("a clone behaves as expected")
            {
                auto ATVClone = func.clone();

                REQUIRE_NE(ATVClone.get(), &func);
                REQUIRE_EQ(*ATVClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec << 1, 2, 3, 4, 5, 6, 7, 8;
                DataContainer<TestType> dc(dd, dataVec);

                REQUIRE(checkApproxEq(func.evaluate(dc), 96));
                REQUIRE_THROWS_AS(func.getGradient(dc), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(dc), LogicError);
            }
        }
    }
}