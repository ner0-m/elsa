#include "WeightedL1Norm.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"
#include "TypeCasts.hpp"

#include <doctest/doctest.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("WeightedL1Norm: Testing the weighted, l1 norm functional", TestType, float,
                   double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    // GIVEN("a linear residual and weights with a non-positive element")
    // {
    //     IndexVector_t numCoeff(2);
    //     numCoeff << 25, 27;
    //     VolumeDescriptor dd(numCoeff);
    //
    //     Vector randomData(dd.getNumberOfCoefficients());
    //     randomData.setRandom();
    //     DataContainer<TestType> b(dd, randomData);
    //
    //     Identity<TestType> A(dd);
    //
    //     LinearResidual<TestType> linRes(A, b);
    //
    //     // scaling operator
    //     DataContainer<TestType> scaleFactors(dd);
    //     scaleFactors = 1;
    //     scaleFactors[3] = -8;
    //
    //     WHEN("instantiating an WeightedL1Norm object")
    //     {
    //         THEN("an InvalidArgumentError is thrown")
    //         {
    //             REQUIRE_THROWS_AS(WeightedL1Norm<TestType>{scaleFactors}, InvalidArgumentError);
    //             REQUIRE_THROWS_AS(WeightedL1Norm<TestType>(linRes, scaleFactors),
    //                               InvalidArgumentError);
    //         }
    //     }
    // }

    GIVEN("weights of value 1 and no residual")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 17;
        VolumeDescriptor dd(numCoeff);

        DataContainer<TestType> scaleFactors(dd);
        scaleFactors = 1;

        WHEN("instantiating an WeightedL1Norm object")
        {
            WeightedL1Norm<TestType> func(scaleFactors);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);
                REQUIRE(func.getWeightingOperator() == scaleFactors);
            }

            THEN("a clone behaves as expected")
            {
                auto wl1Clone = func.clone();

                REQUIRE(wl1Clone.get() != &func);
                REQUIRE(*wl1Clone == func);
            }

            Vector dataVec(dd.getNumberOfCoefficients());
            dataVec.setRandom();
            DataContainer<TestType> x(dd, dataVec);

            THEN("the evaluate works as expected")
            {
                REQUIRE(func.evaluate(x) == Approx(scaleFactors.dot(cwiseAbs(x))));
            }

            THEN("the gradient and Hessian throw as expected")
            {
                REQUIRE_THROWS_AS(func.getGradient(x), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(x), LogicError);
            }
        }
    }

    // GIVEN("different sizes of the linear residual and weighting operator")
    // {
    //     // linear residual
    //     IndexVector_t numCoeff(2);
    //     numCoeff << 47, 11;
    //     VolumeDescriptor dd(numCoeff);
    //
    //     // linear residual
    //     IndexVector_t otherNumCoeff(3);
    //     otherNumCoeff << 15, 24, 4;
    //     VolumeDescriptor otherDD(otherNumCoeff);
    //
    //     Vector randomData(dd.getNumberOfCoefficients());
    //     randomData.setRandom();
    //     DataContainer<TestType> b(dd, randomData);
    //
    //     Identity<TestType> A(dd);
    //
    //     LinearResidual<TestType> linRes(A, b);
    //
    //     // scaling operator
    //     DataContainer<TestType> scaleFactors(otherDD);
    //     scaleFactors = 1;
    //
    //     WHEN("instantiating an WeightedL1Norm object")
    //     {
    //         THEN("an InvalidArgumentError is thrown")
    //         {
    //             REQUIRE_THROWS_AS(WeightedL1Norm<TestType>(linRes, scaleFactors),
    //                               InvalidArgumentError);
    //         }
    //     }
    // }

    // GIVEN("weights of value 1 and a linear residual")
    // {
    //     // linear residual
    //     IndexVector_t numCoeff(2);
    //     numCoeff << 47, 11;
    //     VolumeDescriptor dd(numCoeff);
    //
    //     Vector randomData(dd.getNumberOfCoefficients());
    //     randomData.setRandom();
    //     DataContainer<TestType> b(dd, randomData);
    //
    //     Identity<TestType> A(dd);
    //
    //     LinearResidual<TestType> linRes(A, b);
    //
    //     // scaling operator
    //     DataContainer<TestType> scaleFactors(dd);
    //     scaleFactors = 1;
    //
    //     WHEN("instantiating an WeightedL1Norm object")
    //     {
    //         WeightedL1Norm<TestType> func(linRes, scaleFactors);
    //
    //         THEN("the functional is as expected")
    //         {
    //             REQUIRE(func.getDomainDescriptor() == dd);
    //             REQUIRE(func.getWeightingOperator() == scaleFactors);
    //
    //             const auto* lRes =
    //                 dynamic_cast<const LinearResidual<TestType>*>(&func.getResidual());
    //             REQUIRE(lRes);
    //             REQUIRE(*lRes == linRes);
    //         }
    //
    //         THEN("a clone behaves as expected")
    //         {
    //             auto wl1Clone = func.clone();
    //
    //             REQUIRE(wl1Clone.get() != &func);
    //             REQUIRE(*wl1Clone == func);
    //         }
    //
    //         THEN("the evaluate, gradient and Hessian work was expected")
    //         {
    //             Vector dataVec(dd.getNumberOfCoefficients());
    //             dataVec.setRandom();
    //             DataContainer<TestType> x(dd, dataVec);
    //
    //             REQUIRE(func.evaluate(x) == Approx(scaleFactors.dot(cwiseAbs(x - b))));
    //             REQUIRE_THROWS_AS(func.getGradient(x), LogicError);
    //             REQUIRE_THROWS_AS(func.getHessian(x), LogicError);
    //         }
    //     }
    // }
}

TEST_SUITE_END();
