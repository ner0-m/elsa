/**
 * @file test_Problem.cpp
 *
 * @brief Tests for the Problem class
 *
 * @author David Frank - initial code
 * @author Tobias Lasser - rewrite
 */

#include "doctest/doctest.h"

#include <iostream>
#include <Logger.h>
#include "Problem.h"
#include "Identity.h"
#include "Scaling.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE("Testing Problem without regularization")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("some data term")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 17, 23;
        VolumeDescriptor dd(numCoeff);

        RealVector_t scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        RealVector_t dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer dcData(dd, dataVec);

        LinearResidual linRes(scaleOp, dcData);
        L2NormPow2 func(linRes);

        WHEN("setting up the problem without x0")
        {
            Problem prob(func);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer dcZero(dd);
                dcZero = 0;
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == Approx(0.5 * dataVec.squaredNorm()));
                REQUIRE(prob.getGradient() == -1.0f * dcScaling * dcData);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i] == Approx(scaling[i] * scaling[i] * dataVec[i]));

                REQUIRE_UNARY(checkApproxEq(prob.getLipschitzConstant(100), 1.0f));
            }
        }

        WHEN("setting up the problem with x0")
        {
            RealVector_t x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer dcX0(dd, x0Vec);

            Problem prob(func, dcX0);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == dcX0);

                REQUIRE(prob.evaluate()
                        == Approx(0.5
                                  * (scaling.array() * x0Vec.array() - dataVec.array())
                                        .matrix()
                                        .squaredNorm()));

                DataContainer gradientDirect = dcScaling * (dcScaling * dcX0 - dcData);
                auto gradient = prob.getGradient();
                for (index_t i = 0; i < gradientDirect.getSize(); ++i)
                    REQUIRE(gradient[i] == Approx(gradientDirect[i]));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i] == Approx(scaling[i] * scaling[i] * dataVec[i]));

                REQUIRE_UNARY(checkApproxEq(prob.getLipschitzConstant(100), 1.0f));
            }
        }
    }
}

SCENARIO("Testing Problem with one regularization term")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("some data term and some regularization term")
    {
        // least squares data term
        IndexVector_t numCoeff(2);
        numCoeff << 23, 47;
        VolumeDescriptor dd(numCoeff);

        RealVector_t scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        RealVector_t dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer dcData(dd, dataVec);

        LinearResidual linRes(scaleOp, dcData);
        L2NormPow2 func(linRes);

        // l2 norm regularization term
        L2NormPow2 regFunc(dd);
        real_t weight = 2.0;
        RegularizationTerm regTerm(weight, regFunc);

        WHEN("setting up the problem without x0")
        {
            Problem prob(func, regTerm);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer dcZero(dd);
                dcZero = 0;
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == Approx(0.5 * dataVec.squaredNorm()));
                REQUIRE(prob.getGradient() == -1.0f * dcScaling * dcData);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight * dataVec[i]));

                // REQUIRE(prob.getLipschitzConstant(100) == Approx(1.0f + weight).margin(0.05));
                REQUIRE_UNARY(checkApproxEq(prob.getLipschitzConstant(100), 1.0 + weight));
            }
        }

        WHEN("setting up the problem with x0")
        {
            RealVector_t x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer dcX0(dd, x0Vec);

            Problem prob(func, regTerm, dcX0);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == dcX0);

                auto valueData =
                    0.5
                    * (scaling.array() * x0Vec.array() - dataVec.array()).matrix().squaredNorm();
                REQUIRE(prob.evaluate() == Approx(valueData + weight * 0.5 * x0Vec.squaredNorm()));

                DataContainer gradientDirect =
                    dcScaling * (dcScaling * dcX0 - dcData) + weight * dcX0;
                auto gradient = prob.getGradient();
                for (index_t i = 0; i < gradient.getSize(); ++i)
                    REQUIRE(gradient[i] == Approx(gradientDirect[i]));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight * dataVec[i]));

                // REQUIRE(prob.getLipschitzConstant(100) == Approx(1.0f + weight).margin(0.05));
                REQUIRE_UNARY(checkApproxEq(prob.getLipschitzConstant(100), 1.0 + weight));
            }
        }
    }
}

SCENARIO("Testing Problem with several regularization terms")
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("some data term and several regularization terms")
    {
        // least squares data term
        IndexVector_t numCoeff(3);
        numCoeff << 17, 33, 52;
        VolumeDescriptor dd(numCoeff);

        RealVector_t scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        RealVector_t dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer dcData(dd, dataVec);

        LinearResidual linRes(scaleOp, dcData);
        L2NormPow2 func(linRes);

        // l2 norm regularization term
        L2NormPow2 regFunc(dd);
        real_t weight1 = 2.0;
        RegularizationTerm regTerm1(weight1, regFunc);

        real_t weight2 = 3.0;
        RegularizationTerm regTerm2(weight2, regFunc);

        std::vector<RegularizationTerm<real_t>> vecReg{regTerm1, regTerm2};

        WHEN("setting up the problem without x0")
        {
            Problem prob(func, vecReg);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer dcZero(dd);
                dcZero = 0;
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == Approx(0.5 * dataVec.squaredNorm()));
                REQUIRE(prob.getGradient() == -1.0f * dcScaling * dcData);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight1 * dataVec[i]
                                      + weight2 * dataVec[i]));

                // REQUIRE(prob.getLipschitzConstant(100)
                //         == Approx(1.0f + weight1 + weight2).margin(0.05));
                REQUIRE_UNARY(
                    checkApproxEq(prob.getLipschitzConstant(100), 1.0 + weight1 + weight2));
            }
        }

        WHEN("setting up the problem with x0")
        {
            RealVector_t x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer dcX0(dd, x0Vec);

            Problem prob(func, vecReg, dcX0);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == dcX0);

                auto valueData =
                    0.5
                    * (scaling.array() * x0Vec.array() - dataVec.array()).matrix().squaredNorm();
                REQUIRE(prob.evaluate()
                        == Approx(valueData + weight1 * 0.5 * x0Vec.squaredNorm()
                                  + weight2 * 0.5 * x0Vec.squaredNorm()));

                auto gradient = prob.getGradient();
                DataContainer gradientDirect =
                    dcScaling * (dcScaling * dcX0 - dcData) + weight1 * dcX0 + weight2 * dcX0;
                for (index_t i = 0; i < gradient.getSize(); ++i)
                    REQUIRE(checkApproxEq(gradient[i], gradientDirect[i]));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]
                                                         + weight1 * dataVec[i]
                                                         + weight2 * dataVec[i]));

                // REQUIRE(prob.getLipschitzConstant(100)
                //         == Approx(1.0f + weight1 + weight2).margin(0.05));
                REQUIRE_UNARY(
                    checkApproxEq(prob.getLipschitzConstant(100), 1.0 + weight1 + weight2));
            }
        }
    }
}

TEST_SUITE_END();
