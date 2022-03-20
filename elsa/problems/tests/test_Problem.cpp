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
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE("Problem: Testing without regularization")
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

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer dcZero(dd);
                dcZero = 0;
                REQUIRE_EQ(prob.getCurrentSolution(), dcZero);

                REQUIRE_UNARY(checkApproxEq(prob.evaluate(), 0.5f * dataVec.squaredNorm()));
                REQUIRE_UNARY(checkApproxEq(prob.getGradient(), -1.0f * dcScaling * dcData));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]));

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

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE_EQ(prob.getCurrentSolution(), dcX0);

                REQUIRE_UNARY(checkApproxEq(
                    prob.evaluate(), 0.5f
                                         * (scaling.array() * x0Vec.array() - dataVec.array())
                                               .matrix()
                                               .squaredNorm()));

                DataContainer gradientDirect = dcScaling * (dcScaling * dcX0 - dcData);
                auto gradient = prob.getGradient();
                for (index_t i = 0; i < gradientDirect.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(gradient[i], gradientDirect[i]));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]));

                REQUIRE_UNARY(checkApproxEq(prob.getLipschitzConstant(100), 1.0f));
            }
        }
    }
}

TEST_CASE("Problem: Testing with one regularization term")
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

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer dcZero(dd);
                dcZero = 0;
                REQUIRE_UNARY(checkApproxEq(prob.getCurrentSolution(), dcZero));

                REQUIRE_UNARY(checkApproxEq(prob.evaluate(), 0.5f * dataVec.squaredNorm()));
                REQUIRE_UNARY(checkApproxEq(prob.getGradient(), -1.0f * dcScaling * dcData));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]
                                                               + weight * dataVec[i]));

                REQUIRE_UNARY(checkApproxEq(prob.getLipschitzConstant(100), 1.0f + weight));
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

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE_EQ(prob.getCurrentSolution(), dcX0);

                auto valueData =
                    0.5f
                    * (scaling.array() * x0Vec.array() - dataVec.array()).matrix().squaredNorm();
                REQUIRE_UNARY(checkApproxEq(prob.evaluate(),
                                            valueData + weight * 0.5f * x0Vec.squaredNorm()));

                DataContainer gradientDirect =
                    dcScaling * (dcScaling * dcX0 - dcData) + weight * dcX0;
                auto gradient = prob.getGradient();
                for (index_t i = 0; i < gradient.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(gradient[i], gradientDirect[i]));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]
                                                               + weight * dataVec[i]));

                REQUIRE_UNARY(checkApproxEq(prob.getLipschitzConstant(100), 1.0f + weight));
            }
        }

        WHEN("given a different data descriptor and another regularization term with a different "
             "domain descriptor")
        {
            // three-dimensional data descriptor
            IndexVector_t otherNumCoeff(3);
            otherNumCoeff << 15, 38, 22;
            VolumeDescriptor otherDD(otherNumCoeff);

            // l2 norm regularization term
            L2NormPow2 otherRegFunc(otherDD);
            RegularizationTerm otherRegTerm(weight, otherRegFunc);

            THEN("no exception is thrown when setting up a problem with different domain "
                 "descriptors")
            {
                REQUIRE_NOTHROW(Problem{func, otherRegTerm});
            }
        }
    }
}

TEST_CASE("Problem: Testing with several regularization terms")
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

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer dcZero(dd);
                dcZero = 0;
                REQUIRE_UNARY(checkApproxEq(prob.getCurrentSolution(), dcZero));

                REQUIRE_UNARY(checkApproxEq(prob.evaluate(), 0.5f * dataVec.squaredNorm()));
                REQUIRE_UNARY(checkApproxEq(prob.getGradient(), -1.0f * dcScaling * dcData));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]
                                                               + weight1 * dataVec[i]
                                                               + weight2 * dataVec[i]));

                REQUIRE_UNARY(
                    checkApproxEq(prob.getLipschitzConstant(100), 1.0f + weight1 + weight2));
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

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcX0));

                auto valueData =
                    0.5f
                    * (scaling.array() * x0Vec.array() - dataVec.array()).matrix().squaredNorm();
                REQUIRE_UNARY(
                    checkApproxEq(prob.evaluate(), valueData + weight1 * 0.5f * x0Vec.squaredNorm()
                                                       + weight2 * 0.5f * x0Vec.squaredNorm()));

                auto gradient = prob.getGradient();
                DataContainer gradientDirect =
                    dcScaling * (dcScaling * dcX0 - dcData) + weight1 * dcX0 + weight2 * dcX0;
                for (index_t i = 0; i < gradient.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(gradient[i], gradientDirect[i]));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]
                                                               + weight1 * dataVec[i]
                                                               + weight2 * dataVec[i]));

                REQUIRE_UNARY(
                    checkApproxEq(prob.getLipschitzConstant(100), 1.0f + weight1 + weight2));
            }
        }

        WHEN("given two different data descriptors and two regularization terms with a different "
             "domain descriptor")
        {
            // three-dimensional data descriptor
            IndexVector_t otherNumCoeff(3);
            otherNumCoeff << 15, 38, 22;
            VolumeDescriptor otherDD(otherNumCoeff);

            // four-dimensional data descriptor
            IndexVector_t anotherNumCoeff(4);
            anotherNumCoeff << 7, 9, 21, 17;
            VolumeDescriptor anotherDD(anotherNumCoeff);

            // l2 norm regularization term
            L2NormPow2 otherRegFunc(otherDD);
            RegularizationTerm otherRegTerm(weight1, otherRegFunc);

            // l2 norm regularization term
            L2NormPow2 anotherRegFunc(anotherDD);
            RegularizationTerm anotherRegTerm(weight2, anotherRegFunc);

            THEN("no exception is thrown when setting up a problem with different domain "
                 "descriptors")
            {
                REQUIRE_NOTHROW(Problem{func, std::vector{otherRegTerm, anotherRegTerm}});
            }
        }
    }
}

TEST_SUITE_END();
