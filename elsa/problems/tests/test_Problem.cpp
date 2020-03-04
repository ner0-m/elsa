/**
 * \file test_Problem.cpp
 *
 * \brief Tests for the Problem class
 *
 * \author David Frank - initial code
 * \author Tobias Lasser - rewrite
 */

#include <catch2/catch.hpp>
#include "Problem.h"
#include "Identity.h"
#include "Scaling.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"

using namespace elsa;

SCENARIO("Testing Problem without regularization")
{
    GIVEN("some data term")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 17, 23;
        DataDescriptor dd(numCoeff);

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
                REQUIRE(prob.getGradient() == dcScaling * (dcScaling * dcX0 - dcData));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i] == Approx(scaling[i] * scaling[i] * dataVec[i]));
            }
        }
    }
}

SCENARIO("Testing Problem with one regularization term")
{
    GIVEN("some data term and some regularization term")
    {
        // least squares data term
        IndexVector_t numCoeff(2);
        numCoeff << 23, 47;
        DataDescriptor dd(numCoeff);

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
                REQUIRE(prob.getGradient()
                        == dcScaling * (dcScaling * dcX0 - dcData) + weight * dcX0);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight * dataVec[i]));
            }
        }
    }
}

SCENARIO("Testing Problem with several regularization terms")
{
    GIVEN("some data term and several regularization terms")
    {
        // least squares data term
        IndexVector_t numCoeff(3);
        numCoeff << 17, 33, 52;
        DataDescriptor dd(numCoeff);

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
                REQUIRE(prob.getGradient()
                        == dcScaling * (dcScaling * dcX0 - dcData) + weight1 * dcX0
                               + weight2 * dcX0);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight1 * dataVec[i]
                                      + weight2 * dataVec[i]));
            }
        }
    }
}