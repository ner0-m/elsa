/**
 * @file test_TikhonovProblem.cpp
 *
 * @brief Tests for the TikhonovProblem class
 *
 * @author Nikola Dinev
 */

#include "doctest/doctest.h"

#include "Problem.h"
#include "Identity.h"
#include "Scaling.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include "L1Norm.h"
#include "TikhonovProblem.h"
#include "VolumeDescriptor.h"
#include "Logger.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("TikhonovProblem: Testing with one regularization term", data_t, float, double)
{
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("some data term and some regularization term")
    {
        // least squares data term
        VolumeDescriptor dd({23, 47});

        Vector_t<data_t> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<data_t> dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        Vector_t<data_t> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<data_t> dcData(dd, dataVec);

        WLSProblem<data_t> wls(scaleOp, dcData);

        // l2 norm regularization term
        L2NormPow2<data_t> regFunc(dd);
        auto weight = data_t{2.0};
        RegularizationTerm<data_t> regTerm(weight, regFunc);

        WHEN("setting up a TikhonovProblem without regularization terms")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(
                    TikhonovProblem<data_t>(wls, std::vector<RegularizationTerm<data_t>>{}),
                    InvalidArgumentError);
            }
        }

        WHEN("setting up a TikhonovProblem with a non (Weighted)L2NormPow2 regularization term")
        {
            L1Norm<data_t> invalidRegFunc(dd);
            RegularizationTerm<data_t> invalidRegTerm(1.0, invalidRegFunc);
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(TikhonovProblem<data_t>(wls, invalidRegTerm),
                                  InvalidArgumentError);
            }
        }

        WHEN("setting up the TikhonovProblem without x0")
        {
            TikhonovProblem<data_t> prob(wls, regTerm);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the TikhonovProblem behaves as expected")
            {
                DataContainer<data_t> dcZero(dd);
                dcZero = 0;
                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcZero));

                REQUIRE_UNARY(
                    checkApproxEq(prob.evaluate(), as<data_t>(0.5) * dataVec.squaredNorm()));
                REQUIRE_UNARY(isApprox(prob.getGradient(), as<data_t>(-1.0) * dcScaling * dcData));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]
                                                               + weight * dataVec[i]));
            }

            THEN("the TikhonovProblem is different from a Problem with the same terms")
            {
                Problem optProb(prob.getDataTerm(), prob.getRegularizationTerms());
                REQUIRE_NE(prob, optProb);
                REQUIRE_NE(optProb, prob);
            }
        }

        WHEN("setting up the TikhonovProblem with x0")
        {
            Vector_t<data_t> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<data_t> dcX0(dd, x0Vec);

            wls.getCurrentSolution() = dcX0;
            TikhonovProblem<data_t> prob(wls, regTerm);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the TikhonovProblem behaves as expected")
            {
                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcX0));

                auto valueData =
                    as<data_t>(0.5)
                    * (scaling.array() * x0Vec.array() - dataVec.array()).matrix().squaredNorm();
                REQUIRE_UNARY(checkApproxEq(
                    prob.evaluate(), valueData + weight * as<data_t>(0.5) * x0Vec.squaredNorm()));

                auto gradient = prob.getGradient();
                DataContainer gradientDirect =
                    dcScaling * (dcScaling * dcX0 - dcData) + weight * dcX0;

                for (index_t i = 0; i < gradient.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(gradient[i], gradientDirect[i]));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]
                                                               + weight * dataVec[i]));
            }

            THEN("the TikhonovProblem is different from a Problem with the same terms")
            {
                Problem optProb(prob.getDataTerm(), prob.getRegularizationTerms());
                REQUIRE_NE(prob, optProb);
                REQUIRE_NE(optProb, prob);
            }
        }
    }
}

TEST_CASE_TEMPLATE("TikhonovProblem: Testing with several regularization terms", data_t, float,
                   double)
{
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("some data term and several regularization terms")
    {
        // least squares data term
        VolumeDescriptor dd({17, 33, 52});

        Vector_t<data_t> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<data_t> dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        Vector_t<data_t> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<data_t> dcData(dd, dataVec);

        WLSProblem<data_t> wls(scaleOp, dcData);

        // l2 norm regularization term
        L2NormPow2<data_t> regFunc(dd);
        auto weight1 = data_t{2.0};
        RegularizationTerm<data_t> regTerm1(weight1, regFunc);

        auto weight2 = data_t{3.0};
        RegularizationTerm<data_t> regTerm2(weight2, regFunc);

        std::vector<RegularizationTerm<data_t>> vecReg{regTerm1, regTerm2};

        WHEN("setting up a TikhonovProblem with a non (Weighted)L2NormPow2 regularization term")
        {
            L1Norm<data_t> invalidRegFunc(dd);
            RegularizationTerm<data_t> invalidRegTerm(1.0, invalidRegFunc);
            std::vector<RegularizationTerm<data_t>> invalidVecReg1{regTerm1, invalidRegTerm};
            std::vector<RegularizationTerm<data_t>> invalidVecReg2{invalidRegTerm, regTerm2};
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(TikhonovProblem<data_t>(wls, invalidVecReg1),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(TikhonovProblem<data_t>(wls, invalidVecReg2),
                                  InvalidArgumentError);
            }
        }

        WHEN("setting up the TikhonovProblem without x0")
        {
            TikhonovProblem<data_t> prob(wls, vecReg);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the TikhonovProblem behaves as expected")
            {
                DataContainer<data_t> dcZero(dd);
                dcZero = 0;
                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcZero));

                REQUIRE_UNARY(
                    checkApproxEq(prob.evaluate(), as<data_t>(0.5) * dataVec.squaredNorm()));
                REQUIRE_UNARY(isApprox(prob.getGradient(), as<data_t>(-1.0) * dcScaling * dcData));

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(result[i], scaling[i] * scaling[i] * dataVec[i]
                                                               + weight1 * dataVec[i]
                                                               + weight2 * dataVec[i]));
            }

            THEN("the TikhonovProblem is different from a Problem with the same terms")
            {
                Problem optProb(prob.getDataTerm(), prob.getRegularizationTerms());
                REQUIRE_NE(prob, optProb);
                REQUIRE_NE(optProb, prob);
            }
        }

        WHEN("setting up the TikhonovProblem with x0")
        {
            Vector_t<data_t> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<data_t> dcX0(dd, x0Vec);

            wls.getCurrentSolution() = dcX0;
            TikhonovProblem<data_t> prob(wls, vecReg);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the TikhonovProblem behaves as expected")
            {
                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcX0));

                auto valueData =
                    as<data_t>(0.5)
                    * (scaling.array() * x0Vec.array() - dataVec.array()).matrix().squaredNorm();
                REQUIRE_UNARY(checkApproxEq(
                    prob.evaluate(), valueData + weight1 * as<data_t>(0.5) * x0Vec.squaredNorm()
                                         + weight2 * as<data_t>(0.5) * x0Vec.squaredNorm()));

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
            }

            THEN("the TikhonovProblem is different from a Problem with the same terms")
            {
                Problem optProb(prob.getDataTerm(), prob.getRegularizationTerms());
                REQUIRE_NE(prob, optProb);
                REQUIRE_NE(optProb, prob);
            }
        }
    }
}

TEST_SUITE_END();
