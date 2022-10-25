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
        IndexVector_t numCoeff(2);
        numCoeff << 23, 47;
        VolumeDescriptor dd(numCoeff);

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

        WHEN("setting up the TikhonovProblem")
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

                REQUIRE_UNARY(
                    checkApproxEq(prob.evaluate(dcZero), as<data_t>(0.5) * dataVec.squaredNorm()));
                REQUIRE_UNARY(
                    isApprox(prob.getGradient(dcZero), as<data_t>(-1.0) * dcScaling * dcData));

                auto hessian = prob.getHessian(dcZero);
                auto result = hessian->apply(dcData);
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
        IndexVector_t numCoeff(3);
        numCoeff << 17, 33, 52;
        VolumeDescriptor dd(numCoeff);

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

        WHEN("setting up the TikhonovProblem")
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

                REQUIRE_UNARY(
                    checkApproxEq(prob.evaluate(dcZero), as<data_t>(0.5) * dataVec.squaredNorm()));
                REQUIRE_UNARY(
                    isApprox(prob.getGradient(dcZero), as<data_t>(-1.0) * dcScaling * dcData));

                auto hessian = prob.getHessian(dcZero);
                auto result = hessian->apply(dcData);
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
