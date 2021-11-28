/**
 * @file test_ADMM.cpp
 *
 * @brief Tests for the ADMM class
 *
 * @author Andi Braimllari
 */

#include "doctest/doctest.h"

#include "ADMM.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "HardThresholding.h"
#include "FISTA.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("ADMM: Solving problems", TestType, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 21, 11;
        VolumeDescriptor volDescr(numCoeff);

        Vector_t<TestType> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<TestType> dcB(volDescr, bVec);

        Identity<TestType> idOp(volDescr);
        Scaling<TestType> negativeIdOp(volDescr, -1);
        DataContainer<TestType> dCC(volDescr);
        dCC = 0;

        WLSProblem<TestType> wlsProb(idOp, dcB);

        Constraint<TestType> constraint(idOp, negativeIdOp, dCC);

        WHEN("setting up ADMM and FISTA to solve a LASSOProblem")
        {
            L1Norm<TestType> regFunc(volDescr);
            RegularizationTerm<TestType> regTerm(0.000001f, regFunc);

            SplittingProblem<TestType> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CG, SoftThresholding, TestType> admm(splittingProblem, 5, 1e-5f, 1e-5f, true,
                                                      false);

            LASSOProblem<TestType> lassoProb(wlsProb, regTerm);
            FISTA<TestType> fista(lassoProb);

            THEN("the solutions match")
            {
                REQUIRE_UNARY(isApprox(admm.solve(100), fista.solve(100)));
            }
        }

        WHEN("setting up ADMM to solve a WLSProblem + L0PseudoNorm")
        {
            L0PseudoNorm<TestType> regFunc(volDescr);
            RegularizationTerm<TestType> regTerm(0.000001f, regFunc);

            SplittingProblem<TestType> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CG, HardThresholding, TestType> admm(splittingProblem, 5, 1e-5f, 1e-5f, true,
                                                      false);

            THEN("the solution doesn't throw, is not nan and is approximate to the b vector")
            {
                REQUIRE_NOTHROW(admm.solve(10));
                REQUIRE_UNARY(!std::isnan(admm.solve(20).squaredL2Norm()));
                REQUIRE_UNARY(isApprox(admm.solve(20), dcB));
            }
        }

        WHEN("running unsqueeze")
        {
            L0PseudoNorm<TestType> regFunc(volDescr);
            RegularizationTerm<TestType> regTerm(0.000001f, regFunc);

            SplittingProblem<TestType> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CG, HardThresholding, TestType> admm(splittingProblem);

            VolumeDescriptor otherVolDescr({2, 4});

            DataContainer<TestType> dc(otherVolDescr);
            dc[0] = 5;
            dc[1] = 9;
            dc[2] = 2;
            dc[3] = 2;
            dc[4] = 8;
            dc[5] = 1;
            dc[6] = 0;
            dc[7] = 4;
            DataContainer<TestType> resDC = admm.unsqueezeLastDimension(dc);

            VolumeDescriptor unsqueezedVolDescr({2, 4, 1});
            THEN("data descriptor of the result matches the expected the data descriptor")
            {
                REQUIRE_EQ(resDC.getDataDescriptor(), unsqueezedVolDescr);
            }

            THEN("the contents are the same")
            {
                for (index_t i = 0; i < dc.getSize(); ++i) {
                    REQUIRE_EQ(dc[i], resDC[i]);
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE("ADMM: Solving problems with solutions restricted to only positive values",
                   TestType, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        index_t n = 32;
        IndexVector_t size(2);
        size << n, n;
        VolumeDescriptor volDescr(size);

        ShearletTransform<TestType, TestType> shearletTransform(size);
        index_t layers = shearletTransform.getNumOfLayers();

        IndexVector_t slicesInBlock(2);
        slicesInBlock << layers, 1;

        WHEN("setting up ADMM to solve a problem")
        {
            Vector_t<TestType> bVec(VolumeDescriptor{n, n}.getNumberOfCoefficients());
            bVec.setRandom();
            DataContainer<TestType> dcB(VolumeDescriptor{n, n}, bVec);

            Identity<TestType> idOp(VolumeDescriptor{n, n});

            WLSProblem<TestType> wlsProb(idOp, dcB);

            DataContainer<TestType> wL1NWeights(VolumeDescriptor{{n, n, layers}});
            wL1NWeights = 0.001f;
            WeightedL1Norm<TestType> weightedL1Norm(LinearResidual<TestType>{shearletTransform},
                                                    wL1NWeights);
            RegularizationTerm<TestType> wL1NormRegTerm(1, weightedL1Norm);

            SplittingProblem<TestType> splittingProblem(wlsProb.getDataTerm(), wL1NormRegTerm,
                                                        VolumeDescriptor{{n, n, layers + 1}});

            ADMM<CG, SoftThresholding, TestType> admm(splittingProblem, true);

            THEN("the solution doesn't throw, is not nan, and is approximate to the bvector")
            {
                REQUIRE_NOTHROW(admm.solve(1));
                REQUIRE_UNARY(!std::isnan(admm.solve(2).squaredL2Norm()));
            }
        }
    }
}

TEST_SUITE_END();
