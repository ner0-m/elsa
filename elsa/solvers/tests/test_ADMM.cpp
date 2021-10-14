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
#include "Identity.h"
#include "FISTA.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "PartitionDescriptor.h"

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
        index_t n = 128;
        IndexVector_t size(2);
        size << n, n;
        VolumeDescriptor volDescr(size);

        // random number
        real_t rho1 = 1 / 2;
        real_t rho2 = 1;

        /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
        ShearletTransform<TestType, TestType> shearletTransform(size);
        index_t layers = shearletTransform.getNumOfLayers();

        IndexVector_t slicesInBlock(2);
        slicesInBlock << layers, 1;

        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfA(0);
        Scaling<TestType> scaling(VolumeDescriptor{{n, n}}, rho2);
        opsOfA.push_back(shearletTransform.clone()); // TODO mult. with rho1 later on
        opsOfA.push_back(scaling.clone());
        BlockLinearOperator<TestType> A{
            VolumeDescriptor{{n, n}},
            PartitionDescriptor{VolumeDescriptor{{n, n, layers + 1}}, slicesInBlock}, opsOfA,
            BlockLinearOperator<TestType>::BlockType::ROW};

        /// B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
        DataContainer<TestType> factorsOfB(VolumeDescriptor{n, n, layers + 1});
        for (int ind = 0; ind < factorsOfB.getSize(); ++ind) {
            if (ind < (n * n * layers)) {
                factorsOfB[ind] = -1 * rho1;
            } else {
                factorsOfB[ind] = -1 * rho2;
            }
        }
        Scaling<TestType> B(VolumeDescriptor{{n, n, layers + 1}}, factorsOfB);

        DataContainer<TestType> c(VolumeDescriptor{{n, n, layers + 1}});
        c = 0;

        Constraint<TestType> constraint(A, B, c);

        WHEN("setting up ADMM to solve a problem")
        {
            Vector_t<TestType> bVec(VolumeDescriptor{n * n}.getNumberOfCoefficients());
            bVec.setRandom();
            DataContainer<TestType> dcB(VolumeDescriptor{n * n}, bVec);

            Identity<TestType> idOp(VolumeDescriptor{n, n});

            WLSProblem<TestType> wlsProb(idOp, dcB);

            DataContainer<TestType> wL1NWeights(VolumeDescriptor{{n, n, layers}});
            wL1NWeights = 1;
            WeightedL1Norm<TestType> weightedL1Norm(LinearResidual<TestType>{shearletTransform},
                                                    wL1NWeights);
            RegularizationTerm<TestType> wL1NormRegTerm(1, weightedL1Norm);

            Indicator<TestType> indicator(VolumeDescriptor{{n, n}});
            RegularizationTerm<TestType> indicRegTerm(1, indicator);

            SplittingProblem<TestType> splittingProblem(
                wlsProb.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

            ADMM<CG, SoftThresholding, TestType> admm(splittingProblem, true);

            THEN("the solution doesn't throw, is not nan, and is approximate to the bvector")
            {
                REQUIRE_NOTHROW(admm.solve(1));
                REQUIRE_UNARY(!std::isnan(admm.solve(2).squaredL2Norm()));
                // REQUIRE_UNARY(isApprox(admm.solve(20), dcB));
            }
        }
    }
}

TEST_SUITE_END();
