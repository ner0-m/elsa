/**
 * @file test_SHADMM.cpp
 *
 * @brief Tests for the SHADMM class
 *
 * @author Andi Braimllari
 */

#include "doctest/doctest.h"

#include "SHADMM.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "Identity.h"
#include "FISTA.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "BlockLinearOperator.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("SHADMM: Solving problems", data_t, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        IndexVector_t size(2);
        size << 7, 7;
        VolumeDescriptor volDescr(size);

        Vector_t<data_t> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB(volDescr, bVec);

        Identity<data_t> idOp(volDescr);

        WLSProblem<data_t> wlsProb(idOp, dcB);

        // random number
        data_t rho1 = 23;
        // random number
        data_t rho2 = 86;

        ShearletTransform<data_t> shearletTransform(rho1, size[0], size[1]);
        index_t L = shearletTransorm.getL();
        // size[0] == size[1], for now at least
        index_t n = shearletTransform.getWidth();

        VolumeDescriptor volDescrOfn2({n * n});
        VolumeDescriptor volDescrOfLn2({L * n * n});
        VolumeDescriptor volDescrOfLp1n2({(L + 1) * n * n});

        /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
        std::vector<std::unique_ptr<LinearOperator<data_t>>> opsOfA(2);
        DataContainer<data_t> rho2s(volDescrOfn2);
        rho2s = rho2;
        Scaling<data_t> scaling1(volDescrOfn2, rho2s);
        opsOfA.push_back(std::move(shearletTransorm.clone()));
        opsOfA.push_back(std::move(scaling1.clone()));
        BlockLinearOperator<data_t> A{opsOfA, BlockLinearOperator<data_t>::BlockType::COL};

        /// B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
        std::vector<std::unique_ptr<LinearOperator<data_t>>> opsOfB(2);
        DataContainer<data_t> nRho1s(volDescrOfLn2);
        nRho1s = -rho1;
        Scaling<data_t> scaling2(volDescrOfLn2, nRho1s);
        DataContainer<data_t> nRho2s(volDescrOfn2);
        nRho2s = -rho2;
        Scaling<data_t> scaling3(volDescrOfn2, nRho2s);
        opsOfB.push_back(std::move(scaling2.clone()));
        opsOfB.push_back(std::move(scaling3.clone()));
        BlockLinearOperator<data_t> B{opsOfB, BlockLinearOperator<data_t>::BlockType::COL};

        DataContainer<data_t> dCC(volDescr);
        dCC = 0;

        Constraint<data_t> constraint(A, B, dCC);

        WHEN("setting up SHADMM to solve a problem")
        {
            DataContainer<data_t> ones(volDescr);
            ones = 1;
            WeightedL1Norm weightedL1Norm(ones);
            RegularizationTerm<data_t> wL1NormRegTerm(1, weightedL1Norm);

            Indicator<data_t> indicator(volDescr);
            RegularizationTerm<data_t> indicRegTerm(1, indicator);

            SplittingProblem<data_t> splittingProblem(
                wlsProb.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

            SHADMM<CG, data_t> shadmm(splittingProblem);

            THEN("the solution doesn't throw, is not nan and is approximate to the bvector")
            {
                REQUIRE_NOTHROW(shadmm.solve(10));
                printf("%f\n", shadmm.solve(20).squaredL2Norm());
                REQUIRE_UNARY(!std::isnan(shadmm.solve(20).squaredL2Norm()));
                REQUIRE_UNARY(isApprox(shadmm.solve(20), dcB));
            }
        }
    }
}

TEST_SUITE_END();
