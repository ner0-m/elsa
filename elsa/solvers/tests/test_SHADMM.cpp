/**
 * @file test_SHADMM.cpp
 *
 * @brief Tests for the SHADMM class
 *
 * @author Andi Braimllari
 */

#include "doctest/doctest.h"

#include "SHADMM.h"
#include "VolumeDescriptor.h"
#include "BlockLinearOperator.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "Identity.h"
#include "FISTA.h"
#include "Logger.h"

#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("SHADMM: Solving problems", TestType, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        IndexVector_t size(2);
        size << 7, 7;
        VolumeDescriptor volDescr(size);

        Vector_t<TestType> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<TestType> dcB(volDescr, bVec);

        Identity<TestType> idOp(volDescr);

        WLSProblem<TestType> wlsProb(idOp, dcB);

        // random number
        // TODO ShearletTransform<data_t> shearletTransform(rho, size[0], size[1])
        TestType rho1 = 23;
        // random number
        TestType rho2 = 86;

        ShearletTransform<TestType> shearletTransform(size[0], size[1]);
        index_t L = shearletTransform.getL();
        // size[0] == size[1], for now at least
        index_t n = shearletTransform.getWidth();

        //        /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
        //        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfA(0);
        //        DataContainer<TestType> rho2s(volDescrOfn2);
        //        rho2s = rho2;
        //        Scaling<TestType> scaling1(volDescrOfn2, rho2s);
        //        opsOfA.push_back(std::move(shearletTransform.clone()));
        //        opsOfA.push_back(std::move(scaling1.clone()));
        //        BlockLinearOperator<TestType> A{opsOfA,
        //        BlockLinearOperator<TestType>::BlockType::COL};

        /// B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2

        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfB1(0);

        DataContainer<TestType> nRho1s(VolumeDescriptor{{L * n * n, L * n * n}});
        nRho1s = -rho1;
        Scaling<TestType> B11(VolumeDescriptor{{L * n * n, L * n * n}}, nRho1s);
        DataContainer<TestType> zeroes12(VolumeDescriptor{{L * n * n, n * n}});
        zeroes12 = 0;
        Scaling<TestType> B12(VolumeDescriptor{{L * n * n, n * n}}, zeroes12);
        opsOfB1.push_back(std::move(B11.clone()));
        opsOfB1.push_back(std::move(B12.clone()));
        BlockLinearOperator<TestType> B1{opsOfB1, BlockLinearOperator<TestType>::BlockType::ROW};

        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfB2(0);
        DataContainer<TestType> zeroes21(VolumeDescriptor{{n * n, L * n * n}});
        zeroes21 = 0;
        Scaling<TestType> B21(VolumeDescriptor{{n * n, L * n * n}}, zeroes21);
        DataContainer<TestType> nRho2s(VolumeDescriptor{{n * n, n * n}});
        nRho2s = -rho2;
        Scaling<TestType> B22(VolumeDescriptor{{n * n, n * n}}, nRho2s);
        opsOfB2.push_back(std::move(B21.clone()));
        opsOfB2.push_back(std::move(B22.clone()));
        BlockLinearOperator<TestType> B2{opsOfB2, BlockLinearOperator<TestType>::BlockType::COL};

        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfB(0);
        opsOfB1.push_back(std::move(B1.clone()));
        opsOfB1.push_back(std::move(B1.clone()));
        BlockLinearOperator<TestType> B{opsOfB, BlockLinearOperator<TestType>::BlockType::COL};

        //        DataContainer<TestType> dCC(volDescr);
        //        dCC = 0;
        //
        //        Constraint<TestType> constraint(A, B, dCC);
        //
        //        WHEN("setting up SHADMM to solve a problem")
        //        {
        //            DataContainer<TestType> ones(volDescr);
        //            ones = 1;
        //            WeightedL1Norm<TestType> weightedL1Norm(ones);
        //            RegularizationTerm<TestType> wL1NormRegTerm(1, weightedL1Norm);
        //
        //            Indicator<TestType> indicator(volDescr);
        //            RegularizationTerm<TestType> indicRegTerm(1, indicator);
        //
        //            SplittingProblem<TestType> splittingProblem(
        //                wlsProb.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm},
        //                constraint);
        //
        //            SHADMM<CG, TestType> shadmm(splittingProblem);
        //
        //            THEN("the solution doesn't throw, is not nan and is approximate to the
        //            bvector")
        //            {
        //                REQUIRE_NOTHROW(shadmm.solve(10));
        //                printf("%f\n", shadmm.solve(20).squaredL2Norm());
        //                REQUIRE_UNARY(!std::isnan(shadmm.solve(20).squaredL2Norm()));
        //                REQUIRE_UNARY(isApprox(shadmm.solve(20), dcB));
        //            }
        //        }
    }
}

TEST_SUITE_END();
