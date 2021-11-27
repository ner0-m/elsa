/**
 * @file test_SplittingProblem.cpp
 *
 * @brief Tests for the SplittingProblem class
 *
 * @author Andi Braimllari
 */

#include "doctest/doctest.h"

#include "SplittingProblem.h"
#include "L1Norm.h"
#include "VolumeDescriptor.h"
#include "Identity.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("SplittingProblem: Simple Test", data_t, float, std::complex<float>, double,
                   std::complex<double>)
{
    GIVEN("some data terms, a regularization term and a constraint")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 91, 32;
        VolumeDescriptor dd(numCoeff);

        Vector_t<data_t> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<data_t> dcScaling(dd, scaling);
        Scaling<data_t> scaleOp(dd, dcScaling);

        Vector_t<data_t> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<data_t> dcData(dd, dataVec);

        WLSProblem<data_t> wlsProblem(scaleOp, dcData);

        WHEN("setting up a SplittingProblem")
        {
            L1Norm<data_t> l1NormRegFunc(dd);
            RegularizationTerm<data_t> l1NormRegTerm(1.0 / 2, l1NormRegFunc);

            SplittingProblem<data_t> splittingProblem(wlsProblem.getDataTerm(), l1NormRegTerm);

            THEN("cloned SplittingProblem equals original SplittingProblem")
            {

                auto splittingProblemClone = splittingProblem.clone();

                REQUIRE_NE(splittingProblemClone.get(), &splittingProblem);
                REQUIRE_EQ(*splittingProblemClone, splittingProblem);
            }

            THEN("evaluating SplittingProblem throws an exception as it is not yet supported")
            {
                REQUIRE_THROWS_AS(splittingProblem.evaluate(), std::runtime_error);
            }

            THEN("calculating the gradient, Hessian and Lipschitz constant from SplittingProblem "
                 "throws an exception as it is not yet supported")
            {
                REQUIRE_THROWS_AS(splittingProblem.getGradient(), std::runtime_error);
                REQUIRE_THROWS_AS(splittingProblem.getHessian(), std::runtime_error);
                REQUIRE_THROWS_AS(splittingProblem.getLipschitzConstant(), std::runtime_error);
            }
        }
    }
}

TEST_SUITE_END();
