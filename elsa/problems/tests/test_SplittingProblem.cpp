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

TEST_CASE_TEMPLATE("SplittingProblem: Simple Test", data_t, float, complex<float>, double,
                   complex<double>)
{
    GIVEN("some data terms, a regularization term and a constraint")
    {
        VolumeDescriptor dd({91, 32});

        Vector_t<data_t> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<data_t> dcScaling(dd, scaling);
        Scaling<data_t> scaleOp(dd, dcScaling);

        Vector_t<data_t> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<data_t> dcData(dd, dataVec);

        WLSProblem<data_t> wlsProblem(scaleOp, dcData);

        Identity<data_t> A(dd);
        Scaling<data_t> B(dd, -1);
        DataContainer<data_t> c(dd);
        c = 0;
        Constraint<data_t> constraint(A, B, c);

        WHEN("setting up a SplittingProblem")
        {
            L1Norm<data_t> l1NormRegFunc(dd);
            RegularizationTerm<data_t> l1NormRegTerm(1 / 2, l1NormRegFunc);

            SplittingProblem<data_t> splittingProblem(wlsProblem.getDataTerm(), l1NormRegTerm,
                                                      constraint);

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
