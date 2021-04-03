/**
 * @file test_SplittingProblem.cpp
 *
 * @brief Tests for the SplittingProblem class
 *
 * @author Andi Braimllari
 */

#include "SplittingProblem.h"
#include "L1Norm.h"
#include "VolumeDescriptor.h"
#include "Identity.h"

#include <catch2/catch.hpp>

using namespace elsa;

TEMPLATE_TEST_CASE("Scenario: Testing SplittingProblem", "", float, std::complex<float>, double,
                   std::complex<double>)
{
    GIVEN("some data terms, a regularization term and a constraint")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 91, 32;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<TestType> dcScaling(dd, scaling);
        Scaling<TestType> scaleOp(dd, dcScaling);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<TestType> dcData(dd, dataVec);

        WLSProblem<TestType> wlsProblem(scaleOp, dcData);

        Identity<TestType> A(dd);
        Scaling<TestType> B(dd, -1);
        DataContainer<TestType> c(dd);
        c = 0;
        Constraint<TestType> constraint(A, B, c);

        WHEN("setting up a SplittingProblem")
        {
            L1Norm<TestType> l1NormRegFunc(dd);
            RegularizationTerm<TestType> l1NormRegTerm(1 / 2, l1NormRegFunc);

            SplittingProblem<TestType> splittingProblem(wlsProblem.getDataTerm(), l1NormRegTerm,
                                                        constraint);

            THEN("cloned SplittingProblem equals original SplittingProblem")
            {

                auto splittingProblemClone = splittingProblem.clone();

                REQUIRE(splittingProblemClone.get() != &splittingProblem);
                REQUIRE(*splittingProblemClone == splittingProblem);
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