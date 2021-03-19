#include "SplittingProblem.h"
#include "L1Norm.h"
#include "VolumeDescriptor.h"
#include "Identity.h"

#include <catch2/catch.hpp>

using namespace elsa;

TEMPLATE_TEST_CASE("Scenario: Testing SplittingProblem", "", float, double)
{
    GIVEN("some data terms, a regularization term and a constraint")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 17, 53;
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

        WHEN("setting up some SplittingProblem components")
        {

            THEN("cloned SplittingProblem equals original SplittingProblem")
            {
                L1Norm<TestType> l1NormRegFunc(dd);
                RegularizationTerm<TestType> l1NormRegTerm(1 / 2, l1NormRegFunc);

                SplittingProblem<TestType> splittingProblem(wlsProblem.getDataTerm(), l1NormRegTerm,
                                                            constraint);

                auto splittingProblemClone = splittingProblem.clone();

                REQUIRE(splittingProblemClone.get() != &splittingProblem);
                REQUIRE(*splittingProblemClone == splittingProblem);
            }

            THEN("evaluating SplittingProblem throws an exception as it is not yet supported")
            {
                L1Norm<TestType> l1NormRegFunc(dd);
                RegularizationTerm<TestType> l1NormRegTerm(1 / 2, l1NormRegFunc);

                SplittingProblem<TestType> splittingProblem(wlsProblem.getDataTerm(), l1NormRegTerm,
                                                            constraint);

                REQUIRE_THROWS_AS(splittingProblem.evaluate(), std::runtime_error);
            }
        }
    }
}
