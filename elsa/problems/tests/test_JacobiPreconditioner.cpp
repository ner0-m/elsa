/**
 * @file test_JacobiPreconditioner.cpp
 *
 * @brief Tests for the JacobiPreconditioner class
 *
 * @author Michael Loipführer - initial code
 */

#include <doctest/doctest.h>

#include <VolumeDescriptor.h>
#include "Logger.h"
#include "JacobiPreconditioner.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TYPE_TO_STRING(JacobiPreconditioner<float>);
TYPE_TO_STRING(JacobiPreconditioner<double>);

TEST_CASE_TEMPLATE("JacobiPreconditioner: Testing standard use cases", data_t, float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // no log spamming in tests
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a simple linear operator")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 24;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<data_t, -1, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        bVec = bVec.cwiseAbs();
        Scaling<data_t> scalingOp{dd, DataContainer<data_t>{dd, bVec}};

        WHEN("setting up a Jacobi Preconditioner")
        {
            JacobiPreconditioner<data_t> preconditioner{scalingOp, false};

            THEN("the clone works correctly")
            {
                auto preconditionerClone = preconditioner.clone();

                REQUIRE_NE(preconditionerClone.get(), &preconditioner);
                REQUIRE_EQ(*preconditionerClone, preconditioner);
            }

            THEN("the preconditioner actually represents the diagonal of the operator")
            {

                DataContainer<data_t> e(scalingOp.getDomainDescriptor());
                e = 0;
                DataContainer<data_t> diag(scalingOp.getDomainDescriptor());
                for (index_t i = 0; i < e.getSize(); i++) {
                    e[i] = 1;
                    REQUIRE_UNARY(checkApproxEq(preconditioner.apply(e), scalingOp.apply(e)));
                    e[i] = 0;
                }
            }
        }
    }
}

TEST_SUITE_END();
