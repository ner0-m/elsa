/**
 * @file test_JacobiPreconditioner.cpp
 *
 * @brief Tests for the JacobiPreconditioner class
 *
 * @author Michael Loipführer - initial code
 */

#include <catch2/catch.hpp>
#include <VolumeDescriptor.h>
#include "Logger.h"
#include "JacobiPreconditioner.h"

using namespace elsa;

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TEMPLATE_TEST_CASE("Scenario: Testing JacobiPreconditioner", "", JacobiPreconditioner<float>,
                   JacobiPreconditioner<double>)
{
    using data_t = decltype(return_data_t(std::declval<TestType>()));
    // no log spamming in tests
    Logger::setLevel(Logger::LogLevel::WARN);

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
            TestType preconditioner{scalingOp, false};

            THEN("the clone works correctly")
            {
                auto preconditionerClone = preconditioner.clone();

                REQUIRE(preconditionerClone.get() != &preconditioner);
                REQUIRE(*preconditionerClone == preconditioner);
            }

            THEN("the preconditioner actually represents the diagonal of the operator")
            {

                DataContainer<data_t> e(scalingOp.getDomainDescriptor());
                e = 0;
                DataContainer<data_t> diag(scalingOp.getDomainDescriptor());
                for (index_t i = 0; i < e.getSize(); i++) {
                    e[i] = 1;
                    REQUIRE(preconditioner.apply(e) == scalingOp.apply(e));
                    e[i] = 0;
                }
            }
        }
    }
}