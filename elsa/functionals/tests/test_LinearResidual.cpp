/**
 * @file test_LinearResidual.cpp
 *
 * @brief Tests for LinearResidual class
 *
 * @author Matthias Wieczorek - main code
 * @author Tobias Lasser - rewrite
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

// mock operator, which outputs 1 for apply and 3 for applyAdjoint
template <typename data_t = real_t>
class MockOperator : public LinearOperator<data_t>
{
public:
    MockOperator(const DataDescriptor& domain, const DataDescriptor& range)
        : LinearOperator<data_t>(domain, range)
    {
    }

protected:
    void applyImpl([[maybe_unused]] const DataContainer<data_t>& x,
                   DataContainer<data_t>& Ax) const override
    {
        Ax = 1;
    }

    void applyAdjointImpl([[maybe_unused]] const DataContainer<data_t>& y,
                          DataContainer<data_t>& Aty) const override
    {
        Aty = 3;
    }

protected:
    MockOperator<data_t>* cloneImpl() const override
    {
        return new MockOperator<data_t>(this->getDomainDescriptor(), this->getRangeDescriptor());
    }
};

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("LinearResidual: Testing trivial linear residual", TestType, float, double,
                   complex<float>, complex<double>)
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 33, 55;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            LinearResidual<TestType> linRes(dd);

            THEN("the residual is as expected")
            {
                REQUIRE_EQ(linRes.getDomainDescriptor(), dd);
                REQUIRE_EQ(linRes.getRangeDescriptor(), dd);

                REQUIRE_UNARY_FALSE(linRes.hasOperator());
                REQUIRE_UNARY_FALSE(linRes.hasDataVector());

                REQUIRE_THROWS_AS(linRes.getOperator(), Error);
                REQUIRE_THROWS_AS(linRes.getDataVector(), Error);
            }

            THEN("a copy behaves as expected")
            {
                auto linResClone = linRes;

                REQUIRE_NE(&linResClone, &linRes);
                REQUIRE_EQ(linResClone, linRes);
            }

            THEN("the Jacobian and evaluate work as expected")
            {
                Identity<TestType> idOp(dd);
                DataContainer<TestType> dcX(dd);
                dcX = 1;

                REQUIRE_EQ(linRes.getJacobian(dcX), leaf(idOp));
                REQUIRE_UNARY(isApprox(linRes.evaluate(dcX), dcX));
            }
        }
    }
}

TEST_CASE_TEMPLATE("LinearResidual: Testing with just an data vector", TestType, float, double,
                   complex<float>, complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("a descriptor and data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 18, 36;
        VolumeDescriptor dd(numCoeff);

        Vector randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> dc(dd, randomData);

        WHEN("instantiating")
        {
            LinearResidual<TestType> linRes(dc);

            THEN("the residual is as expected")
            {
                REQUIRE_EQ(linRes.getDomainDescriptor(), dd);
                REQUIRE_EQ(linRes.getRangeDescriptor(), dd);

                REQUIRE_UNARY_FALSE(linRes.hasOperator());
                REQUIRE_UNARY(linRes.hasDataVector());

                REQUIRE_UNARY(isApprox(linRes.getDataVector(), dc));
                REQUIRE_THROWS_AS(linRes.getOperator(), Error);
            }

            THEN("a copy behaves as expected")
            {
                auto linResClone = linRes;

                REQUIRE_NE(&linResClone, &linRes);
                REQUIRE_EQ(linResClone, linRes);
            }

            THEN("the Jacobian and evaluate work as expected")
            {
                Identity<TestType> idOp(dd);
                DataContainer<TestType> dcX(dd);
                dcX = 1;

                REQUIRE_EQ(linRes.getJacobian(dcX), leaf(idOp));

                DataContainer<TestType> tmp = dcX - dc;
                REQUIRE_UNARY(isApprox(linRes.evaluate(dcX), tmp));
            }
        }
    }
}

TEST_CASE_TEMPLATE("LinearResidual: Testing with just an operator", TestType, float, double,
                   complex<float>, complex<double>)
{
    GIVEN("descriptors and an operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 33, 55;
        VolumeDescriptor ddDomain(numCoeff);

        IndexVector_t numCoeff2(2);
        numCoeff2 << 18, 36;
        VolumeDescriptor ddRange(numCoeff2);

        MockOperator<TestType> mockOp(ddDomain, ddRange);

        WHEN("instantiating")
        {
            LinearResidual<TestType> linRes(mockOp);

            THEN("the residual is as expected")
            {
                REQUIRE_EQ(linRes.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(linRes.getRangeDescriptor(), ddRange);

                REQUIRE_UNARY(linRes.hasOperator());
                REQUIRE_UNARY_FALSE(linRes.hasDataVector());

                REQUIRE_EQ(linRes.getOperator(), mockOp);
                REQUIRE_THROWS_AS(linRes.getDataVector(), Error);
            }

            THEN("a copy behaves as expected")
            {
                auto linResClone = linRes;

                REQUIRE_NE(&linResClone, &linRes);
                REQUIRE_EQ(linResClone, linRes);
            }

            THEN("the Jacobian and evaluate work as expected")
            {
                DataContainer<TestType> dcX(ddDomain);
                dcX = 1;

                REQUIRE_EQ(linRes.getJacobian(dcX), leaf(mockOp));
                REQUIRE_UNARY(isApprox(linRes.evaluate(dcX), mockOp.apply(dcX)));
            }
        }
    }
}

TEST_CASE_TEMPLATE("LinearResidual: Testing with operator and data", TestType, float, double,
                   complex<float>, complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("an operator and data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 33, 55;
        VolumeDescriptor ddDomain(numCoeff);
        IndexVector_t numCoeff2(2);
        numCoeff2 << 18, 36;
        VolumeDescriptor ddRange(numCoeff2);

        MockOperator<TestType> mockOp(ddDomain, ddRange);

        Vector randomData(ddRange.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> dc(ddRange, randomData);

        WHEN("instantiating")
        {
            LinearResidual<TestType> linRes(mockOp, dc);

            THEN("the residual is as expected")
            {
                REQUIRE_EQ(linRes.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(linRes.getRangeDescriptor(), ddRange);

                REQUIRE_UNARY(linRes.hasOperator());
                REQUIRE_UNARY(linRes.hasDataVector());

                REQUIRE_EQ(linRes.getOperator(), mockOp);
                REQUIRE_UNARY(isApprox(linRes.getDataVector(), dc));
            }

            THEN("a copy behaves as expected")
            {
                auto linResClone = linRes;

                REQUIRE_NE(&linResClone, &linRes);
                REQUIRE_EQ(linResClone, linRes);
            }

            THEN("the Jacobian and evaluate work as expected")
            {
                DataContainer<TestType> dcX(ddDomain);
                dcX = 1;

                REQUIRE_EQ(linRes.getJacobian(dcX), leaf(mockOp));

                DataContainer<TestType> tmp = mockOp.apply(dcX) - dc;
                REQUIRE_UNARY(isApprox(linRes.evaluate(dcX), tmp));
            }
        }
    }
}

TEST_SUITE_END();
