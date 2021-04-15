/**
 * @file test_LinearResidual.cpp
 *
 * @brief Tests for LinearResidual class
 *
 * @author Matthias Wieczorek - main code
 * @author Tobias Lasser - rewrite
 */

#include <catch2/catch.hpp>
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;

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

SCENARIO("Trivial LinearResidual")
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 33, 55;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            LinearResidual linRes(dd);

            THEN("the residual is as expected")
            {
                REQUIRE(linRes.getDomainDescriptor() == dd);
                REQUIRE(linRes.getRangeDescriptor() == dd);

                REQUIRE(linRes.hasOperator() == false);
                REQUIRE(linRes.hasDataVector() == false);

                REQUIRE_THROWS_AS(linRes.getOperator(), Error);
                REQUIRE_THROWS_AS(linRes.getDataVector(), Error);
            }

            THEN("a clone behaves as expected")
            {
                auto linResClone = linRes.clone();

                REQUIRE(linResClone.get() != &linRes);
                REQUIRE(*linResClone == linRes);
            }

            THEN("the Jacobian and evaluate work as expected")
            {
                Identity idOp(dd);
                DataContainer dcX(dd);
                dcX = 1;

                REQUIRE(linRes.getJacobian(dcX) == leaf(idOp));
                REQUIRE(linRes.evaluate(dcX) == dcX);
            }
        }
    }
}

SCENARIO("LinearResidual with just a data vector")
{
    GIVEN("a descriptor and data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 18, 36;
        VolumeDescriptor dd(numCoeff);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer dc(dd, randomData);

        WHEN("instantiating")
        {
            LinearResidual linRes(dc);

            THEN("the residual is as expected")
            {
                REQUIRE(linRes.getDomainDescriptor() == dd);
                REQUIRE(linRes.getRangeDescriptor() == dd);

                REQUIRE(linRes.hasOperator() == false);
                REQUIRE(linRes.hasDataVector() == true);

                REQUIRE(linRes.getDataVector() == dc);
                REQUIRE_THROWS_AS(linRes.getOperator(), Error);
            }

            THEN("a clone behaves as expected")
            {
                auto linResClone = linRes.clone();

                REQUIRE(linResClone.get() != &linRes);
                REQUIRE(*linResClone == linRes);
            }

            THEN("the Jacobian and evaluate work as expected")
            {
                Identity idOp(dd);
                DataContainer dcX(dd);
                dcX = 1;

                REQUIRE(linRes.getJacobian(dcX) == leaf(idOp));
                REQUIRE(linRes.evaluate(dcX) == dcX - dc);
            }
        }
    }
}

SCENARIO("LinearResidual with just an operator")
{
    GIVEN("descriptors and an operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 33, 55;
        VolumeDescriptor ddDomain(numCoeff);

        IndexVector_t numCoeff2(2);
        numCoeff2 << 18, 36;
        VolumeDescriptor ddRange(numCoeff2);

        MockOperator mockOp(ddDomain, ddRange);

        WHEN("instantiating")
        {
            LinearResidual linRes(mockOp);

            THEN("the residual is as expected")
            {
                REQUIRE(linRes.getDomainDescriptor() == ddDomain);
                REQUIRE(linRes.getRangeDescriptor() == ddRange);

                REQUIRE(linRes.hasOperator() == true);
                REQUIRE(linRes.hasDataVector() == false);

                REQUIRE(linRes.getOperator() == mockOp);
                REQUIRE_THROWS_AS(linRes.getDataVector(), Error);
            }

            THEN("a clone behaves as expected")
            {
                auto linResClone = linRes.clone();

                REQUIRE(linResClone.get() != &linRes);
                REQUIRE(*linResClone == linRes);
            }

            THEN("the Jacobian and evaluate work as expected")
            {
                DataContainer dcX(ddDomain);
                dcX = 1;

                REQUIRE(linRes.getJacobian(dcX) == leaf(mockOp));
                REQUIRE(linRes.evaluate(dcX) == mockOp.apply(dcX));
            }
        }
    }
}

SCENARIO("LinearResidual with operator and data")
{
    GIVEN("an operator and data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 33, 55;
        VolumeDescriptor ddDomain(numCoeff);
        IndexVector_t numCoeff2(2);
        numCoeff2 << 18, 36;
        VolumeDescriptor ddRange(numCoeff2);

        MockOperator mockOp(ddDomain, ddRange);

        RealVector_t randomData(ddRange.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer dc(ddRange, randomData);

        WHEN("instantiating")
        {
            LinearResidual linRes(mockOp, dc);

            THEN("the residual is as expected")
            {
                REQUIRE(linRes.getDomainDescriptor() == ddDomain);
                REQUIRE(linRes.getRangeDescriptor() == ddRange);

                REQUIRE(linRes.hasOperator() == true);
                REQUIRE(linRes.hasDataVector() == true);

                REQUIRE(linRes.getOperator() == mockOp);
                REQUIRE(linRes.getDataVector() == dc);
            }

            THEN("a clone behaves as expected")
            {
                auto linResClone = linRes.clone();

                REQUIRE(linResClone.get() != &linRes);
                REQUIRE(*linResClone == linRes);
            }

            THEN("the Jacobian and evaluate work as expected")
            {
                DataContainer dcX(ddDomain);
                dcX = 1;

                REQUIRE(linRes.getJacobian(dcX) == leaf(mockOp));
                REQUIRE(linRes.evaluate(dcX) == mockOp.apply(dcX) - dc);
            }
        }
    }
}
