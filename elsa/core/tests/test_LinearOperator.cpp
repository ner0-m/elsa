/**
 * \file test_LinearOperator.cpp
 *
 * \brief Tests for LinearOperator class
 *
 * \author Tobias Lasser - main code
 * \author David Frank - composite tests
 * \author Nikola Dinev - fixes
 */

#include <catch2/catch.hpp>
#include "LinearOperator.h"
#include "VolumeDescriptor.h"

using namespace elsa;

// mock operator, which outputs 1 for apply and 2 for applyAdjoint
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

SCENARIO("Constructing a LinearOperator")
{
    GIVEN("DataDescriptors")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 31, 23;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);

        WHEN("instantiating a LinearOperator")
        {
            LinearOperator linOp(ddDomain, ddRange);

            THEN("the DataDescriptors are as expected")
            {
                REQUIRE(linOp.getDomainDescriptor() == ddDomain);
                REQUIRE(linOp.getRangeDescriptor() == ddRange);
            }

            THEN("the apply* operations throw")
            {
                DataContainer dc(ddDomain);
                REQUIRE_THROWS_AS(linOp.apply(dc), std::logic_error);
                REQUIRE_THROWS_AS(linOp.applyAdjoint(dc), std::logic_error);
            }

            THEN("copies are good")
            {
                auto newOp = linOp;
                REQUIRE(newOp == linOp);
            }
        }
    }
}

SCENARIO("Cloning LinearOperators")
{
    GIVEN("a LinearOperator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 23, 45, 67;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 78, 90;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        LinearOperator linOp(ddDomain, ddRange);

        WHEN("cloning the LinearOperator")
        {
            auto linOpClone = linOp.clone();

            THEN("everything matches")
            {
                REQUIRE(linOpClone.get() != &linOp);
                REQUIRE(*linOpClone == linOp);
            }

            THEN("copies are also identical")
            {
                auto newOp = *linOpClone;
                REQUIRE(newOp == linOp);
            }
        }
    }
}

SCENARIO("Leaf LinearOperator")
{
    GIVEN("a non-adjoint leaf linear operator")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 12, 23;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 34, 45;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        MockOperator mockOp(ddDomain, ddRange);

        auto leafOp = leaf(mockOp);

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE(leafOp.getDomainDescriptor() == ddDomain);
                REQUIRE(leafOp.getRangeDescriptor() == ddRange);
            }
        }

        WHEN("given data")
        {
            DataContainer dcDomain(ddDomain);
            DataContainer dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = leafOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApply[i] == 1);

                auto resultApplyAdjoint = leafOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE(resultApplyAdjoint[i] == 3);
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(leafOp.apply(dcRange), std::invalid_argument);
                REQUIRE_THROWS_AS(leafOp.applyAdjoint(dcDomain), std::invalid_argument);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = leafOp;
            auto assignedOp = leaf(newOp);

            THEN("it should be identical")
            {
                REQUIRE(newOp == leafOp);
                REQUIRE(assignedOp == leaf(newOp));

                assignedOp = newOp;
                REQUIRE(assignedOp == newOp);
            }
        }
    }

    GIVEN("an adjoint linear operator")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 12, 23;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 34, 45;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        MockOperator mockOp(ddDomain, ddRange);

        auto adjointOp = adjoint(mockOp);

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE(adjointOp.getDomainDescriptor() == ddRange);
                REQUIRE(adjointOp.getRangeDescriptor() == ddDomain);
            }
        }

        WHEN("given data")
        {
            DataContainer dcDomain(ddDomain);
            DataContainer dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = adjointOp.apply(dcRange);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApply[i] == 3);

                auto resultApplyAdjoint = adjointOp.applyAdjoint(dcDomain);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE(resultApplyAdjoint[i] == 1);
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(adjointOp.apply(dcDomain), std::invalid_argument);
                REQUIRE_THROWS_AS(adjointOp.applyAdjoint(dcRange), std::invalid_argument);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = adjointOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE(newOp == adjointOp);
                REQUIRE(assignedOp == adjoint(newOp));

                assignedOp = newOp;
                REQUIRE(assignedOp == newOp);
            }
        }
    }
}

SCENARIO("Composite LinearOperator")
{
    GIVEN("an additive composite linear operator")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 45, 67;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 13, 48;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);

        MockOperator op1(ddDomain, ddRange);
        MockOperator op2(ddDomain, ddRange);

        auto addOp = op1 + op2;

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE(addOp.getDomainDescriptor() == ddDomain);
                REQUIRE(addOp.getRangeDescriptor() == ddRange);
            }
        }

        WHEN("given data")
        {
            DataContainer dcDomain(ddDomain);
            DataContainer dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = addOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApply[i] == 2);

                auto resultApplyAdjoint = addOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE(resultApplyAdjoint[i] == 6);
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(addOp.apply(dcRange), std::invalid_argument);
                REQUIRE_THROWS_AS(addOp.applyAdjoint(dcDomain), std::invalid_argument);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = addOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE(newOp == addOp);
                REQUIRE(assignedOp == adjoint(newOp));

                assignedOp = newOp;
                REQUIRE(assignedOp == newOp);
            }
        }
    }

    GIVEN("a multiplicative composite linear operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 47, 69;
        IndexVector_t numCoeff2(2);
        numCoeff2 << 15, 28;
        IndexVector_t numCoeff3(4);
        numCoeff3 << 7, 30, 83, 13;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddMiddle(numCoeff2);
        VolumeDescriptor ddRange(numCoeff3);

        MockOperator op1(ddDomain, ddMiddle);
        MockOperator op2(ddMiddle, ddRange);

        auto multOp = op2 * op1;

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE(multOp.getDomainDescriptor() == ddDomain);
                REQUIRE(multOp.getRangeDescriptor() == ddRange);
            }
        }

        WHEN("given data")
        {
            DataContainer dcDomain(ddDomain);
            DataContainer dcRange(ddRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = multOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApply[i] == 1);

                auto resultApplyAdjoint = multOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE(resultApplyAdjoint[i] == 3);
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(multOp.apply(dcRange), std::invalid_argument);
                REQUIRE_THROWS_AS(multOp.applyAdjoint(dcDomain), std::invalid_argument);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = multOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE(newOp == multOp);
                REQUIRE(assignedOp == adjoint(newOp));

                assignedOp = newOp;
                REQUIRE(assignedOp == newOp);
            }
        }
    }

    GIVEN("a complex composite with multiple leafs and levels")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 38;
        IndexVector_t numCoeff2(1);
        numCoeff2 << 16;
        IndexVector_t numCoeff3(3);
        numCoeff3 << 17, 38, 15;
        VolumeDescriptor ddDomain(numCoeff);
        VolumeDescriptor ddRange(numCoeff2);
        VolumeDescriptor ddFinalRange(numCoeff3);

        MockOperator op1(ddDomain, ddRange);
        MockOperator op2(ddFinalRange, ddRange);
        MockOperator op3(ddRange, ddFinalRange);

        auto compositeOp = (op3 + adjoint(op2)) * op1;

        WHEN("the operator is there")
        {
            THEN("the descriptors are set correctly")
            {
                REQUIRE(compositeOp.getDomainDescriptor() == ddDomain);
                REQUIRE(compositeOp.getRangeDescriptor() == ddFinalRange);
            }
        }

        WHEN("given data")
        {
            DataContainer dcDomain(ddDomain);
            DataContainer dcFinalRange(ddFinalRange);

            THEN("the apply operations return the correct result")
            {
                auto resultApply = compositeOp.apply(dcDomain);
                for (int i = 0; i < resultApply.getSize(); ++i)
                    REQUIRE(resultApply[i] == 4);

                auto resultApplyAdjoint = compositeOp.applyAdjoint(dcFinalRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE(resultApplyAdjoint[i] == 3);
            }

            THEN("the apply operations expect appropriately sized containers")
            {
                REQUIRE_THROWS_AS(compositeOp.apply(dcFinalRange), std::invalid_argument);
                REQUIRE_THROWS_AS(compositeOp.applyAdjoint(dcDomain), std::invalid_argument);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = compositeOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE(newOp == compositeOp);
                REQUIRE(assignedOp == adjoint(newOp));

                assignedOp = newOp;
                REQUIRE(assignedOp == newOp);
            }
        }
    }
}
