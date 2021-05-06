/**
 * @file test_LinearOperator.cpp
 *
 * @brief Tests for LinearOperator class
 *
 * @author Tobias Lasser - main code
 * @author David Frank - composite tests
 * @author Nikola Dinev - fixes
 */

#include "doctest/doctest.h"
#include "LinearOperator.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

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

TEST_SUITE_BEGIN("core");

TEST_CASE("LinearOperator: Testing construction")
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
                REQUIRE_EQ(linOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(linOp.getRangeDescriptor(), ddRange);
            }

            THEN("the apply* operations throw")
            {
                DataContainer dc(ddDomain);
                REQUIRE_THROWS_AS(linOp.apply(dc), LogicError);
                REQUIRE_THROWS_AS(linOp.applyAdjoint(dc), LogicError);
            }

            THEN("copies are good")
            {
                auto newOp = linOp;
                REQUIRE_EQ(newOp, linOp);
            }
        }
    }
}

TEST_CASE("LinearOperator: Testing clone()")
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
                REQUIRE_NE(linOpClone.get(), &linOp);
                REQUIRE_EQ(*linOpClone, linOp);
            }

            THEN("copies are also identical")
            {
                auto newOp = *linOpClone;
                REQUIRE_EQ(newOp, linOp);
            }
        }
    }
}

TEST_CASE("LinearOperator: Testing a leaf LinearOperator")
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
                REQUIRE_EQ(leafOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(leafOp.getRangeDescriptor(), ddRange);
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
                    REQUIRE_EQ(resultApply[i], static_cast<real_t>(1));

                auto resultApplyAdjoint = leafOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<real_t>(3));
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(leafOp.apply(dcRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(leafOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = leafOp;
            auto assignedOp = leaf(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, leafOp);
                REQUIRE_EQ(assignedOp, leaf(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
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
                REQUIRE_EQ(adjointOp.getDomainDescriptor(), ddRange);
                REQUIRE_EQ(adjointOp.getRangeDescriptor(), ddDomain);
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
                    REQUIRE_EQ(resultApply[i], static_cast<real_t>(3));

                auto resultApplyAdjoint = adjointOp.applyAdjoint(dcDomain);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<real_t>(1));
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(adjointOp.apply(dcDomain), InvalidArgumentError);
                REQUIRE_THROWS_AS(adjointOp.applyAdjoint(dcRange), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = adjointOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, adjointOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
            }
        }
    }
}

TEST_CASE("LinearOperator: Testing composite LinearOperator")
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
                REQUIRE_EQ(addOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(addOp.getRangeDescriptor(), ddRange);
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
                    REQUIRE_EQ(resultApply[i], static_cast<real_t>(2));

                auto resultApplyAdjoint = addOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<real_t>(6));
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(addOp.apply(dcRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(addOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = addOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, addOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
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
                REQUIRE_EQ(multOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(multOp.getRangeDescriptor(), ddRange);
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
                    REQUIRE_EQ(resultApply[i], static_cast<real_t>(1));

                auto resultApplyAdjoint = multOp.applyAdjoint(dcRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<real_t>(3));
            }

            THEN("the apply operations care for appropriately sized containers")
            {
                REQUIRE_THROWS_AS(multOp.apply(dcRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(multOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = multOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, multOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
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
                REQUIRE_EQ(compositeOp.getDomainDescriptor(), ddDomain);
                REQUIRE_EQ(compositeOp.getRangeDescriptor(), ddFinalRange);
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
                    REQUIRE_EQ(resultApply[i], static_cast<real_t>(4));

                auto resultApplyAdjoint = compositeOp.applyAdjoint(dcFinalRange);
                for (int i = 0; i < resultApplyAdjoint.getSize(); ++i)
                    REQUIRE_EQ(resultApplyAdjoint[i], static_cast<real_t>(3));
            }

            THEN("the apply operations expect appropriately sized containers")
            {
                REQUIRE_THROWS_AS(compositeOp.apply(dcFinalRange), InvalidArgumentError);
                REQUIRE_THROWS_AS(compositeOp.applyAdjoint(dcDomain), InvalidArgumentError);
            }
        }

        WHEN("copying/assigning")
        {
            auto newOp = compositeOp;
            auto assignedOp = adjoint(newOp);

            THEN("it should be identical")
            {
                REQUIRE_EQ(newOp, compositeOp);
                REQUIRE_EQ(assignedOp, adjoint(newOp));

                assignedOp = newOp;
                REQUIRE_EQ(assignedOp, newOp);
            }
        }
    }
}

TEST_SUITE_END();
